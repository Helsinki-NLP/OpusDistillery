#!/bin/bash
##
# Generates alignment and lexical shortlist for a corpus.
#

set -x
set -euo pipefail

echo "###### Generating alignments and shortlist"
test -v MARIAN
test -v BIN
test -v SRC
test -v TRG

corpus_prefix=$1
vocab_path=$2
output_dir=$3
o2m_student=$4
threads=$5

cd "$(dirname "${0}")"

mkdir -p "${output_dir}"
dir="${output_dir}/tmp"
mkdir -p "${dir}"

corpus_src="${corpus_prefix}.${SRC}.gz"
corpus_trg="${corpus_prefix}.${TRG}.gz"

test -s "${dir}/cleaned_empty_lines" || {
  echo "### Removing empty target lines + escaping |||"

  paste <(pigz -dc "${corpus_src}" | tr -d '\r') \
        <(pigz -dc "${corpus_trg}" | tr -d '\r') |
  awk -F $'\t' -v OFS=$'\t' '
    {
      # drop whitespace-only src/trg
      if ($1 !~ /[^[:space:]]/ || $2 !~ /[^[:space:]]/) next

      # escape any occurrence of ||| inside either sentence
      gsub(/\|\|\|/, "<PIPE3>", $1)
      gsub(/\|\|\|/, "<PIPE3>", $2)

      print
    }
  ' > "${dir}/corpus_dedup"

  echo "### Splitting corpus back into source and target files and overwriting source and target files"
  cut -f1 "${dir}/corpus_dedup" | pigz -c > "${corpus_src}"
  cut -f2 "${dir}/corpus_dedup" | pigz -c > "${corpus_trg}"

  rm -f "${dir}/corpus_dedup"
  printf 'ok\n' > "${dir}/cleaned_empty_lines"
}


echo "### Subword segmentation with SentencePiece"
test -s "${dir}/corpus.spm.${SRC}.gz" ||
  pigz -dc "${corpus_src}" | sed -E "s/^>>[a-z]{3}<< //" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_path}" |
  pigz >"${dir}/corpus.spm.${SRC}.gz"
test -s "${dir}/corpus.spm.${TRG}.gz" ||
  pigz -dc "${corpus_trg}" |
  parallel --no-notice --pipe -k -j "${threads}" --block 50M "${MARIAN}/spm_encode" --model "${vocab_path}" |
  pigz >"${dir}/corpus.spm.${TRG}.gz"

echo "### Creating merged corpus"
test -s "${output_dir}/corpus.aln.gz" || test -s "${dir}/corpus_spm" ||
  paste <(pigz -dc "${dir}/corpus.spm.${SRC}.gz") <(pigz -dc "${dir}/corpus.spm.${TRG}.gz") |
  sed 's/\t/ ||| /' >"${dir}/corpus_spm"

echo "### Training alignments"
test -s "${output_dir}/corpus.aln.gz" || test -s "${dir}/align.s2t" || test -s "${dir}/align.t2s" ||
  "eflomal-align" -i "${dir}/corpus_spm" -f "${dir}/align.s2t" -r "${dir}/align.t2s" -m 3
  
echo "### Symmetrizing alignments"
test -s "${output_dir}/corpus.aln.gz" ||
  "${BIN}/atools" -i "${dir}/align.s2t" -j "${dir}/align.t2s" -c grow-diag-final-and |
  pigz >"${output_dir}/corpus.aln.gz"

echo "### Creating shortlist"
test -s "${dir}/lex.s2t.gz" ||
  "${BIN}/extract_lex" \
    "${dir}/corpus.spm.${TRG}.gz" \
    "${dir}/corpus.spm.${SRC}.gz" \
    "${output_dir}/corpus.aln.gz" \
    "${dir}/lex.s2t" \
    "${dir}/lex.t2s"
test -s "${dir}/lex.s2t" && pigz "${dir}/lex.s2t"

echo "### Shortlist pruning"
test -s "${dir}/vocab.txt" ||
  "${MARIAN}/spm_export_vocab" --model="${vocab_path}" --output="${dir}/vocab.txt"
test -s "${output_dir}/lex.s2t.pruned.gz" ||
  pigz -dc "${dir}/lex.s2t.gz" |
  grep -v NULL |
  python3 "prune_shortlist.py" 100 "${dir}/vocab.txt" |
  pigz >"${output_dir}/lex.s2t.pruned.gz"

echo "### Deleting tmp dir"
rm -rf "${dir}"

# If there are language tags, we need to modify the alignments by adding index 1 to every source token
if [ $o2m_student == "True" ]; then

    echo "###### Correcting alignments taking into account language tags"
    pigz -dc "${output_dir}/corpus.aln.gz" | parallel --no-notice --pipe -k -j "${threads}" --block 50M \
    'sed -E "s/([0-9]+)-([0-9]+)/echo \$((\1+1))-\2/ge" | sed "s/echo //g"'| gzip > "${output_dir}/corpus.aln.fixed.gz"
    mv "${output_dir}/corpus.aln.fixed.gz" "${output_dir}/corpus.aln.gz"
fi
echo "###### Done: Generating alignments and shortlist"
