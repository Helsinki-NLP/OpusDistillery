#!/bin/bash
#
# Merges translation outputs into a dataset
#

set -x
set -euo pipefail


src_prefix=$1
trg_prefix=$2
aln_prefix=$3
output_src=$4
output_trg=$5
output_aln=$6
output_src_omit=$7
output_trg_omit=$8
output_aln_omit=$9

echo "### Collecting annotations"
cat "${src_prefix}".[0-9]* >"${output_src}"
cat "${trg_prefix}".[0-9]* >"${output_trg}"
cat "${aln_prefix}".[0-9]* >"${output_aln}"

echo "### Comparing number of sentences in source and artificial target files"
src_len=$(pigz -dc "${output_src}" | wc -l)
trg_len=$(pigz -dc "${output_trg}" | wc -l)
aln_len=$(pigz -dc "${output_aln}" | wc -l)

if [ "${src_len}" != "${trg_len}" ] || [ ${trg_len} != "${aln_len}" ]; then
  echo "### Error: lengths of source, target and alignment files are not identical"
  exit 1
fi

echo "### Filtering annotated sentences"
paste <(pigz -dc "${output_src}") <(pigz -dc "${output_trg}") <(pigz -dc "${output_aln}") |
grep "augmentsymbol0" | tee >(cut -f 1 > "${output_src_omit}") | tee >(cut -f 2 > "${output_trg_omit}") | cut -f 3 > "${output_aln_omit}"
