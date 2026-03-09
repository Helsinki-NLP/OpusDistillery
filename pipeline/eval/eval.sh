#!/bin/bash
##
# Evaluate a model.
#

set -x
set -euo pipefail

echo "###### Evaluation of a model"

langpair=$1
res_prefix=${2}
dataset_prefix=${3}
src=$4
trg=$5
trg_langtag=">>${6}<< "
marian=$7
decoder_config=$8
o2m=$9
args=( "${@:10}" )

mkdir -p "$(basename "${res_prefix}")"

echo "### Evaluating dataset: ${dataset_prefix}, pair: ${langpair}, Results prefix: ${res_prefix}"

if [ ! -f "${res_prefix}.${trg}" ]; then
  if [ -s "${dataset_prefix}.${trg}.gz" ]; then
      pigz -dc "${dataset_prefix}.${trg}.gz" > "${res_prefix}.${trg}.ref"
  else
      echo "File ${dataset_prefix}.${trg}.gz is empty. We assume that the dataset ${dataset_prefix} does not exist for the language pair ${langpair}. Creating dummy file and exiting the script."
      touch "${res_prefix}.metrics"
      exit 0
  fi

  if [ $o2m == "True" ]; then # If the model is multitarget, add language tag for decoding
    pigz -dc "${dataset_prefix}.${src}.gz" | sed "s/^/${trg_langtag}/" | tee "${res_prefix}.${src}" | 
      "${marian}"/marian-decoder \
        -c "${decoder_config}" \
        --quiet \
        --quiet-translation \
        --log "${res_prefix}.log" \
        "${args[@]}" | tee "${res_prefix}.${trg}"
  else 
    pigz -dc "${dataset_prefix}.${src}.gz" | tee "${res_prefix}.${src}" |
      "${marian}"/marian-decoder \
        -c "${decoder_config}" \
        --quiet \
        --quiet-translation \
        --log "${res_prefix}.log" \
        "${args[@]}" | tee "${res_prefix}.${trg}"
  fi
else
  echo "Translation already exists. Remove if you want to overwrite!"
fi

if [[ "$dataset_prefix" == *opusmt* && "$res_prefix" == *teacher* ]]; then
  sed -i 's/ //g; s/▁/ /g; s/^ //g' "${res_prefix}.${trg}"
fi

sacrebleu "${res_prefix}.${trg}.ref" -d -f text --score-only -l "${langpair}" -m bleu chrf < "${res_prefix}.${trg}" | tee "${res_prefix}.metrics"
comet-score -s "${res_prefix}.${src}" -t "${res_prefix}.${trg}" -r "${res_prefix}.${trg}.ref" --quiet --only_system | tee -a "${res_prefix}.metrics"

echo "###### Done: Evaluation of a model"