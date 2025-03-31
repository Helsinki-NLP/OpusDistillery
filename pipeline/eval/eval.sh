#!/bin/bash
##
# Evaluate a model.
#

set -x
set -euo pipefail

echo "###### Evaluation of a model"

res_prefix=$1
dataset_prefix=$2
src=$3
trg=$4
marian_decoder=$5
decoder_config=$6
model_dir=$(dirname "${decoder_config}")
model_step=$(basename "${model_dir}")
args=( "${@:7}" )

mkdir -p "$(basename "${res_prefix}")"

raw_source_file="${dataset_prefix}.${src}.gz"

output_file="${res_prefix}.${trg}"

if [[ "${model_step}" == *opus* ]]; then
  source_spm_path="${model_dir}/source.spm"
  target_spm_path="${model_dir}/target.spm"
  source_file="${dataset_prefix}.sp.${src}.gz"
  pigz -dc "${raw_source_file}" | "${MARIAN}/spm_encode" --model "${source_spm_path}" | pigz >"${source_file}" 
  output_file="${res_prefix}.sp.${trg}"
else
  source_file="${raw_source_file}"
fi

echo "### Evaluating dataset: ${dataset_prefix}, pair: ${src}-${trg}, Results prefix: ${res_prefix}"

pigz -dc "${dataset_prefix}.${trg}.gz" > "${res_prefix}.${trg}.ref"

pigz -dc "${source_file}" |
  "${marian_decoder}" \
    -c "${decoder_config}" \
    --quiet \
    --quiet-translation \
    --log "${res_prefix}.log" \
    "${args[@]}" > "${output_file}"
 
if [[ "${model_step}" == *opus* ]]; then
  "${MARIAN}/spm_decode" --model "${target_spm_path}" < "${output_file}" > "${res_prefix}.${trg}" 
fi
sacrebleu "${res_prefix}.${trg}.ref" -d -f text --score-only -l "${src}-${trg}" -m bleu chrf < "${res_prefix}.${trg}" > "${res_prefix}.metrics"

unzipped_source="${res_prefix}.${src}"
pigz -dc "${raw_source_file}" > "${unzipped_source}"
comet-score -s "${unzipped_source}" -t "${res_prefix}.${trg}" -r "${res_prefix}.${trg}.ref" --only_system >> "${res_prefix}.metrics"

echo "###### Done: Evaluation of a model"
