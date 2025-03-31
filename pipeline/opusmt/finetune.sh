#!/bin/bash
##
# Finetune model with term data.
#

set -x
set -euo pipefail

echo "###### Finetune am model"

# On LUMI, having CUDA_VISIBLE_DEVICES set causes a segfault when using multiple GPUs
unset CUDA_VISIBLE_DEVICES

src=$1
trg=$2
train_set_prefix=$3
valid_set_prefix=$4
output_model=$5
model_dir=$(dirname ${output_model})
base_model=$6
base_model_dir=$(dirname ${base_model})
best_model_metric=$7
threads=$8
learn_rate=$9
epochs=${10}
extra_params=( "${@:11}" )
vocab="${model_dir}/vocab.yml"

#test -v GPUS
#test -v MARIAN
#test -v WORKSPACE

cd "$(dirname "${0}")"
mkdir -p "${model_dir}/tmp"

cp ${base_model_dir}/source.spm ${model_dir}
cp ${base_model_dir}/target.spm ${model_dir}
cp ${base_model_dir}/vocab.yml ${model_dir}
cp ${base_model} ${model_dir}/model_unfinished.npz

optimizer_file=${base_model_dir}/model_unfinished.npz.optimizer.npz
if [ -e "${optimizer_file}" ]; then
  cp ${optimizer_file} ${model_dir}
  cp ${base_model_dir}/model_unfinished.npz.progress.yml ${model_dir}
fi

corpus_src="${train_set_prefix}.${src}.gz"
corpus_trg="${train_set_prefix}.${trg}.gz"

spm_train_set_prefix="${model_dir}/train.sp"

echo "### Subword segmentation with SentencePiece"
source_spm_path="${model_dir}/source.spm"
target_spm_path="${model_dir}/target.spm"


pigz -dc "${corpus_src}" | "${MARIAN}/spm_encode" --model "${source_spm_path}" | pigz >"${model_dir}/train.sp.${src}.gz"
pigz -dc "${corpus_trg}" | "${MARIAN}/spm_encode" --model "${target_spm_path}" | pigz >"${model_dir}/train.sp.${trg}.gz"

valid_src="${valid_set_prefix}.${src}.gz"
valid_trg="${valid_set_prefix}.${trg}.gz"

spm_valid_set_prefix="${model_dir}/valid.sp"

pigz -dc "${valid_src}" | "${MARIAN}/spm_encode" --model "${source_spm_path}" | pigz >"${model_dir}/valid.sp.${src}.gz"
pigz -dc "${valid_trg}" | "${MARIAN}/spm_encode" --model "${target_spm_path}" | pigz >"${model_dir}/valid.sp.${trg}.gz"


# Modify vocab to contain three augmentation symbols
# TODO: generalize this to work with a list of symbols, for combined RAT and term work
# python ./add_term_symbols.py \
#  --source_spm_model ${model_dir}/source.spm \
#  --target_spm_model ${model_dir}/target.spm \
#  --yaml_vocab ${vocab}

all_model_metrics=(chrf ce-mean-words bleu-detok)

echo "### Training ${model_dir}"

# if doesn't fit in RAM, remove --shuffle-in-ram and add --shuffle batches
# Continued training can also be done by using --pretrained-model, but that's more complex,
# since that doesn't preserve the architecture.

"${MARIAN}"/marian \
  --type transformer \
  --model "${model_dir}/model_unfinished.npz" \
  --train-sets "${spm_train_set_prefix}".{"${src}","${trg}"}.gz \
  --valid-sets "${spm_valid_set_prefix}".{"${src}","${trg}"}.gz \
  -T "${model_dir}/tmp" \
  --shuffle-in-ram \
  --vocabs "${vocab}" "${vocab}" \
  -w "${WORKSPACE}" \
  --devices ${GPUS} \
  --beam-size 6 \
  --sharding local \
  --lr-report \
  --learn-rate "${learn_rate}" \
  --sync-sgd \
  --valid-metrics "${best_model_metric}" ${all_model_metrics[@]/$best_model_metric} \
  --valid-translation-output "${model_dir}/devset.out" \
  --quiet-translation \
  --overwrite \
  --after "${epochs}e" \
  --no-restore-corpus \
  --valid-reset-stalled \
  --log "${model_dir}/train.log" \
  --valid-log "${model_dir}/valid.log" \
  --valid-freq 2000u \
  "${extra_params[@]}"


ln -f "${model_dir}/model_unfinished.npz" "${output_model}"  
ln -f "${model_dir}/model_unfinished.npz.decoder.yml" "${output_model}.decoder.yml"  
ln -f "${model_dir}/model_unfinished.npz.optimizer.npz" "${output_model}.optimizer.npz"  

echo "### Model training is completed: ${model_dir}"
echo "###### Done: Training a model"
