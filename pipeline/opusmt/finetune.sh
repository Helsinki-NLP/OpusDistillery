#!/bin/bash
##
# Finetune model with term data.
#

set -x
set -euo pipefail

echo "###### Finetune an OPUS-MT model"

# On LUMI, having CUDA_VISIBLE_DEVICES set causes a segfault when using multiple GPUs
unset CUDA_VISIBLE_DEVICES

src=$1
trg=$2
output_model=$3
model_dir=$(dirname ${output_model})
base_model=$4
base_model_dir=$(dirname ${base_model})
best_model_metric=$5
threads=$6
learn_rate=$7
epochs=$8
segmented_input=$9
vocab=${10}
spm_train_set_src=${11}
spm_train_set_trg=${12}
spm_valid_set_src=${13}
spm_valid_set_trg=${14}
extra_params=( "${@:15}" )
#vocab="${model_dir}/vocab.yml"

#test -v GPUS
#test -v MARIAN
#test -v WORKSPACE

cd "$(dirname "${0}")"
mkdir -p "${model_dir}/tmp"

cp ${base_model} ${model_dir}/model_unfinished.npz

optimizer_file=${base_model_dir}/model_unfinished.npz.optimizer.npz
if [ -e "${optimizer_file}" ]; then
  cp ${optimizer_file} ${model_dir}
  cp ${base_model_dir}/model_unfinished.npz.progress.yml ${model_dir}
fi

all_model_metrics=(chrf ce-mean-words bleu-detok)

# disable early stopping, valid set does not reflect specialized performance that we are fine-tuning for, training stops after n epochs.
early_stopping=1000

echo "### Training ${model_dir}"

# if doesn't fit in RAM, remove --shuffle-in-ram and add --shuffle batches
# Continued training can also be done by using --pretrained-model, but that's more complex,
# since that doesn't preserve the architecture.

"${MARIAN}"/marian \
  --type transformer \
  --model "${model_dir}/model_unfinished.npz" \
  --train-sets "${spm_train_set_src}" "${spm_train_set_trg}" \
  --valid-sets "${spm_valid_set_src}" "${spm_valid_set_trg}" \
  -T "${model_dir}/tmp" \
  --shuffle-in-ram \
  --vocabs "${vocab}" "${vocab}" \
  --mini-batch 256 \
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
  --early-stopping "${early_stopping}" \
  --no-restore-corpus \
  --valid-reset-stalled \
  --log "${model_dir}/train.log" \
  --valid-log "${model_dir}/valid.log" \
  --valid-freq 5000u \
  "${extra_params[@]}"


ln -f "${model_dir}/model_unfinished.npz" "${output_model}"  
ln -f "${model_dir}/model_unfinished.npz.decoder.yml" "${output_model}.decoder.yml"  
ln -f "${model_dir}/model_unfinished.npz.optimizer.npz" "${output_model}.optimizer.npz"  

echo "### Model training is completed: ${model_dir}"
echo "###### Done: Training a model"
