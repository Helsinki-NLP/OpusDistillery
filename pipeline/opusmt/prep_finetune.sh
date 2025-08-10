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
train_set_prefix=$3
valid_set_prefix=$4
model_dir=$5
base_model_dir=$6
segmented_input=$7

vocab="${model_dir}/vocab.yml"

#test -v GPUS
#test -v MARIAN
#test -v WORKSPACE

cd "$(dirname "${0}")"
mkdir -p "${model_dir}/tmp"

cp ${base_model_dir}/source.spm ${model_dir}
cp ${base_model_dir}/target.spm ${model_dir}
cp ${base_model_dir}/vocab.yml ${model_dir}

corpus_src="${train_set_prefix}.${src}.gz"
corpus_trg="${train_set_prefix}.${trg}.gz"

echo "### Subword segmentation with SentencePiece"
source_spm_path="${model_dir}/source.spm"
target_spm_path="${model_dir}/target.spm"

# Modify vocab to contain augmentation symbols
python ./add_term_symbols.py \
  --source_spm_model ${source_spm_path} \
  --target_spm_model ${target_spm_path} \
  --yaml_vocab ${vocab} \
  --num_special_symbols 10

if [[ ${segmented_input} == "true" ]]; then
    ln -s "${corpus_src}" "${model_dir}/train.sp.${src}.gz"
    ln -s "${corpus_trg}" "${model_dir}/train.sp.${trg}.gz"
else
    # This is used in case of RAT models, where target side translation are prefixed to the source sentence.
    # Those target language sentences need to be segmented with the target sp model in the case of OPUS-MT models, since
    # even though OPUS-MT models share the vocab, the sp models are different.
    if [[ "$corpus_src" == *append_terms* ]]; then
        python ./segment_rat.py --input "${corpus_src}" --output "${model_dir}/train.sp.${src}.gz" --model1 "${source_spm_path}" --model2 "${target_spm_path}" --terms_appended
    else
        python ./segment_rat.py --input "${corpus_src}" --output "${model_dir}/train.sp.${src}.gz" --model1 "${source_spm_path}" --model2 "${target_spm_path}" --terms_appended
    fi
    
    #TODO: add form of RAT where source-sides of fuzzies are on the source side, and target sides on the target side. This would simplify processing, as both sides can be segmented with just their own sp model
    #pigz -dc "${corpus_src}" | "${MARIAN}/spm_encode" --model "${source_spm_path}" | pigz >"${model_dir}/train.sp.${src}.gz"
    pigz -dc "${corpus_trg}" | "${MARIAN}/spm_encode" --model "${target_spm_path}" | pigz >"${model_dir}/train.sp.${trg}.gz"
fi

# always segment valid set, it's small, and even if it is already segmented, re-segmenting does not change it
valid_src="${valid_set_prefix}.${src}.gz"
valid_trg="${valid_set_prefix}.${trg}.gz"

pigz -dc "${valid_src}" | "${MARIAN}/spm_encode" --model "${source_spm_path}" | pigz >"${model_dir}/valid.sp.${src}.gz"
pigz -dc "${valid_trg}" | "${MARIAN}/spm_encode" --model "${target_spm_path}" | pigz >"${model_dir}/valid.sp.${trg}.gz"