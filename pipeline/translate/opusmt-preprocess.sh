#!/bin/bash
##
# Applies OPUS-MT preprocessing to corpus
#

set -x
set -euo pipefail


source_file=$1
opusmt_model=$2
spm_name=$3
spm_encoder=$4
o2m_teacher=$5
src=$6
o2m_backward=$7
export PATH=$PATH:$(dirname ${spm_encoder})
model_dir=$(dirname $opusmt_model)

# When splits are preprocessed, different models need different preprocessing,
# so model index is given. Check for unset parameter.
if [ $# -ge 8 ]; then
    model_index_suffix=".$8"
else
    model_index_suffix=""
fi

# If the model is the best available, we need to check again whether the model is multilingual at the target side
if [ $o2m_teacher == "best" ]; then   
    o2m_teacher=$(cat ${model_dir}/one2many.txt)  # Read the content of the file
    echo "Model is multilingual to the target side: $o2m_teacher"
fi

if [ $o2m_backward == "best" ]; then   
    o2m_backward=$(cat ${model_dir}/one2many.txt)  # Read the content of the file
    echo "Model is multilingual to the target side: $o2m_backward"
fi

if [ "${source_file##*.}" == "gz" ]; then #This applies when scoring
    echo "source file is gzipped"
    if [ $o2m_teacher == "True" ]; then
        zcat $1 |  sed "s/^>>.*<< //" | pipeline/translate/preprocess.sh "${model_dir}/${spm_name}" | gzip > ${source_file%%.gz}${model_index_suffix}.opusmt.gz
    elif [ "${o2m_backward}" == "True" ]; then # In case the are multiple source languages, we need to add langtag for scoring with the backward model
        zcat $1 | pipeline/translate/preprocess.sh "${model_dir}/${spm_name}" | sed "s/^/>>${src}<< /" | gzip > ${source_file%%.gz}${model_index_suffix}.opusmt.gz
    else
        zcat $1 | pipeline/translate/preprocess.sh "${model_dir}/${spm_name}" | gzip > ${source_file%%.gz}${model_index_suffix}.opusmt.gz
    fi
else
    echo "source file is not gzipped"
    out_file=$1${model_index_suffix}.opusmt
    if [ $o2m_teacher == "True" ]; then
        # Get language tag
        target_lang_token="$(head -n 1 "$source_file" | egrep -o -E "^>>[a-z]{2,3}<< ")"
        echo "Target language token: $target_lang_token"
        echo "Started tokenization..."
        sed "s/^$target_lang_token//" $source_file | pipeline/translate/preprocess_2.sh "${model_dir}/${spm_name}" > $source_file${model_index_suffix}.opusmt
        echo "Tokenization done!"
        echo "Adding target language tag again..."
        sed -i "s/^/$target_lang_token/" $source_file${model_index_suffix}.opusmt
        echo "Done!"
    else
        pipeline/translate/preprocess.sh "${model_dir}/${spm_name}" < $1 > $1${model_index_suffix}.opusmt
    fi
fi