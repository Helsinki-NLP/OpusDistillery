#!/bin/bash
##
# Translates generating n-best lists as output
#

set -x
set -euo pipefail

# On LUMI, having CUDA_VISIBLE_DEVICES set causes a segfault when using multiple GPUs
unset CUDA_VISIBLE_DEVICES

filein=$1
fileout=$2
modelname=$3
modeldir=$4
src_lang=$5
trg_lang=$6
modelclass=$7
langinfo=$8
prompt=$9
langtags=${10}
config=${11}
batch_size=${12}
token=${13}
logfile=${14}

echo "### Translation started"

accelerate launch --mixed_precision="fp16" pipeline/translate/translate_hf.py \
            "${filein}" "${fileout}" "${modelname}" "${modeldir}" "${src_lang}" "${trg_lang}" \
            "${modelclass}" "${langinfo}" "${prompt}" "${langtags}" "${config}" "${batch_size}" "${token}" "${logfile}" >> "${logfile}.hf" 2>&1
        
echo "### Translations done!"

echo "### Sorting started"

cat $fileout.rank*.tmp | sort -n -t '|' -k 1,1 | uniq > $fileout
rm $fileout.rank*.tmp

echo "### Sorting done!"