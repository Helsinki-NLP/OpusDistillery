#!/bin/bash
##
# Splits a parallel dataset with alignment
#

set -x
set -euo pipefail

corpus_src=$1
corpus_trg=$2
alignment=$3
output_dir=$4
length=$5

mkdir -p "${output_dir}"
pigz -dc "${corpus_src}" |  split -d -l ${length} --filter='gzip > $FILE.gz' - "${output_dir}/file.src."
pigz -dc "${corpus_trg}" |  split -d -l ${length} --filter='gzip > $FILE.gz' - "${output_dir}/file.trg."
pigz -dc "${alignment}" |  split -d -l ${length} --filter='gzip > $FILE.gz' - "${output_dir}/file.aln."
