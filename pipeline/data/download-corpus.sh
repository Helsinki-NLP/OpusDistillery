#!/bin/bash
##
# Downloads parallel dataset
#

set -x
set -euo pipefail

test -v SRC
test -v TRG


dataset=$1
output_prefix=$2
src_lang=$3
trg_lang=$4
token=$5

echo "###### Downloading dataset ${dataset}"

cd "$(dirname "${0}")"

dir=$(dirname "${output_prefix}")
mkdir -p "${dir}"

name=${dataset#*_}
type=${dataset%%_*}
bash "importers/corpus/${type}.sh" "${src_lang}" "${trg_lang}" "${output_prefix}" "${name}" "${token}" 

echo "###### Done: Downloading dataset ${dataset}"
