#!/usr/bin/env bash
set -euo pipefail

modelname="$1"
hf_dir="$2"
HF_TOKEN="$3"
hf_owner="odegiber" # HARDCODED; adjust as needed

# Create hf_dir if needed
mkdir -p "$hf_dir"

# Derive pair/model name from parent folder of model_dir
# e.g. /.../kor-eng/student_tf -> kor-eng
model_dir="$(dirname "$modelname")"
pair="$(basename "$(dirname "$model_dir")")"
result="opus-mt_tiny_${pair}"

repo_id="${hf_owner}/${result}"

echo "Model dir : $model_dir"
echo "HF dir    : $hf_dir"
echo "Repo ID   : $repo_id"

# Convert Marian -> HF folder
if test -f "$hf_dir/vocab.json"; then
    echo "HF model already exists. Skipping. Delete if you want to overwrite."
else
    python pipeline/hf/convert_to_pytorch.py --src "$model_dir" --dest "$hf_dir"
fi

# Upload to HF
python pipeline/hf/push_to_hf.py \
  --token "$HF_TOKEN" \
  --repo_id "$repo_id" \
  --local_dir "$hf_dir" \
  --private

echo "Done."
