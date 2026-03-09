#!/usr/bin/env bash
set -euo pipefail

modelname="$1"
hf_dir="$2"
HF_TOKEN="$3"
hf_owner="Helsinki-NLP" # HARDCODED; adjust as needed

pick_python_with_hf_deps() {
    local py="$1"
    "$py" - <<'PY' >/dev/null 2>&1
mods = ("numpy", "torch", "transformers", "huggingface_hub")
for mod in mods:
    __import__(mod)
PY
}

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN=""
fi

if [[ -z "$PYTHON_BIN" ]] && [[ -d "/conda-envs" ]]; then
    for py in /conda-envs/*/bin/python; do
        [[ -x "$py" ]] || continue
        if pick_python_with_hf_deps "$py"; then
            PYTHON_BIN="$py"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

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
echo "Python    : $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import os
import sys

print("sys.executable:", sys.executable)
print("sys.version   :", sys.version.replace("\n", " "))
print("CONDA_PREFIX  :", os.environ.get("CONDA_PREFIX", ""))

for mod in ("numpy", "torch", "transformers", "huggingface_hub"):
    try:
        __import__(mod)
    except Exception as exc:
        print(f"IMPORT_FAIL {mod}: {exc}")
        raise
    else:
        print(f"IMPORT_OK   {mod}")
PY

# Convert Marian -> HF folder
if test -f "$hf_dir/vocab.json"; then
    echo "HF model already exists. Skipping. Delete if you want to overwrite."
else
    ln -s $model_dir/../vocab/vocab.spm $model_dir
    "$PYTHON_BIN" pipeline/hf/convert_to_pytorch.py --src "$model_dir" --dest "$hf_dir"
fi

# Upload to HF
"$PYTHON_BIN" pipeline/hf/push_to_hf.py \
  --token "$HF_TOKEN" \
  --repo_id "$repo_id" \
  --local_dir "$hf_dir" \
  --private

echo "Done."
