#!/usr/bin/env bash
##
# Downloads FLORES+ from Hugging Face
# Dataset split can be "dev" or "devtest"
#
# Usage:
#   ./download_flores_plus.sh <src> <trg> <output_prefix> <dataset> [token]
##

set -x
set -euo pipefail

echo "###### Downloading FLORES+ corpus from Hugging Face"

src=$1
trg=$2
output_prefix=$3
dataset=$4
token="${5:-}"

COMPRESSION_CMD="${COMPRESSION_CMD:-pigz}"
ARTIFACT_EXT="${ARTIFACT_EXT:-gz}"

if [[ "${dataset}" != "dev" && "${dataset}" != "devtest" ]]; then
  echo "ERROR: dataset must be 'dev' or 'devtest', got: ${dataset}" >&2
  exit 1
fi

tmp="$(mktemp -d)"
mkdir -p "${tmp}"

python3 - "${src}" "${trg}" "${dataset}" "${tmp}" "${token}" <<'PY'
import sys
from pathlib import Path

src = sys.argv[1]
trg = sys.argv[2]
split = sys.argv[3]
tmpdir = Path(sys.argv[4])
token = sys.argv[5] or None

def normalize_code(code: str):
    special = {
        "zh": ("zho", "Hans"),
        "zh-Hans": ("zho", "Hans"),
        "zh-Hant": ("zho", "Hant"),
        "sw": ("swh", None),
    }
    if code in special:
        return special[code]

    if "-" in code:
        parts = code.split("-")
        lang = parts[0]
        script = parts[1] if len(parts) > 1 else None
    else:
        lang = code
        script = None

    try:
        from mtdata.iso import iso3_code
        iso3 = iso3_code(lang, fail_error=True)
    except Exception as e:
        raise SystemExit(f"Could not map language code '{code}' to ISO-639-3: {e}")

    return iso3, script

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets`.\n"
        f"Original error: {e}"
    )

src_iso3, src_script = normalize_code(src)
trg_iso3, trg_script = normalize_code(trg)

try:
    load_kwargs = {
        "path": "openlanguagedata/flores_plus",
        "split": split,
    }
    if token is not None:
        load_kwargs["token"] = token

    ds = load_dataset(**load_kwargs)
except Exception as e:
    raise SystemExit(
        "Failed to load openlanguagedata/flores_plus.\n"
        "Pass a HF token as the 5th argument or login with `huggingface-cli login`.\n"
        f"Original error: {e}"
    )

def pick_rows(ds, iso3, script=None):
    rows = [r for r in ds if r.get("iso_639_3") == iso3]
    if script is not None:
        rows2 = [r for r in rows if r.get("iso_15924") == script]
        if rows2:
            return rows2
    return rows

src_rows = pick_rows(ds, src_iso3, src_script)
trg_rows = pick_rows(ds, trg_iso3, trg_script)

src_out = tmpdir / "source.txt"
trg_out = tmpdir / "target.txt"

if src_rows and trg_rows:
    src_by_id = {str(r["id"]): r["text"] for r in src_rows}
    trg_by_id = {str(r["id"]): r["text"] for r in trg_rows}
    common_ids = sorted(set(src_by_id) & set(trg_by_id), key=lambda x: int(x))

    if common_ids:
        with src_out.open("w", encoding="utf-8") as fs, trg_out.open("w", encoding="utf-8") as ft:
            for sid in common_ids:
                fs.write(src_by_id[sid].rstrip("\n") + "\n")
                ft.write(trg_by_id[sid].rstrip("\n") + "\n")
    else:
        src_out.touch()
        trg_out.touch()
        print(f"No overlapping ids found for {src} -> {trg} in split={split}", file=sys.stderr)
else:
    src_out.touch()
    trg_out.touch()
    print(
        f"Created empty files because dataset entries do not exist for "
        f"{src} ({src_iso3},{src_script}) or {trg} ({trg_iso3},{trg_script}) "
        f"in split={split}",
        file=sys.stderr,
    )
PY

if [[ -s "${tmp}/source.txt" && -s "${tmp}/target.txt" ]]; then
  ${COMPRESSION_CMD} -c "${tmp}/source.txt" > "${output_prefix}.source.${ARTIFACT_EXT}"
  ${COMPRESSION_CMD} -c "${tmp}/target.txt" > "${output_prefix}.target.${ARTIFACT_EXT}"
else
  touch "${output_prefix}.source.${ARTIFACT_EXT}"
  touch "${output_prefix}.target.${ARTIFACT_EXT}"
  echo "Fake touch files created since dataset doesn't exist: ${output_prefix}.source.${ARTIFACT_EXT}"
fi

rm -rf "${tmp}"

echo "###### Done: Downloading FLORES+ corpus"