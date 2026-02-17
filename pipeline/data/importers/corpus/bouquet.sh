#!/usr/bin/env bash
##
# Downloads facebook/bouquet via HF datasets
# Accepts ISO-639-1 codes and maps to FLORES/NLLB tags
#

set -euo pipefail
set -x

echo "###### Downloading BOUQuET corpus"

src="$1"
trg="$2"
output_prefix="$3"
split="$4"
token="${5:-}"

COMPRESSION_CMD="${COMPRESSION_CMD:-pigz}"
ARTIFACT_EXT="${ARTIFACT_EXT:-gz}"

tmp="$(mktemp -d)/bouquet/${split}"
mkdir -p "${tmp}"

src_out_plain="${tmp}/source.txt"
trg_out_plain="${tmp}/target.txt"

########################################
# ISO → FLORES/NLLB tag mapping
########################################

langtag() {
  code="$1"

  case "$code" in
    en) echo "eng_Latn" ;;
    fi) echo "fin_Latn" ;;
    fr) echo "fra_Latn" ;;
    de) echo "deu_Latn" ;;
    es) echo "spa_Latn" ;;
    it) echo "ita_Latn" ;;
    pt) echo "por_Latn" ;;
    nl) echo "nld_Latn" ;;
    sv) echo "swe_Latn" ;;
    da) echo "dan_Latn" ;;
    no) echo "nob_Latn" ;;
    pl) echo "pol_Latn" ;;
    cs) echo "ces_Latn" ;;
    sk) echo "slk_Latn" ;;
    sl) echo "slv_Latn" ;;
    hu) echo "hun_Latn" ;;
    ro) echo "ron_Latn" ;;
    bg) echo "bul_Cyrl" ;;
    ru) echo "rus_Cyrl" ;;
    uk) echo "ukr_Cyrl" ;;
    sr) echo "srp_Cyrl" ;;
    hr) echo "hrv_Latn" ;;
    lt) echo "lit_Latn" ;;
    lv) echo "lav_Latn" ;;
    et) echo "est_Latn" ;;
    el) echo "ell_Grek" ;;
    tr) echo "tur_Latn" ;;
    ar) echo "arb_Arab" ;;
    he) echo "heb_Hebr" ;;
    fa) echo "pes_Arab" ;;
    hi) echo "hin_Deva" ;;
    bn) echo "ben_Beng" ;;
    ta) echo "tam_Taml" ;;
    te) echo "tel_Telu" ;;
    ml) echo "mal_Mlym" ;;
    kn) echo "kan_Knda" ;;
    zh) echo "zho_Hans" ;;
    zh-Hant) echo "zho_Hant" ;;
    ja) echo "jpn_Jpan" ;;
    ko) echo "kor_Hang" ;;
    th) echo "tha_Thai" ;;
    vi) echo "vie_Latn" ;;
    id) echo "ind_Latn" ;;
    ms) echo "zsm_Latn" ;;
    sw) echo "swh_Latn" ;;
    *) echo "$code" ;;   # assume already a flores tag
  esac
}

src_tag=$(langtag "$src")
trg_tag=$(langtag "$trg")

########################################
# Python dataset loader
########################################

python importers/corpus/bouquet.py --pair "$src_tag-$trg_tag" --split "$split"  --token "$token" \
--out_src "$src_out_plain" --out_tgt "$trg_out_plain"

########################################
# Compress
########################################

if [ -e "${src_out_plain}" ] && [ -e "${trg_out_plain}" ]; then
  ${COMPRESSION_CMD} -c "${src_out_plain}" > "${output_prefix}.source.${ARTIFACT_EXT}"
  ${COMPRESSION_CMD} -c "${trg_out_plain}" > "${output_prefix}.target.${ARTIFACT_EXT}"
fi

rm -rf "${tmp}"

echo "###### Done: Downloading BOUQuET corpus"
