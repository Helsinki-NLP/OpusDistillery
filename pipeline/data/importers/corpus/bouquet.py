#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

from datasets import load_dataset

def load_bouquet_config(lang: str, token: str):
    try:
        return load_dataset("facebook/bouquet", lang, token=token)
    except Exception as e:
        print(f"WARNING: Language config '{lang}' not found in facebook/bouquet")
        print(f"         HF error: {e}")
        return None

def write_empty_outputs(src_path: str, tgt_path: str):
    Path(src_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tgt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(src_path).write_text("", encoding="utf-8")
    Path(tgt_path).write_text("", encoding="utf-8")

def count_sentences(paras: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in paras.values())

def _id_sort_key(x: str):
    """
    Sort IDs like P2, P10, P458 by numeric component when present.
    """
    m = re.search(r"\d+", x or "")
    if not m:
        return (x or "", 0)
    return (x[: m.start()], int(m.group()), x[m.end():])


def _sent_sort_key(row: Dict[str, Any]) -> Tuple:
    """
    Best-effort sentence ordering key.
    - Prefer explicit sent_id if present.
    - Else try to extract a trailing number from uniq_id.
    - Else return a neutral key (keeps stable-ish ordering).
    """
    if "sent_id" in row and row["sent_id"] not in (None, "", "<na>"):
        return ("sent_id", _id_sort_key(str(row["sent_id"])))

    uid = str(row.get("uniq_id", ""))
    m = re.search(r"(\d+)\s*$", uid)
    if m:
        return ("uniq_tailnum", int(m.group()))

    # fallback: no reliable ordering info
    return ("fallback", 0)


def collect_sentence_paragraphs(
    rows: Iterable[Dict[str, Any]],
    src_code: str
) -> Dict[str, List[str]]:
    """
    Returns: { paragraph_id: [sentence1, sentence2, ...] }
    Only keeps rows belonging to the (src_code -> tgt_code) pair.
    """

    buckets: DefaultDict[str, List[Tuple[Tuple, str]]] = defaultdict(list)

    for r in rows:
        if r.get("level") != "sentence_level":
            continue

        # IMPORTANT: restrict to the requested pair
        if r.get("src_lang") != src_code:
            continue

        text = r.get("src_text")

        pid = r.get("par_id") or r.get("uniq_id")
        if not pid:
            continue

        text = text.replace("\n", " ").strip()
        if not text:
            continue

        buckets[str(pid)].append((_sent_sort_key(r), text))

    out: Dict[str, List[str]] = {}
    for pid in sorted(buckets.keys(), key=_id_sort_key):
        sents = buckets[pid]
        sents.sort(key=lambda x: x[0])
        out[pid] = [t for _, t in sents]

    return out


def write_sentence_paragraphs(paras: Dict[str, List[str]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pids = sorted(paras.keys(), key=_id_sort_key)

    with out_path.open("w", encoding="utf-8") as out:
        for i, pid in enumerate(pids):
            for sent in paras[pid]:
                out.write(sent + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs="+", required=True,
                    help="Language pairs like eng_Latn-hun_Latn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--token", required=True)
    ap.add_argument("--out_src", required=True)
    ap.add_argument("--out_tgt", required=True)
    args = ap.parse_args()

    pair = args.pair[0]
    src, tgt = pair.split("-")

    print("Loading Bouquet from HF…")

    # ---- load SRC config ----
    ds_src = load_bouquet_config(src, args.token)
    if ds_src is None:
        print(f"WARNING: Missing src language '{src}' → writing empty files")
        write_empty_outputs(args.out_src, args.out_tgt)
        return

    if args.split not in ds_src:
        print(f"WARNING: Split '{args.split}' not in src language '{src}' → empty files")
        write_empty_outputs(args.out_src, args.out_tgt)
        return

    # ---- load TGT config ----
    ds_tgt = load_bouquet_config(tgt, args.token)
    if ds_tgt is None:
        print(f"WARNING: Missing tgt language '{tgt}' → writing empty files")
        write_empty_outputs(args.out_src, args.out_tgt)
        return

    if args.split not in ds_tgt:
        print(f"WARNING: Split '{args.split}' not in tgt language '{tgt}' → empty files")
        write_empty_outputs(args.out_src, args.out_tgt)
        return

    print(f"Processing {args.split} {pair} (sentence_level)")

    data_src = ds_src[args.split]
    data_tgt = ds_tgt[args.split]

    src_paras = collect_sentence_paragraphs(data_src, src)
    tgt_paras = collect_sentence_paragraphs(data_tgt, tgt)

    write_sentence_paragraphs(src_paras, args.out_src)
    write_sentence_paragraphs(tgt_paras, args.out_tgt)

    print(f"Wrote src → {args.out_src}")
    print(f"Wrote tgt → {args.out_tgt}")

if __name__ == "__main__":
    main()
