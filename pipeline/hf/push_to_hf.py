#!/usr/bin/env python3
"""
Push a local model directory to the Hugging Face Hub.

Example:
  python push_model_to_hf.py \
    --token "$HF_TOKEN" \
    --repo_id "your-username/my-model" \
    --local_dir "./checkpoints/my-model" \
    --private

Notes:
- local_dir should contain your model artifacts (e.g., config.json, tokenizer files, *.safetensors / *.bin).
- This script uploads everything in local_dir (except common junk patterns).
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


IGNORE_PATTERNS = [
    "*.tmp",
    "*.log",
    "*.ckpt",
    "*.pt",          # remove if you actually store weights as .pt
    ".DS_Store",
    "__pycache__/",
    "*.pyc",
    ".ipynb_checkpoints/",
    "wandb/",
    "runs/",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Hugging Face access token")
    ap.add_argument("--repo_id", required=True, help='Hub repo like "username/model-name"')
    ap.add_argument("--local_dir", required=True, help="Path to local model directory to upload")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--revision", default="main", help="Branch name (default: main)")
    ap.add_argument("--commit_message", default="Upload model", help="Commit message")
    args = ap.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise SystemExit(f"local_dir does not exist or is not a directory: {local_dir}")

    # Optional sanity checks (won't block upload, just warns)
    expected = ["generation_config.json"]
    missing = [f for f in expected if not (local_dir / f).exists()]
    if missing:
        print(f"WARNING: missing expected file(s) in {local_dir}: {', '.join(missing)}")

    token = args.token.strip()
    print("Token length:", len(args.token))
    print("Token starts with:", args.token[:6])
    api = HfApi(token=token)

    # Create the repo if needed
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload the whole folder
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo="",
        commit_message=args.commit_message,
        revision=args.revision,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"Done! Pushed {local_dir} to https://huggingface.co/{args.repo_id}")
    print(f"Edit model card and make your repo public!")


if __name__ == "__main__":
    main()
