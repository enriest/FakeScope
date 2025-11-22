"""Upload the trained DistilBERT fake news model to Hugging Face Hub.

Usage (from project root):

    export HF_TOKEN=your_hf_write_token
    python scripts/upload_model_hf.py --repo-id enriest/fakescope-distilbert-2stage \
        --model-dir distilbert_fakenews_2stage --private

Requirements:
    pip install huggingface_hub

This script:
 1. Validates required files in the local model directory.
 2. Creates or updates the remote repository (optionally private).
 3. Uploads model, tokenizer, and config artifacts.
 4. Generates a README.md with short metadata if absent remote.

It is idempotent: re-running only re-uploads changed files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, whoami
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

REQUIRED_FILES: List[str] = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "vocab.txt",
]

OPTIONAL_FILES: List[str] = [
    "special_tokens_map.json",
    "tokenizer_config.json",
]

README_TEMPLATE = """# FakeScope DistilBERT 2-Stage Fake News Classifier

This repository hosts the 2-stage domain adapted DistilBERT classifier used by FakeScope.

## Model Card
- Architecture: DistilBERT (sequence classification, 2 labels: Fake (0), True (1))
- Training Strategy: Domain adaptation (MLM) on unlabeled news corpus followed by supervised fine-tuning on labeled fake/true news.
- Intended Use: Credibility scoring for news claims and articles.
- Labels: index 0 = Fake, index 1 = True

## Example (Python)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "{{repo_id}}"  # Replace with your repo

model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Scientists discover new renewable energy breakthrough"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)[0]
print(f"Fake={{probs[0]:.3f}} True={{probs[1]:.3f}}")
```

## License
Weights under same license as the FakeScope project repository.

## Citation
If you use this model, please cite the FakeScope project.
"""

def validate_local_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise SystemExit(f"Model directory '{model_dir}' does not exist.")
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    # create_repo is safe; will raise if exists with different visibility
    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create repo (might already exist): {e}")


def upload_model(api: HfApi, repo_id: str, model_dir: Path) -> None:
    # Upload folder preserving structure (only top-level files expected)
    print(f"Uploading model artifacts from '{model_dir}' to '{repo_id}'...")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        path_in_repo=".",
        allow_patterns=REQUIRED_FILES + OPTIONAL_FILES,
    )


def maybe_create_readme(api: HfApi, repo_id: str) -> None:
    # Try to fetch existing README
    try:
        files = api.list_repo_files(repo_id)
        if any(f.lower() == "readme.md" for f in files):
            print("README.md already exists, skipping creation.")
            return
    except Exception:
        pass
    readme_content = README_TEMPLATE.format(repo_id=repo_id)
    upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print("Uploaded README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload FakeScope model to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. user/fakescope-distilbert-2stage")
    parser.add_argument("--model-dir", default="distilbert_fakenews_2stage", help="Local model directory")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN (or HUGGINGFACE_TOKEN) environment variable not set.")

    api = HfApi(token=token)
    user = whoami(token=token)["name"]
    print(f"Authenticated as: {user}")

    model_dir = Path(args.model_dir)
    validate_local_dir(model_dir)
    ensure_repo(api, args.repo_id, private=args.private)
    upload_model(api, args.repo_id, model_dir)
    maybe_create_readme(api, args.repo_id)
    print("Upload complete.")


if __name__ == "__main__":
    main()
