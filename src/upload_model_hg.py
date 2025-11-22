# scripts/upload_model_hf.py
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
import os

def main():
    parser = argparse.ArgumentParser(description="Upload a model directory to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id (e.g. user/model-name)")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        path_in_repo=".",
        commit_message="Upload model",
    )
    print(f"Model uploaded to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()