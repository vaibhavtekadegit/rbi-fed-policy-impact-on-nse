"""
Upload the Gradio app to the Hugging Face Space.

Used by .github/workflows/deploy_space.yml; can also be run by hand from the repository root:
    HF_TOKEN=... python app/deploy_space.py

The Space gets: app.py, model.py, requirements.txt, README.md, data/ and models/ (copied from the repo's models/).
Files are sent through the Hugging Face API, so Git LFS is not needed.
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = os.environ.get("HF_SPACE", "Vaibhavtekade/rbi-fed-nse-app")
APP_FILES = ["app.py", "model.py", "requirements.txt", "README.md"]
MODEL_FILES = ["tone_clf_v1.joblib", "xgb_rq4.json", "xgb_rq4_features.json"]


def stage(dest):
    for f in APP_FILES:
        shutil.copy2(ROOT / "app" / f, dest / f)
    shutil.copytree(ROOT / "app/data", dest / "data")
    (dest / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy2(ROOT / "models" / f, dest / "models" / f)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is empty: add it under GitHub Settings > Secrets and variables > Actions")
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        stage(dest)
        print("Uploading:", sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()))
        HfApi(token=token).upload_folder(
            repo_id=SPACE, repo_type="space", folder_path=str(dest),
            commit_message=f"Deploy from GitHub {sha}",
            delete_patterns=["*.py", "data/*", "models/*"],   # remove files that no longer exist in the repo
        )
    print(f"Deployed to https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
