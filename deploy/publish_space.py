"""Publish the scoring service to a Hugging Face Space.

Uploads only what the running service touches. No notebooks, no reports, no
training data, so the Space stays small and builds quickly.

Prerequisite, once. A Hugging Face token with WRITE access, a read-only token
cannot create a Space.

    huggingface-cli login

Then:

    python deploy/publish_space.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(tempfile.gettempdir()) / "cod_scorer_space_stage"
REPO_ID = os.environ.get("HF_SPACE_ID", "anilkumarlucky/cod-return-risk-scorer")

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)

# Only what the running service touches. No notebooks, no reports, no training
# data, no venv. Keeps the image small and the Space build fast.
(STAGE / "api").mkdir()
(STAGE / "api" / "static").mkdir()
(STAGE / "src").mkdir()
(STAGE / "config").mkdir()
(STAGE / "models").mkdir()
(STAGE / "data" / "external").mkdir(parents=True)
(STAGE / "data" / "processed").mkdir(parents=True)

copy_files = [
    ("deploy/Dockerfile", "Dockerfile"),
    ("deploy/requirements-serve.txt", "requirements-serve.txt"),
    ("deploy/SPACE_README.md", "README.md"),
    ("api/main.py", "api/main.py"),
    ("api/static/index.html", "api/static/index.html"),
    ("config/evidence.yaml", "config/evidence.yaml"),
    ("models/final_model.joblib", "models/final_model.joblib"),
    ("data/external/india_pincodes.csv", "data/external/india_pincodes.csv"),
    ("data/processed/val.parquet", "data/processed/val.parquet"),
    ("LICENSE", "LICENSE"),
]
for src in ROOT.glob("src/*.py"):
    copy_files.append((f"src/{src.name}", f"src/{src.name}"))

for rel_src, rel_dst in copy_files:
    s, d = ROOT / rel_src, STAGE / rel_dst
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)

(STAGE / "api" / "__init__.py").write_text("", encoding="utf-8")

total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
print(f"staged {sum(1 for f in STAGE.rglob('*') if f.is_file())} files, "
      f"{total / 1e6:.1f} MB")

api = HfApi()
try:
    create_repo(REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"space ready: {REPO_ID}")
except Exception as exc:
    print("create_repo failed:", type(exc).__name__, str(exc)[:200])
    if "403" in str(exc):
        print("\nThe stored token cannot create a Space. Run 'huggingface-cli login'")
        print("with a token that has WRITE access, then re-run this script.")
    sys.exit(1)

api.upload_folder(
    folder_path=str(STAGE),
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Deploy scoring service and one-page demo",
)
print(f"uploaded -> https://huggingface.co/spaces/{REPO_ID}")
