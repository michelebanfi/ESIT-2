"""Download the competition dataset via kagglehub and symlink it to ./data/."""
import os
import sys
import json
from pathlib import Path

# Load KAGGLE_API_TOKEN from .env before importing kagglehub
ROOT = Path(__file__).parent.parent
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# kagglehub uses KAGGLE_KEY env var OR ~/.kaggle/kaggle.json
# Write kaggle.json if we have the token but no json file
kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_json.exists():
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        print("ERROR: set KAGGLE_API_TOKEN in .env or create ~/.kaggle/kaggle.json")
        sys.exit(1)
    username = os.environ.get("KAGGLE_USERNAME", "")
    if not username:
        username = input("Enter your Kaggle username: ").strip()
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": token}))
    kaggle_json.chmod(0o600)
    print(f"Wrote {kaggle_json}")

import kagglehub  # noqa: E402

COMPETITION = "esit-d2i-competition-2026-charting-the-path-to-bifrost-task-2"
print(f"Downloading {COMPETITION} ...")
path = kagglehub.competition_download(COMPETITION)
print(f"Downloaded to: {path}")

data_link = ROOT / "data"
if data_link.exists() or data_link.is_symlink():
    data_link.unlink() if data_link.is_symlink() else None
if not data_link.exists():
    data_link.symlink_to(path)
    print(f"Symlinked ./data -> {path}")
else:
    print(f"./data already exists ({data_link})")

# Print shapes of every file
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

data_dir = Path(path)
print("\n=== Files ===")
for f in sorted(data_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(data_dir)}", end="")
        if f.suffix == ".npy":
            arr = np.load(f, mmap_mode="r")
            print(f"  shape={arr.shape} dtype={arr.dtype}")
        elif f.suffix == ".csv":
            df = pd.read_csv(f, nrows=5)
            print(f"  columns={list(df.columns)}")
        elif f.suffix == ".pth":
            print(f"  (checkpoint)")
        elif f.suffix == ".ipynb":
            print(f"  (notebook)")
        else:
            print()
