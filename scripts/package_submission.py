"""
Create a submission zip excluding large/generated artifacts.

Usage (from project root):
    python scripts/package_submission.py
"""

import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dist")
OUT_ZIP = os.path.join(OUT_DIR, "scholar-submission.zip")

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "data",
    "indexes",
    "dist",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTENSIONS = {".pkl", ".bin", ".zip", ".jsonl"}


def should_include(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    if any(rel_path.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return False
    if "evaluation/trec-covid" in rel_path.replace("\\", "/"):
        return False
    if "evaluation/scidocs/scidocs" in rel_path.replace("\\", "/"):
        return False
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ROOT)
                if should_include(rel):
                    zf.write(full, rel)
                    count += 1
    size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
    print(f"Created {OUT_ZIP}")
    print(f"  Files: {count}")
    print(f"  Size:  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
