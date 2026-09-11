"""Bundling script to package the project into sphesihle-anthony-mhlongo-takehome.zip."""

import argparse
import os
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "sphesihle-anthony-mhlongo-takehome.zip"

INCLUDED_PATTERNS = [
    "src/**/*.py",
    "tests/**/*.py",
    "data/loader.py",
    "data/survey_responses.csv",
    "requirements.txt",
    "README.md",
    "TRADEOFFS.md",
    "Makefile",
    "bundle.py",
    "LICENSE",
]

EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".env",
    ".venv",
    "venv",
    ZIP_NAME,
}


def create_bundle(blank_tradeoffs: bool = False) -> None:
    root_dir = Path(__file__).parent.resolve()
    zip_path = root_dir / ZIP_NAME

    print(f"Creating bundle: {zip_path}")
    print(f"Blank TRADEOFFS.md: {blank_tradeoffs}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(root_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_NAMES and not d.startswith(".")]

            for file in files:
                if file in EXCLUDED_NAMES or file.endswith(".pyc") or file == ZIP_NAME:
                    continue

                full_path = Path(root) / file
                rel_path = full_path.relative_to(root_dir)

                # Skip scratch files
                if "scratch" in rel_path.parts:
                    continue

                if rel_path.name == "TRADEOFFS.md" and blank_tradeoffs:
                    print(f"  Adding (blank): {rel_path}")
                    zf.writestr(str(rel_path), "# Architecture, Privacy & Tradeoffs Document\n\n[Blank]\n")
                else:
                    print(f"  Adding: {rel_path}")
                    zf.write(full_path, arcname=str(rel_path))

    print(f"\nBundle created successfully: {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle project into submission zip")
    parser.add_argument(
        "--blank-tradeoffs",
        action="store_true",
        help="Include a blank TRADEOFFS.md instead of populated version",
    )
    args = parser.parse_args()
    create_bundle(blank_tradeoffs=args.blank_tradeoffs)
