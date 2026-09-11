"""Bundling script to package the project into sphesihle-anthony-mhlongo-takehome.zip."""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "sphesihle-anthony-mhlongo-takehome.zip"

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    "env",
    ".vscode",
    ".idea",
    ".agents",
    ".gemini",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "scratch",
}

EXCLUDED_EXACT_FILES = {
    "AGENTS.md",
    "GEMINI.md",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".coverage",
}

EXCLUDED_FILE_EXTENSIONS = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
    ".bak",
    ".log",
    ".zip",
)


def is_excluded_dir(dir_name: str) -> bool:
    """Check if a directory should be pruned from packaging."""
    if dir_name in EXCLUDED_DIR_NAMES:
        return True
    if dir_name.startswith(".") and dir_name != ".":
        return True
    return False


def is_excluded_file(file_name: str, rel_path: Path) -> bool:
    """Check if a file should be excluded from packaging."""
    # Always include .env.example
    if file_name == ".env.example":
        return False

    # Strictly exclude .env and local variants (.env.local, etc.)
    if file_name == ".env" or file_name.startswith(".env."):
        return True

    # Exclude zip files (including output bundles)
    if file_name.endswith(".zip") or file_name == ZIP_NAME:
        return True

    # Exclude bytecode, temporary files, swap files, logs
    if file_name.endswith(EXCLUDED_FILE_EXTENSIONS):
        return True

    # Exclude exact filenames (agent directives, OS metadata, coverage)
    if file_name in EXCLUDED_EXACT_FILES:
        return True

    # Exclude anything in scratch folders
    if any(part.lower() in ("scratch", ".scratch") for part in rel_path.parts):
        return True

    return False


def clean_project() -> None:
    """Remove cache directories, bytecode, test artifacts, and generated zip bundles."""
    root_dir = Path(__file__).parent.resolve()
    print(f"Cleaning build artifacts and temporary files in {root_dir}...")
    removed_dirs = 0
    removed_files = 0

    cleanable_dir_names = {"__pycache__", ".pytest_cache", "htmlcov", ".mypy_cache", ".ruff_cache"}

    # Remove cache directories
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for d in dirs:
            if d in cleanable_dir_names:
                dir_path = Path(root) / d
                shutil.rmtree(dir_path, ignore_errors=True)
                removed_dirs += 1

    cleanable_file_names = {".coverage", ".DS_Store", "Thumbs.db", "desktop.ini"}

    # Remove temporary files, bytecode, coverage, and zip bundles
    # (strictly preserving .venv, venv, .git, and .env)
    for root, _, files in os.walk(root_dir):
        rel_root = Path(root).relative_to(root_dir)
        if any(part in {".venv", "venv", ".git"} for part in rel_root.parts):
            continue

        for f in files:
            file_path = Path(root) / f
            if (
                f.endswith(EXCLUDED_FILE_EXTENSIONS)
                or f in cleanable_file_names
            ):
                try:
                    file_path.unlink(missing_ok=True)
                    removed_files += 1
                except OSError:
                    pass

    print(f"Clean complete: removed {removed_dirs} cache directories and {removed_files} temporary/artifact files.")


def create_bundle(blank_tradeoffs: bool = False) -> None:
    root_dir = Path(__file__).parent.resolve()
    zip_path = root_dir / ZIP_NAME

    print(f"Creating bundle: {zip_path}")
    print(f"Blank TRADEOFFS.md: {blank_tradeoffs}")

    total_added = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(root_dir):
            # Prune excluded directories in-place so walk doesn't descend into them
            dirs[:] = [d for d in dirs if not is_excluded_dir(d)]

            for file in sorted(files):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(root_dir)

                if is_excluded_file(file, rel_path):
                    continue

                if rel_path.name == "TRADEOFFS.md" and blank_tradeoffs:
                    print(f"  Adding (blank): {rel_path}")
                    zf.writestr(str(rel_path), "# Architecture, Privacy & Tradeoffs Document\n\n[Blank]\n")
                else:
                    print(f"  Adding: {rel_path}")
                    zf.write(full_path, arcname=str(rel_path))
                total_added += 1

    print(f"\nBundle created successfully: {zip_path} ({total_added} files, {zip_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle project into submission zip or clean artifacts")
    parser.add_argument(
        "--blank-tradeoffs",
        action="store_true",
        help="Include a blank TRADEOFFS.md instead of populated version",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean cache directories, temporary files, and zip bundles",
    )
    args = parser.parse_args()

    if args.clean:
        clean_project()
    else:
        create_bundle(blank_tradeoffs=args.blank_tradeoffs)
