"""Download and validate the official UCI SECOM raw data files."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


UCI_SECOM_ZIP_URLS = (
    "https://archive.ics.uci.edu/static/public/179/secom.zip",
    "http://archive.ics.uci.edu/static/public/179/secom.zip",
)
REQUIRED_FILES = ("secom.data", "secom_labels.data", "secom.names")
EXPECTED_SHAPE = (1567, 590)
EXPECTED_LABELS = {-1, 1}


def repository_root() -> Path:
    """Return the repository root based on this script's location."""
    return Path(__file__).resolve().parents[1]


def download_zip(destination: Path) -> None:
    """Download the official UCI SECOM ZIP file."""
    errors = []

    for url in UCI_SECOM_ZIP_URLS:
        print(f"Downloading official UCI SECOM data from:\n{url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Download failed with HTTP status {response.status}."
                    )
                with destination.open("wb") as file:
                    shutil.copyfileobj(response, file)
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if destination.exists():
                destination.unlink()
    else:
        raise RuntimeError(
            "Could not download SECOM data from the official UCI source. "
            + " | ".join(errors)
        )

    if destination.stat().st_size == 0:
        raise RuntimeError("Downloaded ZIP file is empty.")


def extract_required_files(zip_path: Path, raw_dir: Path, force: bool) -> None:
    """Extract required SECOM files into data/raw."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [raw_dir / file_name for file_name in REQUIRED_FILES]
    existing_files = [path for path in existing_files if path.exists()]
    if existing_files and not force:
        existing = ", ".join(path.name for path in existing_files)
        raise FileExistsError(
            f"Refusing to overwrite existing raw file(s): {existing}. "
            "Run with --force to replace them."
        )

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        missing = [file_name for file_name in REQUIRED_FILES if file_name not in names]
        if missing:
            raise RuntimeError(
                "The official ZIP did not contain expected file(s): "
                + ", ".join(missing)
            )

        for file_name in REQUIRED_FILES:
            target = raw_dir / file_name
            with archive.open(file_name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)

            if target.stat().st_size == 0:
                raise RuntimeError(f"Extracted file is empty: {target}")


def validate_raw_data(raw_dir: Path) -> None:
    """Validate downloaded SECOM feature and label files."""
    features_path = raw_dir / "secom.data"
    labels_path = raw_dir / "secom_labels.data"

    X = pd.read_csv(features_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["target", "timestamp"],
    )
    y = labels["target"]

    if X.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected feature shape {X.shape}; expected {EXPECTED_SHAPE}.")

    if len(y) != EXPECTED_SHAPE[0]:
        raise ValueError(f"Unexpected label count {len(y)}; expected {EXPECTED_SHAPE[0]}.")

    observed_labels = set(y.dropna().astype(int).unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected target labels {sorted(observed_labels)}; "
            f"expected {sorted(EXPECTED_LABELS)}."
        )

    print("Validation passed.")
    print(f"Feature matrix shape: {X.shape}")
    print("Target distribution:")
    for label, count in y.value_counts().sort_index().items():
        class_name = "Fail" if label == 1 else "Pass"
        print(f"  {label:>2} ({class_name}): {count}")


def main() -> int:
    """Download, extract, and validate the SECOM data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing raw SECOM files.",
    )
    args = parser.parse_args()

    raw_dir = repository_root() / "data" / "raw"

    try:
        existing_files = [raw_dir / file_name for file_name in REQUIRED_FILES]
        existing_files = [path for path in existing_files if path.exists()]
        if existing_files and not args.force:
            existing = ", ".join(path.name for path in existing_files)
            raise FileExistsError(
                f"Refusing to overwrite existing raw file(s): {existing}. "
                "Run with --force to replace them."
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            zip_path = Path(temporary_directory) / "secom.zip"
            download_zip(zip_path)
            extract_required_files(zip_path, raw_dir, force=args.force)
        validate_raw_data(raw_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"SECOM data is ready in: {raw_dir.relative_to(repository_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
