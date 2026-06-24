"""Prepare Brain Tumor MRI dataset for training.

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected layout after extraction:
  <source>/
    Training/
      glioma/
      meningioma/
      notumor/
      pituitary/
    Testing/
      glioma/
      ...
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare brain MRI dataset")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to extracted dataset folder",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="./data/brain_tumor_mri",
        help="Destination directory for organized dataset",
    )
    parser.add_argument(
        "--kaggle-download",
        action="store_true",
        help="Download dataset via kagglehub (no API key required)",
    )
    return parser.parse_args()


def copy_dataset(source: Path, dest: Path) -> None:
    for split in ("Training", "Testing"):
        src_split = source / split
        if not src_split.exists():
            raise FileNotFoundError(f"Missing split folder: {src_split}")

        dest_split = dest / split
        if dest_split.exists():
            shutil.rmtree(dest_split)
        shutil.copytree(src_split, dest_split)
        count = sum(1 for _ in dest_split.rglob("*") if _.is_file())
        print(f"Copied {split}: {count} images -> {dest_split}")


def kaggle_download() -> Path:
    import kagglehub

    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    source = Path(path)
    if (source / "Training").exists():
        return source
    raise FileNotFoundError(
        "Download completed but Training/ folder not found. "
        "Pass --source manually after extracting the zip."
    )


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)

    if args.kaggle_download:
        source = kaggle_download()
    elif args.source:
        source = Path(args.source)
    else:
        print(__doc__)
        print("\nUsage examples:")
        print("  python scripts/prepare_data.py --kaggle-download")
        print("  python scripts/prepare_data.py --source C:/Downloads/brain-tumor-mri")
        sys.exit(1)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    dest.mkdir(parents=True, exist_ok=True)
    copy_dataset(source, dest)
    print(f"\nDataset ready at: {dest.resolve()}")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
