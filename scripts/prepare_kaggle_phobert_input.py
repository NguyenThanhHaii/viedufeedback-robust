from __future__ import annotations

import zipfile
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_file_if_exists(zipf: zipfile.ZipFile, root: Path, relative_path: str) -> None:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    zipf.write(path, arcname=relative_path)


def main() -> None:
    root = project_root()

    output_dir = root / "kaggle" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_zip = output_dir / "viedufeedback_phobert_stage6.zip"

    required_files = [
        "data/processed/train_standardized.csv",
        "data/processed/validation_standardized.csv",
        "data/processed/test_standardized.csv",
        "data/noisy/test_eval_all.csv",
        "outputs/mappings/sentiment_label_mapping.json",
        "outputs/mappings/topic_label_mapping.json",
        "configs/phobert_sentiment.yaml",
        "configs/phobert_topic.yaml",
        "kaggle/stage6_train_phobert.py",
    ]

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for relative_path in required_files:
            add_file_if_exists(zipf, root, relative_path)

    print("Kaggle Stage 6 input package created:")
    print(output_zip)
    print()
    print("Upload this zip as a Kaggle Dataset, or attach it to a Kaggle notebook.")
    print("Inside Kaggle, unzip it and run kaggle/stage6_train_phobert.py.")


if __name__ == "__main__":
    main()
