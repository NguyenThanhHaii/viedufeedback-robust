from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robust_inference import (
    build_noisy_validation,
    ensure_dir,
    load_yaml,
    project_root,
    save_json,
    tune_no_accent_threshold,
)


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "robust_inference.yaml")

    seed = int(config["seed"])
    text_col = config["data"]["text_col"]
    validation_file = root / config["data"]["validation_file"]
    validation_noisy_file = root / config["data"]["validation_noisy_file"]

    tables_dir = ensure_dir(root / config["paths"]["tables_dir"])
    metrics_dir = ensure_dir(root / config["paths"]["metrics_dir"])
    ensure_dir(validation_noisy_file.parent)

    if not validation_file.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_file}")

    print("Loading validation split...")
    validation_df = pd.read_csv(validation_file)

    if text_col not in validation_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in {validation_file}")

    print("Generating noisy validation variants...")
    noisy_validation, generation_summary = build_noisy_validation(
        validation_df=validation_df,
        text_col=text_col,
        variants=config["noisy_validation"]["variants"],
        seed=seed,
        original_id_prefix=config["data"].get("original_id_prefix", "validation"),
    )

    noisy_validation.to_csv(validation_noisy_file, index=False, encoding="utf-8-sig")
    generation_summary.to_csv(
        tables_dir / "noisy_validation_generation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Tuning no-accent detector threshold on noisy validation...")
    threshold_table, selected = tune_no_accent_threshold(
        validation_noisy_df=noisy_validation,
        text_col=text_col,
        no_accent_variants=config["routing"]["no_accent_variants"],
        threshold_candidates=[float(x) for x in config["detector"]["threshold_candidates"]],
        min_alpha_chars=int(config["detector"]["min_alpha_chars"]),
        min_words=int(config["detector"]["min_words"]),
    )

    threshold_table.to_csv(
        tables_dir / "noisy_validation_detection_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(selected, metrics_dir / "robust_inference_selected_threshold.json")

    print("\nStage 8 validation generation and threshold tuning completed.")
    print(f"Noisy validation: {validation_noisy_file}")
    print(f"Generation summary: {tables_dir / 'noisy_validation_generation_summary.csv'}")
    print(f"Threshold table: {tables_dir / 'noisy_validation_detection_thresholds.csv'}")
    print(f"Selected threshold JSON: {metrics_dir / 'robust_inference_selected_threshold.json'}")
    print(f"Selected threshold: {selected['threshold']}")


if __name__ == "__main__":
    main()
