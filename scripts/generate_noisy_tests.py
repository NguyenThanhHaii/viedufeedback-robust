from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.noise import (
    build_clean_eval_dataframe,
    build_noisy_dataframe,
    plot_noisy_summary,
    sample_noisy_examples,
    summarize_noisy_data,
    write_noisy_generation_report,
)
from src.utils import ensure_dir, load_yaml, project_root, save_json, set_seed


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "noise.yaml")

    seed = int(config["seed"])
    set_seed(seed)

    input_test_file = root / config["data"]["input_test_file"]
    text_col = config["data"]["text_col"]

    noise_dir = root / config["paths"]["noise_dir"]
    tables_dir = root / config["paths"]["tables_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    reports_dir = root / config["paths"]["reports_dir"]

    ensure_dir(noise_dir)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    print(f"Loading clean test set: {input_test_file}")
    test_df = pd.read_csv(input_test_file)

    required_cols = [
        "id",
        "split",
        text_col,
        "sentiment_label",
        "sentiment_name",
        "topic_label",
        "topic_name",
    ]
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test set: {missing_cols}")

    print(f"Test rows: {len(test_df)}")

    print("Saving clean evaluation copy...")
    clean_df = build_clean_eval_dataframe(test_df, text_col=text_col)
    clean_df.to_csv(
        noise_dir / "test_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    noisy_dfs = []

    for variant in config["variants"]:
        variant_name = variant["name"]
        print(f"Generating noisy variant: {variant_name}")

        noisy_df = build_noisy_dataframe(
            test_df=test_df,
            variant=variant,
            seed=seed,
            text_col=text_col,
        )

        noisy_path = noise_dir / f"test_noisy_{variant_name}.csv"
        noisy_df.to_csv(
            noisy_path,
            index=False,
            encoding="utf-8-sig",
        )

        noisy_dfs.append(noisy_df)

        changed_count = int(noisy_df["changed"].sum())
        print(
            f"  saved: {noisy_path} | "
            f"changed: {changed_count}/{len(noisy_df)}"
        )

    noisy_all_df = pd.concat(noisy_dfs, ignore_index=True)
    eval_all_df = pd.concat([clean_df, noisy_all_df], ignore_index=True)

    noisy_all_df.to_csv(
        noise_dir / "test_noisy_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eval_all_df.to_csv(
        noise_dir / "test_eval_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Creating generation summary...")
    summary_df = summarize_noisy_data(eval_all_df)
    summary_df.to_csv(
        tables_dir / "noisy_generation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Sampling noisy examples...")
    examples_df = sample_noisy_examples(
        noisy_df=noisy_all_df,
        examples_per_variant=int(config["sampling"]["examples_per_variant"]),
        seed=seed,
    )
    examples_df.to_csv(
        tables_dir / "noisy_text_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saving plots...")
    plot_noisy_summary(
        summary_df=summary_df,
        value_col="changed_percent",
        title="Changed Percent by Noise Variant",
        output_path=figures_dir / "noisy_changed_percent_by_variant.png",
    )
    plot_noisy_summary(
        summary_df=summary_df,
        value_col="avg_char_change_ratio",
        title="Average Character Change Ratio by Noise Variant",
        output_path=figures_dir / "noisy_char_change_ratio_by_variant.png",
    )

    print("Saving report and config snapshot...")
    write_noisy_generation_report(
        report_path=reports_dir / "noisy_generation_report.md",
        summary_df=summary_df,
        examples_df=examples_df,
    )

    save_json(
        {
            "seed": seed,
            "input_test_file": str(input_test_file),
            "text_col": text_col,
            "variants": config["variants"],
            "outputs": {
                "clean": str(noise_dir / "test_clean.csv"),
                "noisy_all": str(noise_dir / "test_noisy_all.csv"),
                "eval_all": str(noise_dir / "test_eval_all.csv"),
            },
        },
        reports_dir / "noise_config_snapshot.json",
    )

    print("\nNoisy test generation completed.")
    print(f"Clean test: {noise_dir / 'test_clean.csv'}")
    print(f"Noisy all: {noise_dir / 'test_noisy_all.csv'}")
    print(f"Eval all: {noise_dir / 'test_eval_all.csv'}")
    print(f"Summary: {tables_dir / 'noisy_generation_summary.csv'}")
    print(f"Examples: {tables_dir / 'noisy_text_examples.csv'}")
    print(f"Report: {reports_dir / 'noisy_generation_report.md'}")


if __name__ == "__main__":
    main()