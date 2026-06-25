from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_analysis import (
    combine_class_reports,
    combine_robustness_results,
    compute_drop_from_clean,
    create_clean_model_comparison,
    create_error_examples,
    create_final_robustness_comparison,
    create_transition_summary,
    load_required_csv,
    plot_clean_macro_f1_comparison,
    plot_robustness_drop,
    plot_robustness_macro_f1,
    save_final_metrics_json,
    write_final_comparison_report,
)
from src.utils import ensure_dir, load_yaml, project_root, set_seed


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "final_analysis.yaml")

    seed = int(config["seed"])
    set_seed(seed)

    tables_dir = ensure_dir(root / config["paths"]["tables_dir"])
    figures_dir = ensure_dir(root / config["paths"]["figures_dir"])
    reports_dir = ensure_dir(root / config["paths"]["reports_dir"])
    predictions_dir = ensure_dir(root / config["paths"]["predictions_dir"])
    metrics_dir = ensure_dir(root / config["paths"]["metrics_dir"])

    inputs = config["inputs"]
    variant_order = config["variant_order"]
    model_order = config["model_order"]

    print("Loading Stage 5 and Stage 6 outputs...")

    baseline_robustness_results = load_required_csv(root / inputs["baseline_robustness_results"])
    baseline_robustness_class_report = load_required_csv(root / inputs["baseline_robustness_class_report"])
    phobert_robustness_results = load_required_csv(root / inputs["phobert_robustness_results"])
    phobert_robustness_class_report = load_required_csv(root / inputs["phobert_robustness_class_report"])
    phobert_robustness_predictions = load_required_csv(root / inputs["phobert_robustness_predictions"])

    print("Combining robustness results...")
    combined_results = combine_robustness_results(
        baseline_results=baseline_robustness_results,
        phobert_results=phobert_robustness_results,
        variant_order=variant_order,
    )

    combined_drop = compute_drop_from_clean(
        results_df=combined_results,
        variant_order=variant_order,
    )

    clean_comparison = create_clean_model_comparison(
        combined_results=combined_results,
        model_order=model_order,
    )

    robustness_comparison = create_final_robustness_comparison(
        combined_results=combined_results,
        combined_drop=combined_drop,
        variant_order=variant_order,
        model_order=model_order,
    )

    print("Combining per-class reports...")
    per_class_comparison = combine_class_reports(
        baseline_report=baseline_robustness_class_report,
        phobert_report=phobert_robustness_class_report,
        variant_order=variant_order,
        model_order=model_order,
    )

    print("Creating error examples...")
    target_variants = config["error_analysis"]["target_variants"]
    max_examples = int(config["error_analysis"]["max_examples_per_variant"])

    sentiment_errors = create_error_examples(
        phobert_predictions=phobert_robustness_predictions,
        task_name="sentiment",
        target_variants=target_variants,
        max_examples_per_variant=max_examples,
        seed=seed,
    )

    topic_errors = create_error_examples(
        phobert_predictions=phobert_robustness_predictions,
        task_name="topic",
        target_variants=target_variants,
        max_examples_per_variant=max_examples,
        seed=seed,
    )

    sentiment_transition = create_transition_summary(
        phobert_predictions=phobert_robustness_predictions,
        task_name="sentiment",
        target_variants=target_variants,
    )
    topic_transition = create_transition_summary(
        phobert_predictions=phobert_robustness_predictions,
        task_name="topic",
        target_variants=target_variants,
    )

    print("Saving tables...")
    clean_comparison.to_csv(
        tables_dir / "final_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    robustness_comparison.to_csv(
        tables_dir / "final_robustness_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_class_comparison.to_csv(
        tables_dir / "final_per_class_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sentiment_errors.to_csv(
        tables_dir / "error_examples_sentiment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_errors.to_csv(
        tables_dir / "error_examples_topic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sentiment_transition.to_csv(
        tables_dir / "error_transition_summary_sentiment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_transition.to_csv(
        tables_dir / "error_transition_summary_topic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saving figures...")
    plot_clean_macro_f1_comparison(
        clean_comparison=clean_comparison,
        output_path=figures_dir / "final_clean_macro_f1_comparison.png",
    )

    for task_name in ["sentiment", "topic"]:
        plot_robustness_macro_f1(
            robustness_comparison=robustness_comparison,
            task_name=task_name,
            variant_order=variant_order,
            output_path=figures_dir / f"final_robustness_macro_f1_{task_name}.png",
        )
        plot_robustness_drop(
            robustness_comparison=robustness_comparison,
            task_name=task_name,
            variant_order=variant_order,
            output_path=figures_dir / f"final_robustness_drop_{task_name}.png",
        )

    print("Saving report and metrics JSON...")
    write_final_comparison_report(
        report_path=reports_dir / "final_comparison_report.md",
        clean_comparison=clean_comparison,
        robustness_comparison=robustness_comparison,
        per_class_comparison=per_class_comparison,
        sentiment_errors=sentiment_errors,
        topic_errors=topic_errors,
    )

    save_final_metrics_json(
        output_path=metrics_dir / "final_comparison_metrics.json",
        clean_comparison=clean_comparison,
        robustness_comparison=robustness_comparison,
    )

    print("\nStage 7 completed.")
    print(f"Clean comparison: {tables_dir / 'final_model_comparison.csv'}")
    print(f"Robustness comparison: {tables_dir / 'final_robustness_comparison.csv'}")
    print(f"Per-class comparison: {tables_dir / 'final_per_class_comparison.csv'}")
    print(f"Sentiment errors: {tables_dir / 'error_examples_sentiment.csv'}")
    print(f"Topic errors: {tables_dir / 'error_examples_topic.csv'}")
    print(f"Report: {reports_dir / 'final_comparison_report.md'}")


if __name__ == "__main__":
    main()
