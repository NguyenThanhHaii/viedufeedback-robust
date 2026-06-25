from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robust_inference import (
    build_robust_inference_comparison,
    build_router_predictions,
    compute_classification_report_table,
    compute_classification_results,
    compute_detector_report,
    compute_drop_from_clean,
    ensure_dir,
    load_json,
    load_yaml,
    plot_metric_by_variant,
    project_root,
    save_json,
    write_robust_inference_report,
)


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "robust_inference.yaml")

    tables_dir = ensure_dir(root / config["paths"]["tables_dir"])
    figures_dir = ensure_dir(root / config["paths"]["figures_dir"])
    reports_dir = ensure_dir(root / config["paths"]["reports_dir"])
    predictions_dir = ensure_dir(root / config["paths"]["predictions_dir"])
    metrics_dir = ensure_dir(root / config["paths"]["metrics_dir"])

    selected_threshold_path = metrics_dir / "robust_inference_selected_threshold.json"

    if selected_threshold_path.exists():
        selected_threshold = load_json(selected_threshold_path)
        threshold = float(selected_threshold["threshold"])
        print(f"Using selected threshold from validation: {threshold}")
    else:
        threshold = float(config["detector"]["fallback_threshold"])
        selected_threshold = {
            "threshold": threshold,
            "selection_metric": "fallback_threshold_from_config",
            "warning": "Selected threshold JSON not found. Run scripts/generate_noisy_validation.py first.",
            "min_alpha_chars": int(config["detector"]["min_alpha_chars"]),
            "min_words": int(config["detector"]["min_words"]),
            "no_accent_variants": config["routing"]["no_accent_variants"],
        }
        print(f"WARNING: selected threshold JSON missing. Using fallback threshold: {threshold}")

    phobert_path = root / config["inputs"]["phobert_predictions"]
    baseline_path = root / config["inputs"]["baseline_predictions"]

    if not phobert_path.exists():
        raise FileNotFoundError(f"PhoBERT predictions not found: {phobert_path}")

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {baseline_path}")

    print("Loading predictions...")
    phobert_predictions = pd.read_csv(phobert_path)
    baseline_predictions = pd.read_csv(baseline_path)

    print("Building router predictions from saved model predictions...")
    router_predictions = build_router_predictions(
        phobert_predictions=phobert_predictions,
        char_svm_predictions=baseline_predictions,
        threshold=threshold,
        min_alpha_chars=int(config["detector"]["min_alpha_chars"]),
        min_words=int(config["detector"]["min_words"]),
        no_accent_variants=config["routing"]["no_accent_variants"],
    )

    router_predictions.to_csv(
        predictions_dir / "robust_inference_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    variant_order = config["evaluation"]["variant_order"]

    print("Computing classification metrics...")
    results = compute_classification_results(
        prediction_df=router_predictions,
        variant_order=variant_order,
    )
    drop = compute_drop_from_clean(
        results_df=results,
        variant_order=variant_order,
    )
    class_report = compute_classification_report_table(
        prediction_df=router_predictions,
        variant_order=variant_order,
    )
    comparison = build_robust_inference_comparison(
        results_df=results,
        drop_df=drop,
        variant_order=variant_order,
    )

    print("Computing detector report...")
    detector_overall, detector_by_variant = compute_detector_report(
        prediction_df=router_predictions,
        no_accent_variants=config["routing"]["no_accent_variants"],
    )

    print("Saving tables...")
    results.to_csv(
        tables_dir / "robust_inference_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    drop.to_csv(
        tables_dir / "robust_inference_drop.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        tables_dir / "robust_inference_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_report.to_csv(
        tables_dir / "robust_inference_class_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detector_overall.to_csv(
        tables_dir / "robust_inference_detector_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detector_by_variant.to_csv(
        tables_dir / "robust_inference_detector_by_variant.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saving figures...")
    for task_name in ["sentiment", "topic"]:
        plot_metric_by_variant(
            results_df=results,
            task_name=task_name,
            metric="macro_f1",
            variant_order=variant_order,
            output_path=figures_dir / f"robust_inference_macro_f1_{task_name}.png",
            title=f"Stage 8 Macro-F1 by Variant - {task_name}",
            ylabel="Macro-F1",
        )

        plot_metric_by_variant(
            results_df=drop,
            task_name=task_name,
            metric="macro_f1_drop",
            variant_order=variant_order,
            output_path=figures_dir / f"robust_inference_drop_{task_name}.png",
            title=f"Stage 8 Macro-F1 Drop from Clean - {task_name}",
            ylabel="Macro-F1 drop",
        )

    print("Saving report and metrics JSON...")
    write_robust_inference_report(
        output_path=reports_dir / "robust_inference_report.md",
        selected_threshold=selected_threshold,
        detector_overall=detector_overall,
        detector_by_variant=detector_by_variant,
        comparison_df=comparison,
    )

    save_json(
        {
            "selected_threshold": selected_threshold,
            "detector_overall": detector_overall.to_dict(orient="records"),
            "detector_by_variant": detector_by_variant.to_dict(orient="records"),
            "comparison": comparison.to_dict(orient="records"),
        },
        metrics_dir / "robust_inference_metrics.json",
    )

    print("\nStage 8 robust inference evaluation completed.")
    print(f"Predictions: {predictions_dir / 'robust_inference_predictions.csv'}")
    print(f"Results: {tables_dir / 'robust_inference_results.csv'}")
    print(f"Comparison: {tables_dir / 'robust_inference_comparison.csv'}")
    print(f"Detector by variant: {tables_dir / 'robust_inference_detector_by_variant.csv'}")
    print(f"Report: {reports_dir / 'robust_inference_report.md'}")


if __name__ == "__main__":
    main()
