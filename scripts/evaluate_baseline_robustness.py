from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robustness import (
    compute_robustness_drop,
    evaluate_model_on_variants,
    load_model,
    plot_macro_f1_by_variant,
    plot_macro_f1_drop_by_variant,
    save_robustness_json,
    write_robustness_report,
)
from src.utils import ensure_dir, load_json, load_yaml, project_root, set_seed


def _load_label_names(mapping_path: Path) -> List[str]:
    mapping = load_json(mapping_path)
    id2label = mapping["id2label"]

    return [
        id2label[str(i)]
        for i in sorted([int(k) for k in id2label.keys()])
    ]


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "robustness.yaml")

    seed = int(config["seed"])
    set_seed(seed)

    eval_file = root / config["data"]["eval_file"]
    text_col = config["data"]["text_col"]

    baseline_models_dir = root / config["paths"]["baseline_models_dir"]
    mappings_dir = root / config["paths"]["mappings_dir"]
    tables_dir = ensure_dir(root / config["paths"]["tables_dir"])
    metrics_dir = ensure_dir(root / config["paths"]["metrics_dir"])
    predictions_dir = ensure_dir(root / config["paths"]["predictions_dir"])
    figures_dir = ensure_dir(root / config["paths"]["figures_dir"])
    reports_dir = ensure_dir(root / config["paths"]["reports_dir"])

    variant_order = config["variant_order"]

    print(f"Loading eval data: {eval_file}")
    eval_df = pd.read_csv(eval_file)

    required_cols = [
        "id",
        "original_id",
        "variant",
        "noise_type",
        "noise_level",
        "original_text",
        text_col,
        "sentiment_label",
        "sentiment_name",
        "topic_label",
        "topic_name",
    ]
    missing_cols = [col for col in required_cols if col not in eval_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in eval data: {missing_cols}")

    print("Variants:")
    print(eval_df.groupby("variant").size())

    all_results = []
    all_reports = []
    all_predictions = []

    for task_name, task_cfg in config["tasks"].items():
        label_col = task_cfg["label_col"]
        mapping_file = task_cfg["mapping_file"]
        label_names = _load_label_names(mappings_dir / mapping_file)

        print(f"\nTask: {task_name}")
        print(f"Labels: {label_names}")

        for model_name in task_cfg["models"]:
            model_path = baseline_models_dir / f"{task_name}_{model_name}.joblib"
            print(f"  Loading model: {model_path}")

            model = load_model(model_path)

            evaluated = evaluate_model_on_variants(
                model=model,
                eval_df=eval_df,
                task_name=task_name,
                model_name=model_name,
                text_col=text_col,
                label_col=label_col,
                label_names=label_names,
                variant_order=variant_order,
            )

            all_results.append(evaluated["results"])
            all_reports.append(evaluated["classification_report"])
            all_predictions.append(evaluated["predictions"])

            clean_row = evaluated["results"][
                evaluated["results"]["variant"] == "clean"
            ].iloc[0]

            print(
                f"    clean Macro-F1={clean_row['macro_f1']:.4f}, "
                f"Accuracy={clean_row['accuracy']:.4f}"
            )

    results_df = pd.concat(all_results, ignore_index=True)
    report_df = pd.concat(all_reports, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    drop_df = compute_robustness_drop(results_df)

    results_df = results_df.sort_values(
        ["task", "model", "variant"],
        ascending=[True, True, True],
    )
    drop_df = drop_df.sort_values(
        ["task", "model", "variant"],
        ascending=[True, True, True],
    )

    results_path = tables_dir / "baseline_robustness_results.csv"
    drop_path = tables_dir / "baseline_robustness_drop.csv"
    report_path = tables_dir / "baseline_robustness_class_report.csv"
    predictions_path = predictions_dir / "baseline_robustness_predictions.csv"
    metrics_path = metrics_dir / "baseline_robustness_results.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    drop_df.to_csv(drop_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    save_robustness_json(
        output_path=metrics_path,
        results_df=results_df,
        drop_df=drop_df,
    )

    for task_name in config["tasks"].keys():
        plot_macro_f1_by_variant(
            results_df=results_df,
            task_name=task_name,
            variant_order=variant_order,
            output_path=figures_dir / f"baseline_robustness_macro_f1_{task_name}.png",
        )

        plot_macro_f1_drop_by_variant(
            drop_df=drop_df,
            task_name=task_name,
            variant_order=variant_order,
            output_path=figures_dir / f"baseline_robustness_macro_f1_drop_{task_name}.png",
        )

    write_robustness_report(
        report_path=reports_dir / "baseline_robustness_report.md",
        results_df=results_df,
        drop_df=drop_df,
    )

    print("\nBaseline robustness evaluation completed.")
    print(f"Results: {results_path}")
    print(f"Drop: {drop_path}")
    print(f"Class report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Report: {reports_dir / 'baseline_robustness_report.md'}")


if __name__ == "__main__":
    main()