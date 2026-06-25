from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import (
    build_baseline_model,
    flatten_classification_report,
    save_confusion_matrix_plot,
    save_model,
    train_and_evaluate_baseline,
    write_baseline_report,
)
from src.utils import ensure_dir, load_json, load_yaml, project_root, save_json, set_seed


def _load_label_names(mapping_path: Path) -> List[str]:
    """
    Load label names theo thứ tự id tăng dần.
    """
    mapping = load_json(mapping_path)
    id2label = mapping["id2label"]

    return [
        id2label[str(i)]
        for i in sorted([int(k) for k in id2label.keys()])
    ]


def main() -> None:
    root = project_root()

    data_config = load_yaml(root / "configs" / "data.yaml")
    baseline_config = load_yaml(root / "configs" / "baseline.yaml")

    seed = int(baseline_config["seed"])
    set_seed(seed)

    processed_dir = root / data_config["paths"]["processed_dir"]
    models_dir = ensure_dir(root / data_config["paths"]["outputs_dir"] / "models" / "baseline")
    tables_dir = ensure_dir(root / data_config["paths"]["tables_dir"])
    metrics_dir = ensure_dir(root / data_config["paths"]["metrics_dir"])
    predictions_dir = ensure_dir(root / data_config["paths"]["predictions_dir"])
    figures_dir = ensure_dir(root / data_config["paths"]["figures_dir"])
    reports_dir = ensure_dir(root / data_config["paths"]["reports_dir"])
    mappings_dir = root / data_config["paths"]["mappings_dir"]

    train_path = processed_dir / "train_standardized.csv"
    validation_path = processed_dir / "validation_standardized.csv"
    test_path = processed_dir / "test_standardized.csv"

    print("Loading standardized splits...")
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    sentiment_label_names = _load_label_names(mappings_dir / "sentiment_label_mapping.json")
    topic_label_names = _load_label_names(mappings_dir / "topic_label_mapping.json")

    tasks = {
        "sentiment": {
            "label_col": "sentiment_label",
            "label_names": sentiment_label_names,
        },
        "topic": {
            "label_col": "topic_label",
            "label_names": topic_label_names,
        },
    }

    model_names = [
        model_name
        for model_name, model_cfg in baseline_config["models"].items()
        if model_cfg.get("enabled", True)
    ]

    all_result_rows = []
    all_report_rows = []
    all_predictions = []
    metrics_json: Dict[str, object] = {}

    for task_name, task_cfg in tasks.items():
        label_col = task_cfg["label_col"]
        label_names = task_cfg["label_names"]

        print(f"\nTask: {task_name}")
        print(f"Labels: {label_names}")

        for model_name in model_names:
            print(f"  Training baseline: {model_name}")

            # Một model train riêng cho validation để kiểm tra,
            # sau đó train lại model riêng cho test để lưu artifact cuối.
            model_for_validation = build_baseline_model(model_name, seed=seed)
            validation_result = train_and_evaluate_baseline(
                model=model_for_validation,
                model_name=model_name,
                task_name=task_name,
                train_df=train_df,
                eval_df=validation_df,
                text_col="text",
                label_col=label_col,
                split_name="validation",
                label_names=label_names,
            )

            model_for_test = build_baseline_model(model_name, seed=seed)
            test_result = train_and_evaluate_baseline(
                model=model_for_test,
                model_name=model_name,
                task_name=task_name,
                train_df=train_df,
                eval_df=test_df,
                text_col="text",
                label_col=label_col,
                split_name="test",
                label_names=label_names,
            )

            for split_name, result in [
                ("validation", validation_result),
                ("test", test_result),
            ]:
                metrics = result["metrics"]

                row = {
                    "task": task_name,
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
                all_result_rows.append(row)

                all_report_rows.append(
                    flatten_classification_report(
                        report_dict=result["classification_report"],
                        task_name=task_name,
                        model_name=model_name,
                        split_name=split_name,
                    )
                )

                all_predictions.append(result["predictions"])

                metrics_json[f"{task_name}/{model_name}/{split_name}"] = {
                    "metrics": metrics,
                    "classification_report": result["classification_report"],
                }

                cm_path = (
                    figures_dir
                    / f"confusion_matrix_baseline_{task_name}_{model_name}_{split_name}.png"
                )
                save_confusion_matrix_plot(
                    cm=result["confusion_matrix"],
                    label_names=label_names,
                    title=f"{task_name} - {model_name} - {split_name}",
                    output_path=cm_path,
                )

            model_path = models_dir / f"{task_name}_{model_name}.joblib"
            save_model(model_for_test, model_path)

            print(
                f"    Test Macro-F1: "
                f"{test_result['metrics']['macro_f1']:.4f} | "
                f"Accuracy: {test_result['metrics']['accuracy']:.4f}"
            )

    results_df = pd.DataFrame(all_result_rows)
    results_df = results_df.sort_values(["task", "split", "macro_f1"], ascending=[True, True, False])

    report_df = pd.concat(all_report_rows, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_df.to_csv(
        tables_dir / "baseline_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_df.to_csv(
        tables_dir / "baseline_classification_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions_df.to_csv(
        predictions_dir / "baseline_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(metrics_json, metrics_dir / "baseline_results.json")

    write_baseline_report(
        report_path=reports_dir / "baseline_report.md",
        results_df=results_df,
    )

    print("\nBaseline training completed.")
    print(f"Results: {tables_dir / 'baseline_results.csv'}")
    print(f"Classification report: {tables_dir / 'baseline_classification_report.csv'}")
    print(f"Predictions: {predictions_dir / 'baseline_predictions.csv'}")
    print(f"Metrics JSON: {metrics_dir / 'baseline_results.json'}")
    print(f"Models: {models_dir}")
    print(f"Report: {reports_dir / 'baseline_report.md'}")


if __name__ == "__main__":
    main()