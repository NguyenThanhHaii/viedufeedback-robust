from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from src.metrics import classification_metrics
from src.utils import ensure_dir, save_json


def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def evaluate_model_on_variants(
    model,
    eval_df: pd.DataFrame,
    task_name: str,
    model_name: str,
    text_col: str,
    label_col: str,
    label_names: List[str],
    variant_order: Iterable[str] | None = None,
) -> Dict[str, object]:
    """
    Evaluate một baseline model trên từng noisy variant.

    Không train lại model.
    """
    result_rows = []
    report_rows = []
    prediction_rows = []

    variants = list(eval_df["variant"].dropna().unique())

    if variant_order is not None:
        ordered = [v for v in variant_order if v in variants]
        remaining = sorted([v for v in variants if v not in ordered])
        variants = ordered + remaining
    else:
        variants = sorted(variants)

    for variant in variants:
        variant_df = eval_df[eval_df["variant"] == variant].copy()

        x = variant_df[text_col].fillna("").astype(str)
        y_true = variant_df[label_col].astype(int)
        y_pred = model.predict(x)

        metrics = classification_metrics(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            label_names=label_names,
        )

        result_rows.append(
            {
                "task": task_name,
                "model": model_name,
                "variant": variant,
                "noise_type": variant_df["noise_type"].iloc[0],
                "noise_level": variant_df["noise_level"].iloc[0],
                "num_samples": int(len(variant_df)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        report_rows.extend(
            flatten_report(
                report_dict=report_dict,
                task_name=task_name,
                model_name=model_name,
                variant=variant,
            )
        )

        prediction_cols = [
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

        # Tránh duplicate column khi label_col là sentiment_label hoặc topic_label.
        prediction_cols = list(dict.fromkeys(prediction_cols))

        pred_df = variant_df[prediction_cols].copy()

        pred_df["task"] = task_name
        pred_df["model"] = model_name
        pred_df["y_true"] = y_true.values
        pred_df["y_pred"] = y_pred
        pred_df["y_true_name"] = pred_df["y_true"].map(
            {i: label for i, label in enumerate(label_names)}
        )
        pred_df["y_pred_name"] = pred_df["y_pred"].map(
            {i: label for i, label in enumerate(label_names)}
        )
        pred_df["is_correct"] = pred_df["y_true"] == pred_df["y_pred"]

        prediction_rows.append(pred_df)

    return {
        "results": pd.DataFrame(result_rows),
        "classification_report": pd.DataFrame(report_rows),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }


def flatten_report(
    report_dict: Dict[str, object],
    task_name: str,
    model_name: str,
    variant: str,
) -> List[Dict[str, object]]:
    rows = []

    for label, values in report_dict.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "variant": variant,
                    "label": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
        else:
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "variant": variant,
                    "label": label,
                    "precision": None,
                    "recall": None,
                    "f1_score": values,
                    "support": None,
                }
            )

    return rows


def compute_robustness_drop(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính mức giảm metric so với clean test.

    drop = clean_metric - noisy_metric
    relative_drop_percent = drop / clean_metric * 100
    """
    rows = []

    for (task, model), group_df in results_df.groupby(["task", "model"]):
        clean_rows = group_df[group_df["variant"] == "clean"]

        if len(clean_rows) != 1:
            raise ValueError(
                f"Expected exactly one clean row for task={task}, model={model}, "
                f"but found {len(clean_rows)}"
            )

        clean = clean_rows.iloc[0]

        for _, row in group_df.iterrows():
            accuracy_drop = float(clean["accuracy"] - row["accuracy"])
            macro_f1_drop = float(clean["macro_f1"] - row["macro_f1"])
            weighted_f1_drop = float(clean["weighted_f1"] - row["weighted_f1"])

            rows.append(
                {
                    "task": task,
                    "model": model,
                    "variant": row["variant"],
                    "noise_type": row["noise_type"],
                    "noise_level": row["noise_level"],
                    "clean_accuracy": clean["accuracy"],
                    "variant_accuracy": row["accuracy"],
                    "accuracy_drop": accuracy_drop,
                    "accuracy_relative_drop_percent": safe_relative_drop(
                        clean["accuracy"], accuracy_drop
                    ),
                    "clean_macro_f1": clean["macro_f1"],
                    "variant_macro_f1": row["macro_f1"],
                    "macro_f1_drop": macro_f1_drop,
                    "macro_f1_relative_drop_percent": safe_relative_drop(
                        clean["macro_f1"], macro_f1_drop
                    ),
                    "clean_weighted_f1": clean["weighted_f1"],
                    "variant_weighted_f1": row["weighted_f1"],
                    "weighted_f1_drop": weighted_f1_drop,
                    "weighted_f1_relative_drop_percent": safe_relative_drop(
                        clean["weighted_f1"], weighted_f1_drop
                    ),
                }
            )

    return pd.DataFrame(rows)


def safe_relative_drop(clean_value: float, drop_value: float) -> float:
    if clean_value == 0:
        return 0.0
    return round(float(drop_value / clean_value * 100), 4)


def _ordered_variants(df: pd.DataFrame, variant_order: List[str]) -> List[str]:
    variants = list(df["variant"].dropna().unique())
    ordered = [v for v in variant_order if v in variants]
    remaining = sorted([v for v in variants if v not in ordered])
    return ordered + remaining


def plot_macro_f1_by_variant(
    results_df: pd.DataFrame,
    task_name: str,
    variant_order: List[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = results_df[results_df["task"] == task_name].copy()
    variants = _ordered_variants(plot_df, variant_order)

    plt.figure(figsize=(11, 6))

    for model_name, model_df in plot_df.groupby("model"):
        model_df = model_df.set_index("variant").reindex(variants).reset_index()
        plt.plot(model_df["variant"], model_df["macro_f1"], marker="o", label=model_name)

    plt.title(f"Baseline Macro-F1 by Variant - {task_name}")
    plt.xlabel("Variant")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_macro_f1_drop_by_variant(
    drop_df: pd.DataFrame,
    task_name: str,
    variant_order: List[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = drop_df[drop_df["task"] == task_name].copy()
    variants = _ordered_variants(plot_df, variant_order)

    plt.figure(figsize=(11, 6))

    for model_name, model_df in plot_df.groupby("model"):
        model_df = model_df.set_index("variant").reindex(variants).reset_index()
        plt.plot(
            model_df["variant"],
            model_df["macro_f1_drop"],
            marker="o",
            label=model_name,
        )

    plt.title(f"Baseline Macro-F1 Drop by Variant - {task_name}")
    plt.xlabel("Variant")
    plt.ylabel("Macro-F1 drop from clean")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_robustness_report(
    report_path: str | Path,
    results_df: pd.DataFrame,
    drop_df: pd.DataFrame,
) -> None:
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    clean_best = (
        results_df[results_df["variant"] == "clean"]
        .sort_values(["task", "macro_f1"], ascending=[True, False])
        .groupby("task")
        .head(1)
    )

    worst_drops = (
        drop_df[drop_df["variant"] != "clean"]
        .sort_values(["task", "macro_f1_drop"], ascending=[True, False])
        .groupby(["task", "model"])
        .head(1)
        .sort_values(["task", "model"])
    )

    lines = []

    lines.append("# Baseline Robustness Evaluation Report")
    lines.append("")
    lines.append("## 1. Mục tiêu")
    lines.append("")
    lines.append(
        "Giai đoạn này đánh giá các baseline đã train trên clean training set "
        "trên các phiên bản clean/noisy test set."
    )
    lines.append("")
    lines.append("Không train lại model trong giai đoạn này.")
    lines.append("")

    lines.append("## 2. Clean baseline reference")
    lines.append("")
    lines.append("```text")
    lines.append(clean_best.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 3. Full robustness results")
    lines.append("")
    lines.append("```text")
    lines.append(results_df.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 4. Robustness drop from clean")
    lines.append("")
    lines.append("```text")
    lines.append(drop_df.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 5. Worst drop per model")
    lines.append("")
    lines.append("```text")
    lines.append(worst_drops.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- Macro-F1 là metric chính vì dữ liệu mất cân bằng.")
    lines.append("- Robustness drop được tính so với clean test cùng model và cùng task.")
    lines.append("- Nếu noisy variant có Macro-F1 cao hơn clean, drop có thể âm.")
    lines.append("- Kết quả này là mốc đối chứng trước khi đánh giá PhoBERT.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_robustness_json(
    output_path: str | Path,
    results_df: pd.DataFrame,
    drop_df: pd.DataFrame,
) -> None:
    payload = {
        "results": results_df.to_dict(orient="records"),
        "drop": drop_df.to_dict(orient="records"),
    }
    save_json(payload, output_path)