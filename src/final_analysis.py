from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir, save_json


SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def load_required_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["model_family"] = output["model"].apply(
        lambda model: "phobert" if str(model) == "phobert" else "baseline"
    )
    return output


def combine_robustness_results(
    baseline_results: pd.DataFrame,
    phobert_results: pd.DataFrame,
    variant_order: List[str],
) -> pd.DataFrame:
    """
    Gộp robustness results của baseline và PhoBERT.

    Đầu vào:
        baseline_results: output từ Stage 5.
        phobert_results: output từ Stage 6.

    Đầu ra:
        Một bảng thống nhất theo task/model/variant.
    """
    baseline = add_model_family(baseline_results)
    phobert = add_model_family(phobert_results)

    common_cols = [
        "task",
        "model",
        "model_family",
        "variant",
        "noise_type",
        "noise_level",
        "num_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    combined = pd.concat(
        [
            baseline[common_cols],
            phobert[common_cols],
        ],
        ignore_index=True,
    )

    combined = add_variant_order(combined, variant_order)
    combined = combined.sort_values(["task", "model_family", "model", "variant_order"])
    return combined.reset_index(drop=True)


def add_variant_order(df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    output = df.copy()
    order_map = {variant: idx for idx, variant in enumerate(variant_order)}
    output["variant_order"] = output["variant"].map(order_map).fillna(999).astype(int)
    return output


def compute_drop_from_clean(results_df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    """
    Tính robustness drop cho bảng kết quả đã gộp.

    Drop được tính riêng theo từng task/model.
    """
    rows = []

    for (task, model), group_df in results_df.groupby(["task", "model"]):
        clean_rows = group_df[group_df["variant"] == "clean"]
        if len(clean_rows) != 1:
            raise ValueError(
                f"Expected exactly one clean row for task={task}, model={model}, "
                f"found {len(clean_rows)}"
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
                    "model_family": row["model_family"],
                    "variant": row["variant"],
                    "noise_type": row["noise_type"],
                    "noise_level": row["noise_level"],
                    "num_samples": row["num_samples"],
                    "clean_accuracy": clean["accuracy"],
                    "variant_accuracy": row["accuracy"],
                    "accuracy_drop": round(accuracy_drop, 6),
                    "accuracy_relative_drop_percent": safe_relative_drop(clean["accuracy"], accuracy_drop),
                    "clean_macro_f1": clean["macro_f1"],
                    "variant_macro_f1": row["macro_f1"],
                    "macro_f1_drop": round(macro_f1_drop, 6),
                    "macro_f1_relative_drop_percent": safe_relative_drop(clean["macro_f1"], macro_f1_drop),
                    "clean_weighted_f1": clean["weighted_f1"],
                    "variant_weighted_f1": row["weighted_f1"],
                    "weighted_f1_drop": round(weighted_f1_drop, 6),
                    "weighted_f1_relative_drop_percent": safe_relative_drop(clean["weighted_f1"], weighted_f1_drop),
                }
            )

    drop_df = pd.DataFrame(rows)
    drop_df = add_variant_order(drop_df, variant_order)
    drop_df = drop_df.sort_values(["task", "model_family", "model", "variant_order"])
    return drop_df.reset_index(drop=True)


def safe_relative_drop(clean_value: float, drop_value: float) -> float:
    clean_value = float(clean_value)
    if clean_value == 0:
        return 0.0
    return round(float(drop_value / clean_value * 100), 4)


def create_clean_model_comparison(combined_results: pd.DataFrame, model_order: List[str]) -> pd.DataFrame:
    """
    Tạo bảng so sánh clean test giữa các mô hình.
    """
    clean_df = combined_results[combined_results["variant"] == "clean"].copy()

    model_order_map = {model: idx for idx, model in enumerate(model_order)}
    clean_df["model_order"] = clean_df["model"].map(model_order_map).fillna(999).astype(int)

    clean_df["rank_macro_f1"] = (
        clean_df.groupby("task")["macro_f1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    clean_df["rank_accuracy"] = (
        clean_df.groupby("task")["accuracy"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    clean_df["is_best_clean_macro_f1"] = clean_df["rank_macro_f1"] == 1

    clean_df = clean_df.sort_values(["task", "rank_macro_f1", "model_order"])

    return clean_df[
        [
            "task",
            "model",
            "model_family",
            "variant",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "rank_macro_f1",
            "rank_accuracy",
            "is_best_clean_macro_f1",
        ]
    ].reset_index(drop=True)


def create_final_robustness_comparison(
    combined_results: pd.DataFrame,
    combined_drop: pd.DataFrame,
    variant_order: List[str],
    model_order: List[str],
) -> pd.DataFrame:
    """
    Tạo bảng robustness cuối cùng gồm score và drop.
    """
    score_cols = [
        "task",
        "model",
        "model_family",
        "variant",
        "noise_type",
        "noise_level",
        "num_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    drop_cols = [
        "task",
        "model",
        "variant",
        "clean_accuracy",
        "variant_accuracy",
        "accuracy_drop",
        "accuracy_relative_drop_percent",
        "clean_macro_f1",
        "variant_macro_f1",
        "macro_f1_drop",
        "macro_f1_relative_drop_percent",
        "clean_weighted_f1",
        "variant_weighted_f1",
        "weighted_f1_drop",
        "weighted_f1_relative_drop_percent",
    ]

    merged = combined_results[score_cols].merge(
        combined_drop[drop_cols],
        on=["task", "model", "variant"],
        how="left",
    )

    merged["rank_macro_f1_within_variant"] = (
        merged.groupby(["task", "variant"])["macro_f1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    merged["is_best_model_for_variant"] = merged["rank_macro_f1_within_variant"] == 1

    merged = add_variant_order(merged, variant_order)
    model_order_map = {model: idx for idx, model in enumerate(model_order)}
    merged["model_order"] = merged["model"].map(model_order_map).fillna(999).astype(int)

    merged = merged.sort_values(
        ["task", "variant_order", "rank_macro_f1_within_variant", "model_order"]
    )

    return merged.reset_index(drop=True)


def combine_class_reports(
    baseline_report: pd.DataFrame,
    phobert_report: pd.DataFrame,
    variant_order: List[str],
    model_order: List[str],
) -> pd.DataFrame:
    """
    Gộp classification report của baseline và PhoBERT.
    """
    baseline = add_model_family(baseline_report)
    phobert = add_model_family(phobert_report)

    combined = pd.concat([baseline, phobert], ignore_index=True)

    combined["is_summary_label"] = combined["label"].isin(SUMMARY_LABELS)
    combined = add_variant_order(combined, variant_order)

    model_order_map = {model: idx for idx, model in enumerate(model_order)}
    combined["model_order"] = combined["model"].map(model_order_map).fillna(999).astype(int)

    combined = combined.sort_values(
        ["task", "variant_order", "model_order", "is_summary_label", "label"]
    )

    return combined.reset_index(drop=True)


def create_error_examples(
    phobert_predictions: pd.DataFrame,
    task_name: str,
    target_variants: Iterable[str],
    max_examples_per_variant: int,
    seed: int,
) -> pd.DataFrame:
    """
    Lấy các mẫu PhoBERT dự đoán đúng ở clean nhưng sai ở noisy.

    Đây là nhóm lỗi quan trọng nhất cho robustness:
        clean_correct_noisy_wrong
    """
    task_df = phobert_predictions[phobert_predictions["task"] == task_name].copy()

    if task_df.empty:
        return pd.DataFrame()

    clean_df = task_df[task_df["variant"] == "clean"].copy()

    clean_cols = [
        "original_id",
        "y_pred",
        "y_pred_name",
        "is_correct",
    ]

    clean_df = clean_df[clean_cols].rename(
        columns={
            "y_pred": "clean_y_pred",
            "y_pred_name": "clean_y_pred_name",
            "is_correct": "clean_is_correct",
        }
    )

    outputs = []

    for variant in target_variants:
        variant_df = task_df[task_df["variant"] == variant].copy()

        if variant_df.empty:
            continue

        merged = variant_df.merge(clean_df, on="original_id", how="left")
        merged = merged[
            (merged["clean_is_correct"] == True) &
            (merged["is_correct"] == False)
        ].copy()

        if merged.empty:
            continue

        merged["error_type"] = "clean_correct_noisy_wrong"

        sample_n = min(max_examples_per_variant, len(merged))
        merged = merged.sample(n=sample_n, random_state=seed)

        keep_cols = [
            "task",
            "variant",
            "noise_type",
            "noise_level",
            "original_id",
            "original_text",
            "text",
            "y_true",
            "y_true_name",
            "clean_y_pred",
            "clean_y_pred_name",
            "y_pred",
            "y_pred_name",
            "error_type",
            "sentiment_label",
            "sentiment_name",
            "topic_label",
            "topic_name",
        ]

        keep_cols = [col for col in keep_cols if col in merged.columns]
        outputs.append(merged[keep_cols])

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def create_transition_summary(
    phobert_predictions: pd.DataFrame,
    task_name: str,
    target_variants: Iterable[str],
) -> pd.DataFrame:
    """
    Tóm tắt hướng nhầm lớp của PhoBERT trên noisy variants.
    """
    task_df = phobert_predictions[phobert_predictions["task"] == task_name].copy()
    task_df = task_df[task_df["variant"].isin(target_variants)].copy()

    if task_df.empty:
        return pd.DataFrame()

    wrong_df = task_df[task_df["is_correct"] == False].copy()

    if wrong_df.empty:
        return pd.DataFrame()

    summary = (
        wrong_df.groupby(["task", "variant", "y_true_name", "y_pred_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["task", "variant", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    summary["total_wrong_in_variant"] = summary.groupby(["task", "variant"])["count"].transform("sum")
    summary["percent_within_wrong"] = (summary["count"] / summary["total_wrong_in_variant"] * 100).round(4)

    return summary


def plot_clean_macro_f1_comparison(
    clean_comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Vẽ Macro-F1 clean của tất cả mô hình.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = clean_comparison.copy()
    plot_df["x_label"] = plot_df["task"] + "\\n" + plot_df["model"]

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["x_label"], plot_df["macro_f1"])
    plt.title("Clean Test Macro-F1 Comparison")
    plt.xlabel("Task / Model")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(1.0, float(plot_df["macro_f1"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_robustness_macro_f1(
    robustness_comparison: pd.DataFrame,
    task_name: str,
    variant_order: List[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = robustness_comparison[robustness_comparison["task"] == task_name].copy()

    plt.figure(figsize=(12, 6))

    for model_name, model_df in plot_df.groupby("model"):
        model_df = model_df.set_index("variant").reindex(variant_order).reset_index()
        plt.plot(model_df["variant"], model_df["macro_f1"], marker="o", label=model_name)

    plt.title(f"Macro-F1 by Variant - {task_name}")
    plt.xlabel("Variant")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_robustness_drop(
    robustness_comparison: pd.DataFrame,
    task_name: str,
    variant_order: List[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = robustness_comparison[robustness_comparison["task"] == task_name].copy()

    plt.figure(figsize=(12, 6))

    for model_name, model_df in plot_df.groupby("model"):
        model_df = model_df.set_index("variant").reindex(variant_order).reset_index()
        plt.plot(model_df["variant"], model_df["macro_f1_drop"], marker="o", label=model_name)

    plt.title(f"Macro-F1 Drop from Clean - {task_name}")
    plt.xlabel("Variant")
    plt.ylabel("Macro-F1 drop")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_final_comparison_report(
    report_path: str | Path,
    clean_comparison: pd.DataFrame,
    robustness_comparison: pd.DataFrame,
    per_class_comparison: pd.DataFrame,
    sentiment_errors: pd.DataFrame,
    topic_errors: pd.DataFrame,
) -> None:
    """
    Ghi báo cáo markdown cuối cho Stage 7.
    """
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    best_clean = (
        clean_comparison.sort_values(["task", "rank_macro_f1"])
        .groupby("task")
        .head(1)
    )

    worst_drop = (
        robustness_comparison[robustness_comparison["variant"] != "clean"]
        .sort_values(["task", "model", "macro_f1_drop"], ascending=[True, True, False])
        .groupby(["task", "model"])
        .head(1)
        .reset_index(drop=True)
    )

    no_accent = robustness_comparison[
        robustness_comparison["variant"].isin(["no_accent", "mixed_no_accent"])
    ].copy()

    lines: List[str] = []

    lines.append("# Final Comparison and Error Analysis Report")
    lines.append("")
    lines.append("## 1. Mục tiêu")
    lines.append("")
    lines.append(
        "Stage 7 tổng hợp kết quả của baseline và PhoBERT trên clean/noisy test, "
        "tính robustness drop và trích xuất các ví dụ lỗi tiêu biểu."
    )
    lines.append("")

    lines.append("## 2. Best clean models")
    lines.append("")
    lines.append("```text")
    lines.append(best_clean.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 3. Full clean comparison")
    lines.append("")
    lines.append("```text")
    lines.append(clean_comparison.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 4. Robustness comparison")
    lines.append("")
    lines.append("```text")
    lines.append(robustness_comparison.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 5. No-accent focused comparison")
    lines.append("")
    lines.append("```text")
    lines.append(no_accent.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 6. Worst drop per model")
    lines.append("")
    lines.append("```text")
    lines.append(worst_drop.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 7. Error examples summary")
    lines.append("")
    lines.append(f"- Sentiment error examples: {len(sentiment_errors)}")
    lines.append(f"- Topic error examples: {len(topic_errors)}")
    lines.append("")

    lines.append("## 8. Notes")
    lines.append("")
    lines.append("- Clean test được dùng để đánh giá hiệu năng in-distribution.")
    lines.append("- Noisy test được dùng để đánh giá robustness dưới nhiễu có kiểm soát.")
    lines.append("- Noisy variants không được dùng để train hoặc chọn checkpoint.")
    lines.append("- Macro-F1 là metric chính vì dữ liệu mất cân bằng.")
    lines.append("- Cần đọc error examples trước khi viết kết luận cuối trong báo cáo.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_final_metrics_json(
    output_path: str | Path,
    clean_comparison: pd.DataFrame,
    robustness_comparison: pd.DataFrame,
) -> None:
    payload = {
        "clean_comparison": clean_comparison.to_dict(orient="records"),
        "robustness_comparison": robustness_comparison.to_dict(orient="records"),
    }
    save_json(payload, output_path)
