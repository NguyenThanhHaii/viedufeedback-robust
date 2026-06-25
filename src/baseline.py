from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.metrics import classification_metrics
from src.utils import ensure_dir, save_json


def build_majority_classifier(seed: int = 42) -> DummyClassifier:
    """
    Majority baseline.

    Mục đích:
        Tạo mốc thấp nhất hợp lý: luôn dự đoán lớp xuất hiện nhiều nhất.

    Ý nghĩa:
        Nếu model phức tạp không vượt qua baseline này về Macro-F1,
        pipeline có vấn đề nghiêm trọng.
    """
    return DummyClassifier(strategy="most_frequent", random_state=seed)


def build_tfidf_word_svm(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int = 50000,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """
    TF-IDF word-level + Linear SVM.

    Mục đích:
        Đại diện cho baseline truyền thống dựa trên từ/cụm từ.

    Lý do:
        Với text classification, TF-IDF + Linear SVM thường là baseline mạnh,
        dễ chạy, dễ giải thích và không cần GPU.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_tfidf_char_svm(
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int = 80000,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """
    TF-IDF char-level + Linear SVM.

    Mục đích:
        Đại diện cho baseline dựa trên n-gram ký tự.

    Lý do:
        Char-level features có thể hữu ích khi văn bản có lỗi chính tả,
        thiếu dấu, kéo dài ký tự hoặc biến thể không chuẩn.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_baseline_model(
    model_name: str,
    seed: int = 42,
) -> object:
    """
    Tạo model theo tên thống nhất.
    """
    if model_name == "majority":
        return build_majority_classifier(seed=seed)

    if model_name == "tfidf_word_svm":
        return build_tfidf_word_svm(seed=seed)

    if model_name == "tfidf_char_svm":
        return build_tfidf_char_svm(seed=seed)

    raise ValueError(f"Unknown baseline model: {model_name}")


def train_and_evaluate_baseline(
    model,
    model_name: str,
    task_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    split_name: str,
    label_names: List[str],
) -> Dict[str, object]:
    """
    Train model trên train_df và evaluate trên eval_df.

    Đầu ra:
        metrics, predictions, classification report và confusion matrix.
    """
    x_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df[label_col].astype(int)

    x_eval = eval_df[text_col].fillna("").astype(str)
    y_eval = eval_df[label_col].astype(int)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)

    metrics = classification_metrics(
        y_true=y_eval.tolist(),
        y_pred=y_pred.tolist(),
        label_names=label_names,
    )

    report_dict = classification_report(
        y_eval,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_eval,
        y_pred,
        labels=list(range(len(label_names))),
    )

    predictions_df = eval_df[["id", "split", "text", label_col]].copy()
    predictions_df["task"] = task_name
    predictions_df["model"] = model_name
    predictions_df["eval_split"] = split_name
    predictions_df["y_true"] = y_eval.values
    predictions_df["y_pred"] = y_pred
    predictions_df["y_true_name"] = predictions_df["y_true"].map(
        {i: label for i, label in enumerate(label_names)}
    )
    predictions_df["y_pred_name"] = predictions_df["y_pred"].map(
        {i: label for i, label in enumerate(label_names)}
    )
    predictions_df["is_correct"] = predictions_df["y_true"] == predictions_df["y_pred"]

    return {
        "model": model,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "predictions": predictions_df,
    }


def flatten_classification_report(
    report_dict: Dict[str, object],
    task_name: str,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Chuyển classification_report dạng dict sang DataFrame dài.
    """
    rows = []

    for label, values in report_dict.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "split": split_name,
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
                    "split": split_name,
                    "label": label,
                    "precision": None,
                    "recall": None,
                    "f1_score": values,
                    "support": None,
                }
            )

    return pd.DataFrame(rows)


def save_confusion_matrix_plot(
    cm,
    label_names: List[str],
    title: str,
    output_path: str | Path,
) -> None:
    """
    Lưu confusion matrix thành ảnh.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )
    display.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model(model, output_path: str | Path) -> None:
    """
    Lưu model scikit-learn bằng joblib.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    joblib.dump(model, output_path)


def write_baseline_report(
    report_path: str | Path,
    results_df: pd.DataFrame,
) -> None:
    """
    Ghi baseline report dạng Markdown.
    """
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = []

    lines.append("# Baseline Report")
    lines.append("")
    lines.append("## 1. Mục tiêu")
    lines.append("")
    lines.append(
        "Giai đoạn này huấn luyện và đánh giá các baseline truyền thống "
        "cho hai tác vụ sentiment classification và topic classification."
    )
    lines.append("")
    lines.append("Các mô hình gồm:")
    lines.append("")
    lines.append("- Majority Class")
    lines.append("- TF-IDF word-level + Linear SVM")
    lines.append("- TF-IDF char-level + Linear SVM")
    lines.append("")
    lines.append("## 2. Kết quả tổng hợp")
    lines.append("")
    lines.append(results_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Ghi chú")
    lines.append("")
    lines.append("- Macro-F1 là metric chính vì dữ liệu mất cân bằng.")
    lines.append("- Accuracy chỉ dùng làm metric phụ.")
    lines.append("- Kết quả baseline là mốc đối chứng trước khi fine-tune PhoBERT.")
    lines.append("- Chưa kết luận về PhoBERT trong giai đoạn này.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")