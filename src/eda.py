from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.utils import ensure_dir


def load_standardized_data(path: str | Path) -> pd.DataFrame:
    """
    Load dữ liệu đã chuẩn hóa từ Giai đoạn 1.

    Đầu vào:
        path: đường dẫn all_standardized.csv

    Đầu ra:
        DataFrame chuẩn hóa gồm id, split, text, sentiment_label,
        sentiment_name, topic_label, topic_name.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Standardized data not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "id",
        "split",
        "text",
        "sentiment_label",
        "sentiment_name",
        "topic_label",
        "topic_name",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required standardized columns: {missing_cols}")

    return df


def add_basic_text_lengths(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Thêm các đặc trưng độ dài cơ bản:
        - char_count
        - whitespace_word_count

    Lưu ý:
        whitespace_word_count không phải tách từ tiếng Việt chuẩn.
        Nó chỉ là thống kê sơ bộ theo khoảng trắng.
    """
    output = df.copy()
    text = output[text_col].fillna("").astype(str).str.strip()

    output["char_count"] = text.str.len()
    output["whitespace_word_count"] = text.apply(lambda x: len(x.split()) if x else 0)

    return output


def load_phobert_tokenizer(model_name: str):
    """
    Load tokenizer PhoBERT.

    Giai đoạn EDA dùng tokenizer để ước lượng độ dài subword.
    Việc fine-tune PhoBERT vẫn thực hiện ở giai đoạn sau trên Kaggle.
    """
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def add_phobert_subword_lengths(
    df: pd.DataFrame,
    tokenizer,
    text_col: str = "text",
    batch_size: int = 256,
    add_special_tokens: bool = True,
) -> pd.DataFrame:
    """
    Tính số subword token theo tokenizer PhoBERT.

    Đầu ra:
        thêm cột phobert_subword_count.

    Ghi chú:
        Đây là độ dài tokenizer trên text hiện tại. Nếu sau này dùng word segmentation
        trước PhoBERT, ta có thể chạy lại thống kê này trên text đã segment.
    """
    output = df.copy()
    texts = output[text_col].fillna("").astype(str).tolist()

    lengths: List[int] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="PhoBERT token lengths"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        lengths.extend([len(ids) for ids in encoded["input_ids"]])

    output["phobert_subword_count"] = lengths
    return output


def label_distribution(
    df: pd.DataFrame,
    split_col: str,
    label_col: str,
    name_col: str,
) -> pd.DataFrame:
    """
    Tính phân bố nhãn theo split.
    """
    rows = []

    for split_name, split_df in df.groupby(split_col):
        total = len(split_df)

        grouped = (
            split_df.groupby([label_col, name_col], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values([label_col, name_col])
        )

        for _, row in grouped.iterrows():
            count = int(row["count"])
            rows.append(
                {
                    "split": split_name,
                    "label_id": int(row[label_col]),
                    "label_name": row[name_col],
                    "count": count,
                    "percent": round(count / total * 100, 4) if total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng tổng quan dữ liệu theo split.
    """
    rows = []

    for split_name, split_df in df.groupby("split"):
        rows.append(
            {
                "split": split_name,
                "num_rows": int(len(split_df)),
                "num_unique_text": int(split_df["text"].nunique()),
                "empty_text": int((split_df["text"].fillna("").astype(str).str.strip() == "").sum()),
                "duplicated_text": int(split_df["text"].duplicated().sum()),
                "sentiment_num_classes": int(split_df["sentiment_label"].nunique()),
                "topic_num_classes": int(split_df["topic_label"].nunique()),
            }
        )

    rows.append(
        {
            "split": "all",
            "num_rows": int(len(df)),
            "num_unique_text": int(df["text"].nunique()),
            "empty_text": int((df["text"].fillna("").astype(str).str.strip() == "").sum()),
            "duplicated_text": int(df["text"].duplicated().sum()),
            "sentiment_num_classes": int(df["sentiment_label"].nunique()),
            "topic_num_classes": int(df["topic_label"].nunique()),
        }
    )

    return pd.DataFrame(rows)


def length_summary_by_split(
    df: pd.DataFrame,
    length_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Tóm tắt mean/std/min/max theo split cho các cột độ dài.
    """
    rows = []

    for split_name, split_df in df.groupby("split"):
        for col in length_cols:
            rows.append(
                {
                    "split": split_name,
                    "metric": col,
                    "mean": round(float(split_df[col].mean()), 4),
                    "std": round(float(split_df[col].std()), 4),
                    "min": int(split_df[col].min()),
                    "max": int(split_df[col].max()),
                }
            )

    return pd.DataFrame(rows)


def percentile_table(
    df: pd.DataFrame,
    value_col: str,
    percentiles: Iterable[int],
) -> pd.DataFrame:
    """
    Tính percentile cho một cột độ dài theo split và toàn bộ dataset.
    """
    rows = []

    def append_rows(scope_name: str, values: pd.Series) -> None:
        for p in percentiles:
            rows.append(
                {
                    "scope": scope_name,
                    "metric": value_col,
                    "percentile": p,
                    "value": float(values.quantile(p / 100)),
                }
            )

    for split_name, split_df in df.groupby("split"):
        append_rows(split_name, split_df[value_col])

    append_rows("all", df[value_col])

    return pd.DataFrame(rows)


def recommend_max_length(
    df: pd.DataFrame,
    length_col: str,
    candidates: Iterable[int],
) -> pd.DataFrame:
    """
    Tính coverage của từng max_length candidate.

    coverage = tỷ lệ mẫu có length <= candidate.
    """
    rows = []
    total = len(df)

    for candidate in candidates:
        covered = int((df[length_col] <= candidate).sum())
        rows.append(
            {
                "max_length_candidate": int(candidate),
                "covered_samples": covered,
                "total_samples": total,
                "coverage_percent": round(covered / total * 100, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def sample_by_label(
    df: pd.DataFrame,
    label_col: str,
    name_col: str,
    n_per_label: int,
    seed: int,
) -> pd.DataFrame:
    """
    Lấy ví dụ văn bản theo từng nhãn.
    """
    samples = []

    for _, group_df in df.groupby([label_col, name_col]):
        n = min(n_per_label, len(group_df))
        samples.append(
            group_df.sample(n=n, random_state=seed)[
                ["id", "split", "text", label_col, name_col]
            ]
        )

    return pd.concat(samples, ignore_index=True)


def plot_label_distribution(
    dist_df: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Vẽ biểu đồ phân bố nhãn theo split.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    pivot = dist_df.pivot(index="label_name", columns="split", values="count").fillna(0)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_length_distribution(
    df: pd.DataFrame,
    length_col: str,
    title: str,
    output_path: str | Path,
    bins: int = 50,
) -> None:
    """
    Vẽ histogram độ dài theo từng split.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 6))

    for split_name, split_df in df.groupby("split"):
        plt.hist(split_df[length_col], bins=bins, alpha=0.5, label=split_name)

    plt.title(title)
    plt.xlabel(length_col)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_eda_report(
    report_path: str | Path,
    basic_df: pd.DataFrame,
    sentiment_dist: pd.DataFrame,
    topic_dist: pd.DataFrame,
    length_summary_df: pd.DataFrame,
    word_percentiles: pd.DataFrame,
    subword_percentiles: pd.DataFrame,
    max_length_df: pd.DataFrame,
) -> None:
    """
    Ghi báo cáo EDA dạng Markdown.

    Không kết luận model ở đây. Chỉ mô tả dữ liệu.
    """
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines: List[str] = []

    lines.append("# EDA Report")
    lines.append("")
    lines.append("## 1. Basic Summary")
    lines.append("")
    lines.append(basic_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 2. Sentiment Label Distribution")
    lines.append("")
    lines.append(sentiment_dist.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Topic Label Distribution")
    lines.append("")
    lines.append(topic_dist.to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Length Summary by Split")
    lines.append("")
    lines.append(length_summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Whitespace Word Count Percentiles")
    lines.append("")
    lines.append(word_percentiles.to_markdown(index=False))
    lines.append("")

    lines.append("## 6. PhoBERT Subword Count Percentiles")
    lines.append("")
    lines.append(subword_percentiles.to_markdown(index=False))
    lines.append("")

    lines.append("## 7. max_length Candidate Coverage")
    lines.append("")
    lines.append(max_length_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 8. Notes")
    lines.append("")
    lines.append("- Giai đoạn này chỉ phân tích dữ liệu, chưa train model.")
    lines.append("- `whitespace_word_count` chỉ là thống kê theo khoảng trắng, không phải tách từ tiếng Việt chuẩn.")
    lines.append("- `phobert_subword_count` dùng tokenizer PhoBERT để ước lượng độ dài đầu vào.")
    lines.append("- Việc chọn `max_length` cho PhoBERT nên dựa trên bảng coverage, không chọn cảm tính.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")