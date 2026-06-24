from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import DatasetDict, load_dataset

from src.utils import ensure_dir


def load_huggingface_dataset(dataset_name: str) -> DatasetDict:
    """
    Load dataset từ HuggingFace.

    Đầu vào:
        dataset_name: tên dataset trên HuggingFace.

    Đầu ra:
        DatasetDict gồm các split như train/dev/test/validation nếu dataset có.

    Lý do cần hàm này:
        Tách phần load dataset khỏi notebook để pipeline dễ tái sử dụng.
    """
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a HuggingFace DatasetDict with train/dev/test splits.")

    return dataset


def datasetdict_to_dataframes(dataset: DatasetDict) -> Dict[str, pd.DataFrame]:
    """
    Chuyển HuggingFace DatasetDict sang dict các pandas DataFrame.

    Đầu vào:
        dataset: DatasetDict.

    Đầu ra:
        dict dạng {split_name: DataFrame}.
    """
    return {split_name: dataset[split_name].to_pandas() for split_name in dataset.keys()}


def save_processed_splits(
    split_dfs: Dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """
    Lưu các split đã xác minh vào data/processed.

    Đầu vào:
        split_dfs: dict {split_name: DataFrame}
        output_dir: thư mục lưu file csv

    Đầu ra:
        Các file CSV theo từng split.
    """
    output_dir = ensure_dir(output_dir)

    for split_name, df in split_dfs.items():
        path = output_dir / f"{split_name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")


def get_column_candidates(columns: List[str]) -> Dict[str, List[str]]:
    """
    Gợi ý các cột có thể là text/sentiment/topic dựa trên tên cột.

    Đây chỉ là gợi ý, không phải kết luận.
    """
    lower_map = {col.lower(): col for col in columns}

    text_keywords = ["sentence", "text", "comment", "feedback", "review", "content"]
    sentiment_keywords = ["sentiment", "polarity", "label_sentiment"]
    topic_keywords = ["topic", "aspect", "category", "label_topic"]

    def match_columns(keywords: List[str]) -> List[str]:
        matched = []
        for lower_col, original_col in lower_map.items():
            if any(keyword in lower_col for keyword in keywords):
                matched.append(original_col)
        return matched

    return {
        "text_candidates": match_columns(text_keywords),
        "sentiment_candidates": match_columns(sentiment_keywords),
        "topic_candidates": match_columns(topic_keywords),
    }


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """
    Tóm tắt nhanh một DataFrame.

    Đầu ra gồm:
        - số dòng
        - số cột
        - danh sách cột
        - missing theo cột
        - số dòng duplicate toàn phần
    """
    return {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicated_rows": int(df.duplicated().sum()),
    }


def value_counts_table(
    split_dfs: Dict[str, pd.DataFrame],
    label_col: str,
) -> pd.DataFrame:
    """
    Tạo bảng phân bố nhãn theo từng split.

    Đầu vào:
        split_dfs: dict {split_name: DataFrame}
        label_col: tên cột nhãn

    Đầu ra:
        DataFrame gồm split, label, count, percent.
    """
    rows = []

    for split_name, df in split_dfs.items():
        total = len(df)
        counts = df[label_col].value_counts(dropna=False).sort_index()

        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "percent": round(float(count / total * 100), 4) if total > 0 else 0.0,
                }
            )

    return pd.DataFrame(rows)


def create_label_mapping(values: List[object]) -> Dict[str, int]:
    """
    Tạo mapping label -> id ổn định.

    Quy tắc:
        - Sort theo dạng string để mapping ổn định.
        - Không giả định ý nghĩa label nếu label là số.

    Đầu ra:
        dict {label_name: label_id}
    """
    unique_values = sorted(set(values), key=lambda x: str(x))
    return {str(label): idx for idx, label in enumerate(unique_values)}


def standardize_uit_vsfc_split(
    df: pd.DataFrame,
    split_name: str,
    text_col: str = "sentence",
    sentiment_col: str = "sentiment",
    topic_col: str = "topic",
    sentiment_id2label: Dict[int, str] | None = None,
    topic_id2label: Dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Chuẩn hóa một split UIT-VSFC về schema thống nhất.

    Đầu vào:
        df: DataFrame gốc từ HuggingFace.
        split_name: train/validation/test.
        text_col: cột văn bản gốc.
        sentiment_col: cột sentiment gốc.
        topic_col: cột topic gốc.
        sentiment_id2label: mapping id -> tên sentiment.
        topic_id2label: mapping id -> tên topic.

    Đầu ra:
        DataFrame chuẩn hóa gồm:
        - id
        - split
        - text
        - sentiment_label
        - sentiment_name
        - topic_label
        - topic_name

    Lý do:
        Các bước sau chỉ dùng schema chuẩn này để tránh nhầm tên cột.
    """
    sentiment_id2label = sentiment_id2label or {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    topic_id2label = topic_id2label or {
        0: "lecturer",
        1: "training_program",
        2: "facility",
        3: "others",
    }

    output = pd.DataFrame()
    output["id"] = [f"{split_name}_{i}" for i in range(len(df))]
    output["split"] = split_name
    output["text"] = df[text_col].astype(str)
    output["sentiment_label"] = df[sentiment_col].astype(int)
    output["sentiment_name"] = output["sentiment_label"].map(sentiment_id2label)
    output["topic_label"] = df[topic_col].astype(int)
    output["topic_name"] = output["topic_label"].map(topic_id2label)

    return output