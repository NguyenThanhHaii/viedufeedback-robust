from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

# Cho phép chạy trực tiếp bằng:
#   python scripts/verify_dataset.py
# mà vẫn import được package src trong project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    datasetdict_to_dataframes,
    get_column_candidates,
    load_huggingface_dataset,
    save_processed_splits,
    standardize_uit_vsfc_split,
    summarize_dataframe,
)
from src.utils import ensure_dir, load_yaml, project_root, save_json


def _get_classlabel_id2label(dataset, split_name: str, label_col: str) -> Dict[int, str]:
    """
    Lấy mapping id -> label name trực tiếp từ HuggingFace ClassLabel feature.

    Lý do:
        Không tự sort nhãn thủ công. Mapping phải bám đúng metadata của dataset.
    """
    feature = dataset[split_name].features[label_col]

    if not hasattr(feature, "names"):
        raise TypeError(
            f"Column `{label_col}` in split `{split_name}` is not a HuggingFace ClassLabel."
        )

    return {idx: label_name for idx, label_name in enumerate(feature.names)}


def _save_label_mapping(
    id2label: Mapping[int, str],
    output_path: Path,
) -> None:
    """
    Lưu mapping theo dạng thống nhất:
        {
          "id2label": {"0": "negative", ...},
          "label2id": {"negative": 0, ...}
        }
    """
    save_json(
        {
            "id2label": {str(k): v for k, v in id2label.items()},
            "label2id": {v: int(k) for k, v in id2label.items()},
        },
        output_path,
    )


def _label_distribution_from_standardized(
    standardized_splits: Dict[str, pd.DataFrame],
    label_col: str,
    name_col: str,
) -> pd.DataFrame:
    """
    Tạo bảng phân bố nhãn từ các split đã chuẩn hóa.

    Đầu ra:
        split, label_id, label_name, count, percent
    """
    rows: List[Dict[str, object]] = []

    for split_name, df in standardized_splits.items():
        total = len(df)
        grouped = (
            df.groupby([label_col, name_col], dropna=False)
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


def _quality_summary(
    standardized_splits: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Kiểm tra chất lượng dữ liệu cơ bản sau chuẩn hóa.

    Các chỉ số:
        - số dòng
        - text bị thiếu/rỗng
        - duplicate text trong từng split
        - missing label/name
    """
    rows: List[Dict[str, object]] = []

    for split_name, df in standardized_splits.items():
        stripped_text = df["text"].fillna("").astype(str).str.strip()

        rows.append(
            {
                "split": split_name,
                "num_rows": int(len(df)),
                "empty_text": int((stripped_text == "").sum()),
                "missing_text": int(df["text"].isna().sum()),
                "duplicated_text": int(df["text"].duplicated().sum()),
                "missing_sentiment_label": int(df["sentiment_label"].isna().sum()),
                "missing_sentiment_name": int(df["sentiment_name"].isna().sum()),
                "missing_topic_label": int(df["topic_label"].isna().sum()),
                "missing_topic_name": int(df["topic_name"].isna().sum()),
            }
        )

    return pd.DataFrame(rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Chuyển DataFrame sang Markdown table mà không cần dependency `tabulate`.

    Lý do:
        pandas.DataFrame.to_markdown() cần package tabulate. Hàm này giúp script
        chạy ổn định hơn trong môi trường local/Kaggle mới tạo.
    """
    if df.empty:
        return "_No rows._"

    columns = [str(col) for col in df.columns]
    rows = []

    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_markdown_report(
    report_path: Path,
    dataset_name: str,
    split_summaries: Dict[str, Dict[str, object]],
    column_candidates: Dict[str, List[str]],
    official_schema: Dict[str, object],
    sentiment_id2label: Dict[int, str],
    topic_id2label: Dict[int, str],
    quality_df: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    """
    Ghi báo cáo Dataset Verification dạng Markdown.

    Báo cáo này dùng làm bằng chứng cho các bước sau:
        - EDA
        - baseline
        - PhoBERT
        - noisy evaluation
    """
    lines: List[str] = []

    lines.append("# Dataset Verification Report")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}`")
    lines.append("")

    lines.append("## 1. Official Schema")
    lines.append("")
    lines.append(f"- Text column: `{official_schema['text_col']}`")
    lines.append(f"- Sentiment column: `{official_schema['sentiment_col']}`")
    lines.append(f"- Topic column: `{official_schema['topic_col']}`")
    lines.append("- Standardized text column: `text`")
    lines.append("- Standardized sentiment columns: `sentiment_label`, `sentiment_name`")
    lines.append("- Standardized topic columns: `topic_label`, `topic_name`")
    lines.append("")

    lines.append("## 2. Data Lineage")
    lines.append("")
    lines.append(f"- Raw snapshot directory: `{raw_dir.as_posix()}`")
    lines.append(f"- Processed standardized directory: `{processed_dir.as_posix()}`")
    lines.append("- Raw files keep the original HuggingFace schema.")
    lines.append("- Processed files use the project-wide standardized schema.")
    lines.append("")

    lines.append("## 3. Split Summary")
    lines.append("")
    for split_name, summary in split_summaries.items():
        lines.append(f"### Split: `{split_name}`")
        lines.append("")
        lines.append(f"- Rows: {summary['num_rows']}")
        lines.append(f"- Columns: {summary['num_columns']}")
        lines.append(f"- Duplicated rows: {summary['duplicated_rows']}")
        lines.append("")
        lines.append("Columns:")
        for col in summary["columns"]:
            lines.append(f"- `{col}`")
        lines.append("")
        lines.append("Missing values:")
        for col, miss_count in summary["missing_by_column"].items():
            lines.append(f"- `{col}`: {miss_count}")
        lines.append("")

    lines.append("## 4. Label Mapping")
    lines.append("")
    lines.append("### Sentiment")
    lines.append("")
    for label_id, label_name in sentiment_id2label.items():
        lines.append(f"- {label_id}: `{label_name}`")
    lines.append("")
    lines.append("### Topic")
    lines.append("")
    for label_id, label_name in topic_id2label.items():
        lines.append(f"- {label_id}: `{label_name}`")
    lines.append("")

    lines.append("## 5. Column Candidates")
    lines.append("")
    lines.append("Các cột dưới đây là kết quả kiểm tra tự động theo tên cột.")
    lines.append("")
    for key, values in column_candidates.items():
        lines.append(f"- {key}: {values}")
    lines.append("")

    lines.append("## 6. Quality Summary")
    lines.append("")
    lines.append(_dataframe_to_markdown(quality_df))
    lines.append("")

    lines.append("## 7. Ghi chú kiểm tra")
    lines.append("")
    lines.append("- Dataset có đủ 3 split: train, validation, test.")
    lines.append("- Dataset có đủ nhãn cho cả sentiment classification và topic classification.")
    lines.append("- Bản raw snapshot được lưu tại `data/raw/`.")
    lines.append("- Các file chuẩn hóa được lưu tại `data/processed/*_standardized.csv`.")
    lines.append("- Chưa train model ở giai đoạn này.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "data.yaml")

    dataset_name = config["dataset"]["hf_name"]

    raw_dir = root / config["paths"].get("raw_dir", "data/raw")
    processed_dir = root / config["paths"]["processed_dir"]
    reports_dir = root / config["paths"]["reports_dir"]
    tables_dir = root / config["paths"]["tables_dir"]
    mappings_dir = root / config["paths"]["mappings_dir"]

    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    ensure_dir(reports_dir)
    ensure_dir(tables_dir)
    ensure_dir(mappings_dir)

    text_col = config["dataset"].get("text_col") or "sentence"
    sentiment_col = config["dataset"].get("sentiment_col") or "sentiment"
    topic_col = config["dataset"].get("topic_col") or "topic"

    print(f"Loading dataset: {dataset_name}")
    dataset = load_huggingface_dataset(dataset_name)

    print("Available splits:")
    for split_name in dataset.keys():
        print(f"- {split_name}: {len(dataset[split_name])} rows")

    print("\nDataset features:")
    for split_name in dataset.keys():
        print(f"\n[{split_name}]")
        print(dataset[split_name].features)

    split_dfs = datasetdict_to_dataframes(dataset)

    first_split_name = list(dataset.keys())[0]
    first_df = split_dfs[first_split_name]

    required_columns = [text_col, sentiment_col, topic_col]
    missing_required_columns = [col for col in required_columns if col not in first_df.columns]
    if missing_required_columns:
        raise ValueError(f"Missing required columns: {missing_required_columns}")

    sentiment_id2label = _get_classlabel_id2label(dataset, first_split_name, sentiment_col)
    topic_id2label = _get_classlabel_id2label(dataset, first_split_name, topic_col)

    print("\nOfficial label mappings:")
    print("Sentiment:", sentiment_id2label)
    print("Topic:", topic_id2label)

    print("\nSaving raw split CSV files...")
    save_processed_splits(split_dfs, raw_dir)

    save_json(
        {
            "dataset_name": dataset_name,
            "source_type": config["dataset"].get("source_type", "huggingface"),
            "hf_name": dataset_name,
            "splits": {split_name: int(len(dataset[split_name])) for split_name in dataset.keys()},
            "text_column": text_col,
            "sentiment_column": sentiment_col,
            "topic_column": topic_col,
            "sentiment_labels": {str(k): v for k, v in sentiment_id2label.items()},
            "topic_labels": {str(k): v for k, v in topic_id2label.items()},
            "raw_files": {
                split_name: f"{split_name}.csv" for split_name in dataset.keys()
            },
        },
        raw_dir / "dataset_source_metadata.json",
    )

    print("\nSaving standardized processed split CSV files...")
    standardized_splits: Dict[str, pd.DataFrame] = {}

    for split_name, df in split_dfs.items():
        standardized_df = standardize_uit_vsfc_split(
            df=df,
            split_name=split_name,
            text_col=text_col,
            sentiment_col=sentiment_col,
            topic_col=topic_col,
            sentiment_id2label=sentiment_id2label,
            topic_id2label=topic_id2label,
        )

        standardized_splits[split_name] = standardized_df

        output_path = processed_dir / f"{split_name}_standardized.csv"
        standardized_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    all_standardized = pd.concat(standardized_splits.values(), ignore_index=True)
    all_standardized.to_csv(
        processed_dir / "all_standardized.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_label_mapping(
        sentiment_id2label,
        mappings_dir / "sentiment_label_mapping.json",
    )
    _save_label_mapping(
        topic_id2label,
        mappings_dir / "topic_label_mapping.json",
    )

    split_summaries = {
        split_name: summarize_dataframe(df)
        for split_name, df in split_dfs.items()
    }

    column_candidates = get_column_candidates(list(first_df.columns))

    split_summary_df = pd.DataFrame(
        [
            {
                "split": split_name,
                "num_rows": summary["num_rows"],
                "num_columns": summary["num_columns"],
                "duplicated_rows": summary["duplicated_rows"],
            }
            for split_name, summary in split_summaries.items()
        ]
    )
    split_summary_df.to_csv(
        tables_dir / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sentiment_dist = _label_distribution_from_standardized(
        standardized_splits,
        label_col="sentiment_label",
        name_col="sentiment_name",
    )
    sentiment_dist.to_csv(
        tables_dir / "label_distribution_sentiment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_dist = _label_distribution_from_standardized(
        standardized_splits,
        label_col="topic_label",
        name_col="topic_name",
    )
    topic_dist.to_csv(
        tables_dir / "label_distribution_topic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quality_df = _quality_summary(standardized_splits)
    quality_df.to_csv(
        tables_dir / "dataset_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    official_schema = {
        "text_col": text_col,
        "sentiment_col": sentiment_col,
        "topic_col": topic_col,
        "raw_columns": list(first_df.columns),
        "standardized_columns": [
            "id",
            "split",
            "text",
            "sentiment_label",
            "sentiment_name",
            "topic_label",
            "topic_name",
        ],
    }

    schema = {
        "dataset_name": dataset_name,
        "source_type": config["dataset"].get("source_type", "huggingface"),
        "splits": list(dataset.keys()),
        "official_schema": official_schema,
        "sentiment_id2label": {str(k): v for k, v in sentiment_id2label.items()},
        "topic_id2label": {str(k): v for k, v in topic_id2label.items()},
        "split_summaries": split_summaries,
        "column_candidates": column_candidates,
        "raw_dir": str(raw_dir.relative_to(root)),
        "processed_dir": str(processed_dir.relative_to(root)),
        "features": {
            split_name: str(dataset[split_name].features)
            for split_name in dataset.keys()
        },
    }
    save_json(schema, reports_dir / "dataset_schema.json")

    write_markdown_report(
        report_path=reports_dir / "dataset_verification.md",
        dataset_name=dataset_name,
        split_summaries=split_summaries,
        column_candidates=column_candidates,
        official_schema=official_schema,
        sentiment_id2label=sentiment_id2label,
        topic_id2label=topic_id2label,
        quality_df=quality_df,
        raw_dir=raw_dir.relative_to(root),
        processed_dir=processed_dir.relative_to(root),
    )

    print("\nColumn candidates:")
    print(json.dumps(column_candidates, ensure_ascii=False, indent=2))

    print("\nDataset verification completed.")
    print(f"Raw data dir: {raw_dir}")
    print(f"Processed data dir: {processed_dir}")
    print(f"Raw metadata: {raw_dir / 'dataset_source_metadata.json'}")
    print(f"Report: {reports_dir / 'dataset_verification.md'}")
    print(f"Schema: {reports_dir / 'dataset_schema.json'}")
    print(f"Split summary: {tables_dir / 'split_summary.csv'}")
    print(f"Quality summary: {tables_dir / 'dataset_quality_summary.csv'}")
    print(f"Sentiment distribution: {tables_dir / 'label_distribution_sentiment.csv'}")
    print(f"Topic distribution: {tables_dir / 'label_distribution_topic.csv'}")


if __name__ == "__main__":
    main()
