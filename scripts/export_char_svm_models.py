from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def project_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "scripts":
        return cwd.parent
    return cwd


def find_label_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find label column. Candidates: {candidates}. Available: {list(df.columns)}")


def train_char_svm(df: pd.DataFrame, text_col: str, label_col: str) -> Pipeline:
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipe.fit(df[text_col].astype(str), df[label_col].astype(int))
    return pipe


def main() -> None:
    root = project_root()

    train_file = root / "data" / "processed" / "train_standardized.csv"
    output_dir = root / "outputs" / "models" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    df = pd.read_csv(train_file)

    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in train_standardized.csv")

    sentiment_col = find_label_col(
        df,
        ["sentiment_label", "sentiment", "sentiment_id", "label_sentiment"],
    )
    topic_col = find_label_col(
        df,
        ["topic_label", "topic", "topic_id", "label_topic"],
    )

    tasks = {
        "sentiment": sentiment_col,
        "topic": topic_col,
    }

    metadata = {}

    for task, label_col in tasks.items():
        print(f"Training TF-IDF char SVM for task={task}, label_col={label_col}")
        model = train_char_svm(df=df, text_col="text", label_col=label_col)

        output_path = output_dir / f"{task}_tfidf_char_svm.joblib"
        joblib.dump(model, output_path)

        metadata[task] = {
            "model_path": str(output_path),
            "text_col": "text",
            "label_col": label_col,
            "num_train_rows": int(len(df)),
            "vectorizer": "TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, sublinear_tf=True)",
            "classifier": "LinearSVC(class_weight='balanced', random_state=42)",
        }

        print(f"Saved: {output_path}")

    metadata_path = output_dir / "char_svm_export_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
