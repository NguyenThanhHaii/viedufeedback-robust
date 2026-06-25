from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import (
    add_basic_text_lengths,
    add_phobert_subword_lengths,
    basic_summary,
    label_distribution,
    length_summary_by_split,
    load_phobert_tokenizer,
    load_standardized_data,
    percentile_table,
    plot_label_distribution,
    plot_length_distribution,
    recommend_max_length,
    sample_by_label,
    write_eda_report,
)
from src.utils import ensure_dir, load_yaml, project_root, set_seed


def main() -> None:
    root = project_root()
    config = load_yaml(root / "configs" / "eda.yaml")
    set_seed(config["seed"])

    input_file = root / config["data"]["input_file"]

    tables_dir = ensure_dir(root / config["paths"]["tables_dir"])
    figures_dir = ensure_dir(root / config["paths"]["figures_dir"])
    reports_dir = ensure_dir(root / config["paths"]["reports_dir"])

    print(f"Loading standardized data: {input_file}")
    df = load_standardized_data(input_file)

    print("Computing basic text lengths...")
    df = add_basic_text_lengths(df, text_col=config["columns"]["text"])

    print("Loading PhoBERT tokenizer...")
    tokenizer = load_phobert_tokenizer(config["tokenizer"]["model_name"])

    print("Computing PhoBERT subword lengths...")
    df = add_phobert_subword_lengths(
        df=df,
        tokenizer=tokenizer,
        text_col=config["columns"]["text"],
        batch_size=config["tokenizer"]["batch_size"],
        add_special_tokens=config["tokenizer"]["add_special_tokens"],
    )

    # Save enriched length table for debugging and later use
    df[
        [
            "id",
            "split",
            "text",
            "sentiment_label",
            "sentiment_name",
            "topic_label",
            "topic_name",
            "char_count",
            "whitespace_word_count",
            "phobert_subword_count",
        ]
    ].to_csv(
        tables_dir / "eda_text_lengths.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Computing label distributions...")
    sentiment_dist = label_distribution(
        df=df,
        split_col=config["columns"]["split"],
        label_col=config["columns"]["sentiment_label"],
        name_col=config["columns"]["sentiment_name"],
    )
    topic_dist = label_distribution(
        df=df,
        split_col=config["columns"]["split"],
        label_col=config["columns"]["topic_label"],
        name_col=config["columns"]["topic_name"],
    )

    sentiment_dist.to_csv(
        tables_dir / "eda_label_distribution_sentiment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_dist.to_csv(
        tables_dir / "eda_label_distribution_topic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Computing summaries...")
    basic_df = basic_summary(df)
    basic_df.to_csv(
        tables_dir / "eda_basic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    length_summary_df = length_summary_by_split(
        df,
        length_cols=[
            "char_count",
            "whitespace_word_count",
            "phobert_subword_count",
        ],
    )
    length_summary_df.to_csv(
        tables_dir / "text_length_summary_by_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    percentiles = config["analysis"]["percentiles"]

    char_percentiles = percentile_table(df, "char_count", percentiles)
    word_percentiles = percentile_table(df, "whitespace_word_count", percentiles)
    subword_percentiles = percentile_table(df, "phobert_subword_count", percentiles)

    char_percentiles.to_csv(
        tables_dir / "char_count_percentiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    word_percentiles.to_csv(
        tables_dir / "text_length_percentiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subword_percentiles.to_csv(
        tables_dir / "phobert_subword_length_percentiles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    max_length_df = recommend_max_length(
        df=df,
        length_col="phobert_subword_count",
        candidates=config["analysis"]["max_length_candidates"],
    )
    max_length_df.to_csv(
        tables_dir / "max_length_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saving sample texts by label...")
    sentiment_samples = sample_by_label(
        df=df,
        label_col="sentiment_label",
        name_col="sentiment_name",
        n_per_label=config["analysis"]["sample_per_label"],
        seed=config["seed"],
    )
    topic_samples = sample_by_label(
        df=df,
        label_col="topic_label",
        name_col="topic_name",
        n_per_label=config["analysis"]["sample_per_label"],
        seed=config["seed"],
    )

    sentiment_samples.to_csv(
        tables_dir / "sample_sentiment_by_label.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_samples.to_csv(
        tables_dir / "sample_topic_by_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Plotting figures...")
    plot_label_distribution(
        sentiment_dist,
        title="Sentiment Label Distribution",
        output_path=figures_dir / "sentiment_distribution.png",
    )
    plot_label_distribution(
        topic_dist,
        title="Topic Label Distribution",
        output_path=figures_dir / "topic_distribution.png",
    )
    plot_length_distribution(
        df,
        length_col="whitespace_word_count",
        title="Whitespace Word Count Distribution",
        output_path=figures_dir / "word_count_distribution.png",
    )
    plot_length_distribution(
        df,
        length_col="char_count",
        title="Character Count Distribution",
        output_path=figures_dir / "char_count_distribution.png",
    )
    plot_length_distribution(
        df,
        length_col="phobert_subword_count",
        title="PhoBERT Subword Count Distribution",
        output_path=figures_dir / "phobert_subword_count_distribution.png",
    )

    print("Writing EDA report...")
    write_eda_report(
        report_path=reports_dir / "eda_report.md",
        basic_df=basic_df,
        sentiment_dist=sentiment_dist,
        topic_dist=topic_dist,
        length_summary_df=length_summary_df,
        word_percentiles=word_percentiles,
        subword_percentiles=subword_percentiles,
        max_length_df=max_length_df,
    )

    print("\nEDA completed.")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Report: {reports_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()