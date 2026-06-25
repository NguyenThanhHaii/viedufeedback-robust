from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


TEENCODE_MAP: Dict[str, List[str]] = {
    "không": ["ko", "k", "khong"],
    "khong": ["ko", "k"],
    "được": ["đc", "dc", "duoc"],
    "duoc": ["dc"],
    "với": ["vs"],
    "voi": ["vs"],
    "rồi": ["r", "roi"],
    "rất": ["rat", "rấtt"],
    "quá": ["qua", "qá"],
    "giảng viên": ["gv"],
    "giang vien": ["gv"],
    "sinh viên": ["sv"],
    "sinh vien": ["sv"],
    "chương trình": ["ct"],
    "chuong trinh": ["ct"],
    "cơ sở vật chất": ["csvc"],
    "co so vat chat": ["csvc"],
    "cơ sở": ["cs"],
    "co so": ["cs"],
    "vật chất": ["vc"],
    "vat chat": ["vc"],
    "bài tập": ["bt"],
    "bai tap": ["bt"],
    "bài giảng": ["bg"],
    "bai giang": ["bg"],
    "thực hành": ["th"],
    "thuc hanh": ["th"],
    "học phí": ["hp"],
    "hoc phi": ["hp"],
    "đại học": ["dh"],
    "dai hoc": ["dh"],
}


def stable_random(seed: int, *parts: object) -> random.Random:
    """
    Tạo random generator ổn định theo seed và định danh mẫu.

    Không dùng hash() mặc định của Python vì hash có thể thay đổi giữa các lần chạy.
    """
    key = "::".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    stable_seed = int(digest[:8], 16)
    return random.Random(stable_seed)


def remove_vietnamese_diacritics(text: str) -> str:
    """
    Bỏ dấu tiếng Việt.

    Ví dụ:
        "giảng viên rất tốt" -> "giang vien rat tot"

    Đây là loại nhiễu phổ biến trong tiếng Việt phi chuẩn.
    """
    text = str(text)
    text = text.replace("đ", "d").replace("Đ", "D")

    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )

    return unicodedata.normalize("NFC", without_marks)


def _is_word_token(token: str) -> bool:
    return any(char.isalpha() for char in token)


def _typo_token(token: str, rng: random.Random) -> str:
    """
    Sinh lỗi typo nhẹ trên một token.

    Các thao tác:
        - xóa một ký tự
        - đảo hai ký tự cạnh nhau
        - lặp một ký tự

    Không áp dụng cho token quá ngắn.
    """
    if len(token) < 4 or not _is_word_token(token):
        return token

    chars = list(token)
    operation = rng.choice(["delete", "swap", "duplicate"])

    if operation == "delete" and len(chars) >= 4:
        idx = rng.randrange(1, len(chars) - 1)
        del chars[idx]

    elif operation == "swap" and len(chars) >= 4:
        idx = rng.randrange(1, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    elif operation == "duplicate":
        idx = rng.randrange(0, len(chars))
        chars.insert(idx, chars[idx])

    return "".join(chars)


def apply_typo_noise(
    text: str,
    token_prob: float,
    rng: random.Random,
) -> str:
    """
    Áp dụng typo noise theo xác suất trên từng token.
    """
    text = str(text)

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if rng.random() < token_prob:
            return _typo_token(token, rng)
        return token

    return re.sub(r"\w+", replace, text, flags=re.UNICODE)


def apply_teencode_noise(
    text: str,
    replace_prob: float,
    rng: random.Random,
    mapping: Mapping[str, List[str]] | None = None,
) -> str:
    """
    Thay một số cụm/từ tiếng Việt bằng dạng viết tắt/teencode.

    Ví dụ:
        không -> ko/k
        được -> đc/dc
        giảng viên -> gv
        sinh viên -> sv
    """
    output = str(text)
    mapping = mapping or TEENCODE_MAP

    # Cụm dài xử lý trước để tránh thay từng từ con trước.
    sorted_items = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)

    for source, candidates in sorted_items:
        pattern = re.compile(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )

        def replace(match: re.Match) -> str:
            if rng.random() < replace_prob:
                return rng.choice(candidates)
            return match.group(0)

        output = pattern.sub(replace, output)

    return output


def apply_noise_variant(
    text: str,
    variant: Mapping[str, object],
    seed: int,
    sample_id: object,
) -> str:
    """
    Áp dụng một biến thể noise lên một văn bản.
    """
    variant_name = str(variant["name"])
    noise_type = str(variant["type"])
    rng = stable_random(seed, variant_name, sample_id)

    output = str(text)

    if noise_type == "remove_diacritics":
        return remove_vietnamese_diacritics(output)

    if noise_type == "typo":
        token_prob = float(variant.get("token_prob", 0.05))
        return apply_typo_noise(output, token_prob=token_prob, rng=rng)

    if noise_type == "teencode":
        replace_prob = float(variant.get("replace_prob", 0.25))
        return apply_teencode_noise(output, replace_prob=replace_prob, rng=rng)

    if noise_type == "mixed":
        teencode_prob = float(variant.get("teencode_prob", 0.20))
        typo_token_prob = float(variant.get("typo_token_prob", 0.04))
        use_remove_diacritics = bool(variant.get("remove_diacritics", True))

        output = apply_teencode_noise(output, replace_prob=teencode_prob, rng=rng)

        if use_remove_diacritics:
            output = remove_vietnamese_diacritics(output)

        output = apply_typo_noise(output, token_prob=typo_token_prob, rng=rng)

        return output

    raise ValueError(f"Unknown noise type: {noise_type}")


def char_change_ratio(original: str, noisy: str) -> float:
    """
    Tính mức thay đổi ký tự tương đối bằng SequenceMatcher.

    0.0 nghĩa là không đổi.
    Giá trị càng cao nghĩa là text thay đổi càng nhiều.
    """
    original = str(original)
    noisy = str(noisy)

    if original == noisy:
        return 0.0

    similarity = SequenceMatcher(None, original, noisy).ratio()
    return round(1.0 - similarity, 6)


def build_clean_eval_dataframe(
    test_df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Tạo bản clean có cùng schema với noisy để tiện evaluate sau này.
    """
    output = test_df.copy()

    output["original_id"] = output["id"]
    output["original_text"] = output[text_col]
    output["variant"] = "clean"
    output["noise_type"] = "clean"
    output["noise_level"] = "none"
    output["changed"] = False
    output["char_change_ratio"] = 0.0

    return output


def build_noisy_dataframe(
    test_df: pd.DataFrame,
    variant: Mapping[str, object],
    seed: int,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Tạo noisy test set cho một biến thể noise.
    """
    rows = []

    variant_name = str(variant["name"])
    noise_type = str(variant["type"])
    noise_level = str(variant.get("level", "unknown"))

    for _, row in test_df.iterrows():
        original_text = str(row[text_col])
        original_id = row["id"]

        noisy_text = apply_noise_variant(
            text=original_text,
            variant=variant,
            seed=seed,
            sample_id=original_id,
        )

        output_row = row.to_dict()
        output_row["id"] = f"{original_id}__{variant_name}"
        output_row["original_id"] = original_id
        output_row["original_text"] = original_text
        output_row[text_col] = noisy_text
        output_row["variant"] = variant_name
        output_row["noise_type"] = noise_type
        output_row["noise_level"] = noise_level
        output_row["changed"] = original_text != noisy_text
        output_row["char_change_ratio"] = char_change_ratio(original_text, noisy_text)

        rows.append(output_row)

    return pd.DataFrame(rows)


def summarize_noisy_data(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tóm tắt mức độ thay đổi theo từng biến thể noise.
    """
    rows = []

    for variant, group_df in eval_df.groupby("variant"):
        num_rows = len(group_df)
        changed_count = int(group_df["changed"].sum())

        rows.append(
            {
                "variant": variant,
                "noise_type": group_df["noise_type"].iloc[0],
                "noise_level": group_df["noise_level"].iloc[0],
                "num_rows": int(num_rows),
                "changed_count": changed_count,
                "unchanged_count": int(num_rows - changed_count),
                "changed_percent": round(changed_count / num_rows * 100, 4) if num_rows else 0.0,
                "avg_char_change_ratio": round(float(group_df["char_change_ratio"].mean()), 6),
                "p50_char_change_ratio": round(float(group_df["char_change_ratio"].quantile(0.50)), 6),
                "p95_char_change_ratio": round(float(group_df["char_change_ratio"].quantile(0.95)), 6),
                "max_char_change_ratio": round(float(group_df["char_change_ratio"].max()), 6),
            }
        )

    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def sample_noisy_examples(
    noisy_df: pd.DataFrame,
    examples_per_variant: int,
    seed: int,
) -> pd.DataFrame:
    """
    Lấy ví dụ clean/noisy theo từng biến thể noise.
    """
    samples = []

    for variant, group_df in noisy_df.groupby("variant"):
        changed_df = group_df[group_df["changed"]].copy()

        if len(changed_df) == 0:
            candidate_df = group_df.copy()
        else:
            candidate_df = changed_df

        n = min(examples_per_variant, len(candidate_df))

        sample_df = candidate_df.sample(n=n, random_state=seed)[
            [
                "variant",
                "noise_type",
                "noise_level",
                "original_id",
                "original_text",
                "text",
                "char_change_ratio",
                "sentiment_label",
                "sentiment_name",
                "topic_label",
                "topic_name",
            ]
        ]

        samples.append(sample_df)

    return pd.concat(samples, ignore_index=True)


def plot_noisy_summary(
    summary_df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Vẽ biểu đồ summary theo variant.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = summary_df[summary_df["variant"] != "clean"].copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["variant"], plot_df[value_col])
    plt.title(title)
    plt.xlabel("Noise variant")
    plt.ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_noisy_generation_report(
    report_path: str | Path,
    summary_df: pd.DataFrame,
    examples_df: pd.DataFrame,
) -> None:
    """
    Ghi report Markdown cho giai đoạn noisy generation.

    Dùng to_string để tránh phụ thuộc tabulate.
    """
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines: List[str] = []

    lines.append("# Noisy Test Generation Report")
    lines.append("")
    lines.append("## 1. Mục tiêu")
    lines.append("")
    lines.append(
        "Giai đoạn này tạo các phiên bản test set bị làm nhiễu để phục vụ "
        "đánh giá độ bền của mô hình trên tiếng Việt phi chuẩn."
    )
    lines.append("")
    lines.append("Chỉ test set được làm nhiễu. Train và validation giữ nguyên.")
    lines.append("")

    lines.append("## 2. Noise variants")
    lines.append("")
    lines.append("- `no_accent`: bỏ dấu tiếng Việt.")
    lines.append("- `typo_light`: lỗi gõ nhẹ.")
    lines.append("- `typo_medium`: lỗi gõ mức trung bình.")
    lines.append("- `teencode_light`: thay một số từ/cụm bằng viết tắt phổ biến.")
    lines.append("- `mixed_light`: kết hợp teencode, bỏ dấu và typo nhẹ.")
    lines.append("")

    lines.append("## 3. Generation summary")
    lines.append("")
    lines.append("```text")
    lines.append(summary_df.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 4. Example noisy texts")
    lines.append("")
    display_cols = [
        "variant",
        "original_text",
        "text",
        "char_change_ratio",
        "sentiment_name",
        "topic_name",
    ]
    example_view = examples_df[display_cols].head(30)
    lines.append("```text")
    lines.append(example_view.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 5. Ghi chú")
    lines.append("")
    lines.append("- Các nhãn sentiment/topic được giữ nguyên từ clean test set.")
    lines.append("- Noise generation dùng seed cố định để có thể tái lập.")
    lines.append("- `test_eval_all.csv` chứa cả clean và noisy variants để phục vụ evaluation sau này.")
    lines.append("- Stage này chưa đánh giá model.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")