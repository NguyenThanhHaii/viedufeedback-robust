from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


VIETNAMESE_MARKED_CHARS = set(
    "ăâđêôơư"
    "áàảãạấầẩẫậắằẳẵặ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶ"
    "ÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰ"
    "ÝỲỶỸỴ"
)

TEENCODE_MAP = {
    "gv": "giảng viên",
    "sv": "sinh viên",
    "ko": "không",
    "khong": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "hok": "không",
    "hem": "không",
    "oke": "ổn",
    "ok": "ổn",
    "mn": "mọi người",
    "bt": "bình thường",
    "nx": "nữa",
    "vs": "với",
    "wa": "quá",
    "qua": "quá",
}


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def project_root() -> Path:
    """
    Resolve project root from scripts/, notebooks/ or project root.
    """
    cwd = Path.cwd()
    if cwd.name in {"scripts", "notebooks"}:
        return cwd.parent
    return cwd


def remove_diacritics(text: str) -> str:
    """
    Remove Vietnamese tone marks and convert đ/Đ to d/D.
    """
    text = str(text)
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def reduce_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Reduce long repeated characters, e.g. 'tốtttt' -> 'tốtt'.
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeat, str(text))


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)


def apply_teencode(text: str, rng: random.Random, replace_prob: float) -> str:
    tokens = tokenize_words(text)
    outputs = []

    for token in tokens:
        key = token.lower()
        if key in TEENCODE_MAP and rng.random() < replace_prob:
            replacement = TEENCODE_MAP[key]
            outputs.append(replacement)
        else:
            outputs.append(token)

    return join_tokens(outputs)


def join_tokens(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+([)])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_typo(text: str, rng: random.Random, token_prob: float) -> str:
    tokens = tokenize_words(text)
    outputs = []

    for token in tokens:
        if not any(ch.isalpha() for ch in token) or len(token) < 4:
            outputs.append(token)
            continue

        if rng.random() >= token_prob:
            outputs.append(token)
            continue

        chars = list(token)
        op = rng.choice(["delete", "swap", "repeat"])

        if op == "delete" and len(chars) >= 4:
            idx = rng.randrange(1, len(chars))
            del chars[idx]
        elif op == "swap" and len(chars) >= 4:
            idx = rng.randrange(0, len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "repeat":
            idx = rng.randrange(0, len(chars))
            chars.insert(idx, chars[idx])

        outputs.append("".join(chars))

    return join_tokens(outputs)


def generate_noisy_text(text: str, variant_cfg: Dict[str, Any], rng: random.Random) -> str:
    variant_type = variant_cfg["type"]

    if variant_type == "clean":
        return str(text)

    if variant_type == "remove_diacritics":
        return remove_diacritics(text)

    if variant_type == "typo":
        return apply_typo(
            text=text,
            rng=rng,
            token_prob=float(variant_cfg.get("token_prob", 0.05)),
        )

    if variant_type == "teencode":
        return apply_teencode(
            text=text,
            rng=rng,
            replace_prob=float(variant_cfg.get("replace_prob", 0.25)),
        )

    if variant_type == "mixed":
        output = str(text)

        if bool(variant_cfg.get("remove_diacritics", False)):
            output = remove_diacritics(output)

        output = apply_teencode(
            text=output,
            rng=rng,
            replace_prob=float(variant_cfg.get("teencode_prob", 0.20)),
        )
        output = apply_typo(
            text=output,
            rng=rng,
            token_prob=float(variant_cfg.get("typo_token_prob", 0.04)),
        )
        output = reduce_repeated_chars(output)
        return output

    raise ValueError(f"Unsupported noise type: {variant_type}")


def build_noisy_validation(
    validation_df: pd.DataFrame,
    text_col: str,
    variants: List[Dict[str, Any]],
    seed: int,
    original_id_prefix: str = "validation",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create controlled noisy variants from validation split.

    This is used only to select the no-accent detector threshold.
    """
    rows = []
    summary_rows = []

    base_df = validation_df.copy().reset_index(drop=True)

    if "original_id" not in base_df.columns:
        base_df["original_id"] = [
            f"{original_id_prefix}_{idx}" for idx in range(len(base_df))
        ]

    if "original_text" not in base_df.columns:
        base_df["original_text"] = base_df[text_col].astype(str)

    for variant_idx, variant_cfg in enumerate(variants):
        variant_name = variant_cfg["name"]
        rng = random.Random(seed + variant_idx * 1009)

        variant_df = base_df.copy()
        variant_df["variant"] = variant_name
        variant_df["noise_type"] = variant_cfg["type"]
        variant_df["noise_level"] = variant_cfg.get("level", "none")
        variant_df[text_col] = [
            generate_noisy_text(text, variant_cfg, rng)
            for text in variant_df["original_text"].astype(str)
        ]

        changed_ratio = (
            variant_df[text_col].astype(str) != variant_df["original_text"].astype(str)
        ).mean()

        summary_rows.append(
            {
                "variant": variant_name,
                "noise_type": variant_cfg["type"],
                "noise_level": variant_cfg.get("level", "none"),
                "num_rows": len(variant_df),
                "changed_percent": round(float(changed_ratio * 100), 4),
            }
        )

        rows.append(variant_df)

    output_df = pd.concat(rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return output_df, summary_df


def count_alpha_chars(text: str) -> int:
    return sum(1 for ch in str(text) if ch.isalpha())


def count_marked_vietnamese_chars(text: str) -> int:
    return sum(1 for ch in str(text) if ch in VIETNAMESE_MARKED_CHARS)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def accented_ratio(text: str) -> float:
    alpha_count = count_alpha_chars(text)
    if alpha_count == 0:
        return 0.0
    return count_marked_vietnamese_chars(text) / alpha_count


def detect_no_accent(
    text: str,
    threshold: float,
    min_alpha_chars: int,
    min_words: int,
) -> bool:
    alpha_count = count_alpha_chars(text)
    words = word_count(text)

    if alpha_count < min_alpha_chars:
        return False

    if words < min_words:
        return False

    return accented_ratio(text) <= float(threshold)


def add_detection_features(
    df: pd.DataFrame,
    text_col: str,
    threshold: float,
    min_alpha_chars: int,
    min_words: int,
) -> pd.DataFrame:
    output = df.copy()
    output["alpha_chars"] = output[text_col].astype(str).apply(count_alpha_chars)
    output["word_count"] = output[text_col].astype(str).apply(word_count)
    output["marked_vietnamese_chars"] = output[text_col].astype(str).apply(
        count_marked_vietnamese_chars
    )
    output["accented_ratio"] = output[text_col].astype(str).apply(accented_ratio)
    output["predicted_no_accent"] = output[text_col].astype(str).apply(
        lambda text: detect_no_accent(
            text=text,
            threshold=threshold,
            min_alpha_chars=min_alpha_chars,
            min_words=min_words,
        )
    )
    return output


def detector_metrics(y_true: Iterable[bool], y_pred: Iterable[bool]) -> Dict[str, Any]:
    true_list = list(bool(x) for x in y_true)
    pred_list = list(bool(x) for x in y_pred)

    tp = sum(t and p for t, p in zip(true_list, pred_list))
    fp = sum((not t) and p for t, p in zip(true_list, pred_list))
    tn = sum((not t) and (not p) for t, p in zip(true_list, pred_list))
    fn = sum(t and (not p) for t, p in zip(true_list, pred_list))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(true_list) if true_list else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "false_positive_rate": round(false_positive_rate, 6),
        "false_negative_rate": round(false_negative_rate, 6),
    }


def tune_no_accent_threshold(
    validation_noisy_df: pd.DataFrame,
    text_col: str,
    no_accent_variants: List[str],
    threshold_candidates: List[float],
    min_alpha_chars: int,
    min_words: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []

    working_df = validation_noisy_df.copy()
    working_df["true_no_accent"] = working_df["variant"].isin(no_accent_variants)

    for threshold in threshold_candidates:
        detected = add_detection_features(
            df=working_df,
            text_col=text_col,
            threshold=float(threshold),
            min_alpha_chars=min_alpha_chars,
            min_words=min_words,
        )

        metrics = detector_metrics(
            y_true=detected["true_no_accent"],
            y_pred=detected["predicted_no_accent"],
        )

        route_rate = detected["predicted_no_accent"].mean()

        rows.append(
            {
                "threshold": float(threshold),
                "route_to_char_rate": round(float(route_rate), 6),
                **metrics,
            }
        )

    result_df = pd.DataFrame(rows)

    selected_row = (
        result_df.sort_values(
            ["f1", "false_positive_rate", "threshold"],
            ascending=[False, True, True],
        )
        .iloc[0]
        .to_dict()
    )

    selected = {
        "threshold": float(selected_row["threshold"]),
        "selection_metric": "detector_f1_then_low_false_positive_rate",
        "selected_row": selected_row,
        "min_alpha_chars": int(min_alpha_chars),
        "min_words": int(min_words),
        "no_accent_variants": no_accent_variants,
    }

    return result_df, selected


def prepare_prediction_frame(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    output = df.copy()

    required = ["task", "variant", "original_id", "text", "y_true", "y_pred"]
    missing = [col for col in required if col not in output.columns]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    if model_name is not None:
        if "model" not in output.columns:
            raise ValueError("Baseline prediction file must contain a 'model' column.")
        output = output[output["model"] == model_name].copy()

    if output.empty:
        raise ValueError(f"No prediction rows found for model={model_name}")

    output["original_id"] = output["original_id"].astype(str)

    if "y_true_name" not in output.columns:
        output["y_true_name"] = output["y_true"].astype(str)

    if "y_pred_name" not in output.columns:
        output["y_pred_name"] = output["y_pred"].astype(str)

    if "noise_type" not in output.columns:
        output["noise_type"] = output["variant"]

    if "noise_level" not in output.columns:
        output["noise_level"] = "unknown"

    return output


def build_router_predictions(
    phobert_predictions: pd.DataFrame,
    char_svm_predictions: pd.DataFrame,
    threshold: float,
    min_alpha_chars: int,
    min_words: int,
    no_accent_variants: List[str],
) -> pd.DataFrame:
    """
    Build predictions for:
    - phobert_only
    - tfidf_char_svm_only
    - robust_router
    - oracle_router

    The function does not rerun models. It reuses saved predictions from Stage 5 and Stage 6.
    """
    key_cols = ["task", "variant", "original_id"]

    phobert = prepare_prediction_frame(phobert_predictions).copy()
    char_svm = prepare_prediction_frame(char_svm_predictions, model_name="tfidf_char_svm").copy()

    phobert = phobert.add_prefix("phobert_")
    char_svm = char_svm.add_prefix("char_")

    merged = phobert.merge(
        char_svm,
        left_on=[f"phobert_{col}" for col in key_cols],
        right_on=[f"char_{col}" for col in key_cols],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No rows after merging PhoBERT and char-SVM predictions. "
            "Check task, variant and original_id columns."
        )

    rows: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        task = row["phobert_task"]
        variant = row["phobert_variant"]
        original_id = row["phobert_original_id"]
        text = row["phobert_text"]

        true_label = row["phobert_y_true"]
        true_name = row["phobert_y_true_name"]
        noise_type = row.get("phobert_noise_type", variant)
        noise_level = row.get("phobert_noise_level", "unknown")

        route_to_char = detect_no_accent(
            text=text,
            threshold=threshold,
            min_alpha_chars=min_alpha_chars,
            min_words=min_words,
        )
        oracle_to_char = variant in no_accent_variants

        base_payload = {
            "task": task,
            "variant": variant,
            "noise_type": noise_type,
            "noise_level": noise_level,
            "original_id": original_id,
            "original_text": row.get("phobert_original_text", text),
            "text": text,
            "y_true": true_label,
            "y_true_name": true_name,
            "alpha_chars": count_alpha_chars(text),
            "word_count": word_count(text),
            "marked_vietnamese_chars": count_marked_vietnamese_chars(text),
            "accented_ratio": accented_ratio(text),
            "detected_no_accent": bool(route_to_char),
            "oracle_no_accent": bool(oracle_to_char),
            "sentiment_label": row.get("phobert_sentiment_label"),
            "sentiment_name": row.get("phobert_sentiment_name"),
            "topic_label": row.get("phobert_topic_label"),
            "topic_name": row.get("phobert_topic_name"),
        }

        system_specs = [
            (
                "phobert_only",
                "phobert",
                row["phobert_y_pred"],
                row["phobert_y_pred_name"],
            ),
            (
                "tfidf_char_svm_only",
                "tfidf_char_svm",
                row["char_y_pred"],
                row["char_y_pred_name"],
            ),
            (
                "robust_router",
                "tfidf_char_svm" if route_to_char else "phobert",
                row["char_y_pred"] if route_to_char else row["phobert_y_pred"],
                row["char_y_pred_name"] if route_to_char else row["phobert_y_pred_name"],
            ),
            (
                "oracle_router",
                "tfidf_char_svm" if oracle_to_char else "phobert",
                row["char_y_pred"] if oracle_to_char else row["phobert_y_pred"],
                row["char_y_pred_name"] if oracle_to_char else row["phobert_y_pred_name"],
            ),
        ]

        for system, routed_model, pred, pred_name in system_specs:
            output = dict(base_payload)
            output.update(
                {
                    "system": system,
                    "routed_model": routed_model,
                    "y_pred": pred,
                    "y_pred_name": pred_name,
                    "is_correct": pred_name == true_name,
                }
            )
            rows.append(output)

    return pd.DataFrame(rows)


def compute_classification_results(
    prediction_df: pd.DataFrame,
    variant_order: List[str],
) -> pd.DataFrame:
    rows = []

    for (task, system, variant), group_df in prediction_df.groupby(["task", "system", "variant"]):
        y_true = group_df["y_true_name"].astype(str)
        y_pred = group_df["y_pred_name"].astype(str)

        precision_macro, recall_macro, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        rows.append(
            {
                "task": task,
                "system": system,
                "variant": variant,
                "noise_type": group_df["noise_type"].iloc[0],
                "noise_level": group_df["noise_level"].iloc[0],
                "num_samples": len(group_df),
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
                "macro_precision": round(float(precision_macro), 6),
                "macro_recall": round(float(recall_macro), 6),
                "macro_f1": round(float(macro_f1), 6),
                "weighted_precision": round(float(precision_weighted), 6),
                "weighted_recall": round(float(recall_weighted), 6),
                "weighted_f1": round(float(weighted_f1), 6),
            }
        )

    output = pd.DataFrame(rows)
    output = add_variant_order(output, variant_order)
    output = output.sort_values(["task", "system", "variant_order"]).reset_index(drop=True)
    return output


def compute_drop_from_clean(results_df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    rows = []

    for (task, system), group_df in results_df.groupby(["task", "system"]):
        clean_rows = group_df[group_df["variant"] == "clean"]
        if len(clean_rows) != 1:
            raise ValueError(f"Expected exactly one clean row for task={task}, system={system}.")

        clean = clean_rows.iloc[0]

        for _, row in group_df.iterrows():
            macro_drop = float(clean["macro_f1"] - row["macro_f1"])
            acc_drop = float(clean["accuracy"] - row["accuracy"])
            weighted_drop = float(clean["weighted_f1"] - row["weighted_f1"])

            rows.append(
                {
                    "task": task,
                    "system": system,
                    "variant": row["variant"],
                    "clean_accuracy": clean["accuracy"],
                    "variant_accuracy": row["accuracy"],
                    "accuracy_drop": round(acc_drop, 6),
                    "accuracy_relative_drop_percent": safe_relative_drop(clean["accuracy"], acc_drop),
                    "clean_macro_f1": clean["macro_f1"],
                    "variant_macro_f1": row["macro_f1"],
                    "macro_f1_drop": round(macro_drop, 6),
                    "macro_f1_relative_drop_percent": safe_relative_drop(clean["macro_f1"], macro_drop),
                    "clean_weighted_f1": clean["weighted_f1"],
                    "variant_weighted_f1": row["weighted_f1"],
                    "weighted_f1_drop": round(weighted_drop, 6),
                    "weighted_f1_relative_drop_percent": safe_relative_drop(clean["weighted_f1"], weighted_drop),
                }
            )

    output = pd.DataFrame(rows)
    output = add_variant_order(output, variant_order)
    output = output.sort_values(["task", "system", "variant_order"]).reset_index(drop=True)
    return output


def safe_relative_drop(clean_value: float, drop_value: float) -> float:
    if float(clean_value) == 0:
        return 0.0
    return round(float(drop_value / clean_value * 100), 4)


def add_variant_order(df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    output = df.copy()
    order_map = {variant: idx for idx, variant in enumerate(variant_order)}
    output["variant_order"] = output["variant"].map(order_map).fillna(999).astype(int)
    return output


def compute_classification_report_table(
    prediction_df: pd.DataFrame,
    variant_order: List[str],
) -> pd.DataFrame:
    rows = []

    for (task, system, variant), group_df in prediction_df.groupby(["task", "system", "variant"]):
        y_true = group_df["y_true_name"].astype(str)
        y_pred = group_df["y_pred_name"].astype(str)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "task": task,
                        "system": system,
                        "variant": variant,
                        "label": label,
                        "precision": round(float(metrics.get("precision", 0.0)), 6),
                        "recall": round(float(metrics.get("recall", 0.0)), 6),
                        "f1_score": round(float(metrics.get("f1-score", 0.0)), 6),
                        "support": int(metrics.get("support", 0)),
                    }
                )
            else:
                rows.append(
                    {
                        "task": task,
                        "system": system,
                        "variant": variant,
                        "label": label,
                        "precision": None,
                        "recall": None,
                        "f1_score": round(float(metrics), 6),
                        "support": len(group_df),
                    }
                )

    output = pd.DataFrame(rows)
    output = add_variant_order(output, variant_order)
    output = output.sort_values(["task", "system", "variant_order", "label"]).reset_index(drop=True)
    return output


def compute_detector_report(
    prediction_df: pd.DataFrame,
    no_accent_variants: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detector report is computed once per text instance, not once per task/system.
    """
    unique_df = (
        prediction_df[
            [
                "variant",
                "original_id",
                "text",
                "detected_no_accent",
                "oracle_no_accent",
                "alpha_chars",
                "word_count",
                "marked_vietnamese_chars",
                "accented_ratio",
            ]
        ]
        .drop_duplicates(["variant", "original_id"])
        .copy()
    )

    unique_df["true_no_accent"] = unique_df["variant"].isin(no_accent_variants)

    overall = detector_metrics(
        y_true=unique_df["true_no_accent"],
        y_pred=unique_df["detected_no_accent"],
    )
    overall_df = pd.DataFrame([{**{"scope": "overall"}, **overall}])

    variant_rows = []
    for variant, group_df in unique_df.groupby("variant"):
        route_count = int(group_df["detected_no_accent"].sum())
        total = len(group_df)
        route_rate = route_count / total if total else 0.0

        row = {
            "scope": "variant",
            "variant": variant,
            "total": total,
            "route_to_char_svm": route_count,
            "route_to_phobert": int(total - route_count),
            "route_to_char_rate": round(float(route_rate), 6),
            "true_no_accent_variant": variant in no_accent_variants,
            "mean_accented_ratio": round(float(group_df["accented_ratio"].mean()), 6),
            "median_accented_ratio": round(float(group_df["accented_ratio"].median()), 6),
        }
        variant_rows.append(row)

    variant_df = pd.DataFrame(variant_rows)

    return overall_df, variant_df


def build_robust_inference_comparison(
    results_df: pd.DataFrame,
    drop_df: pd.DataFrame,
    variant_order: List[str],
) -> pd.DataFrame:
    merged = results_df.merge(
        drop_df[
            [
                "task",
                "system",
                "variant",
                "clean_macro_f1",
                "variant_macro_f1",
                "macro_f1_drop",
                "macro_f1_relative_drop_percent",
            ]
        ],
        on=["task", "system", "variant"],
        how="left",
    )

    merged["rank_macro_f1_within_variant"] = (
        merged.groupby(["task", "variant"])["macro_f1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    phobert_ref = merged[merged["system"] == "phobert_only"][
        ["task", "variant", "macro_f1", "accuracy"]
    ].rename(
        columns={
            "macro_f1": "phobert_only_macro_f1",
            "accuracy": "phobert_only_accuracy",
        }
    )

    oracle_ref = merged[merged["system"] == "oracle_router"][
        ["task", "variant", "macro_f1", "accuracy"]
    ].rename(
        columns={
            "macro_f1": "oracle_router_macro_f1",
            "accuracy": "oracle_router_accuracy",
        }
    )

    merged = merged.merge(phobert_ref, on=["task", "variant"], how="left")
    merged = merged.merge(oracle_ref, on=["task", "variant"], how="left")

    merged["macro_f1_gain_vs_phobert"] = (
        merged["macro_f1"] - merged["phobert_only_macro_f1"]
    ).round(6)
    merged["accuracy_gain_vs_phobert"] = (
        merged["accuracy"] - merged["phobert_only_accuracy"]
    ).round(6)
    merged["macro_f1_gap_to_oracle"] = (
        merged["oracle_router_macro_f1"] - merged["macro_f1"]
    ).round(6)

    output = add_variant_order(merged, variant_order)
    output = output.sort_values(["task", "variant_order", "rank_macro_f1_within_variant", "system"])
    return output.reset_index(drop=True)


def plot_metric_by_variant(
    results_df: pd.DataFrame,
    task_name: str,
    metric: str,
    variant_order: List[str],
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plot_df = results_df[results_df["task"] == task_name].copy()

    plt.figure(figsize=(12, 6))

    for system, system_df in plot_df.groupby("system"):
        system_df = system_df.set_index("variant").reindex(variant_order).reset_index()
        plt.plot(system_df["variant"], system_df[metric], marker="o", label=system)

    plt.title(title)
    plt.xlabel("Variant")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_robust_inference_report(
    output_path: str | Path,
    selected_threshold: Dict[str, Any],
    detector_overall: pd.DataFrame,
    detector_by_variant: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    focus = comparison_df[
        comparison_df["variant"].isin(["no_accent", "mixed_no_accent"])
    ].copy()

    router_focus = focus[focus["system"] == "robust_router"].copy()
    oracle_focus = focus[focus["system"] == "oracle_router"].copy()

    lines = []
    lines.append("# Stage 8 — Robust Inference Strategy Report")
    lines.append("")
    lines.append("## 1. Cách đánh giá")
    lines.append("")
    lines.append(
        "Stage 8 evaluates a robust inference strategy that uses PhoBERT by default "
        "and routes suspected no-accent inputs to TF-IDF char SVM."
    )
    lines.append("")
    lines.append("## 2. Selected no-accent detector threshold")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(selected_threshold, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 3. Detector overall report")
    lines.append("")
    lines.append("```text")
    lines.append(detector_overall.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Detector by variant")
    lines.append("")
    lines.append("```text")
    lines.append(detector_by_variant.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 5. Full robust inference comparison")
    lines.append("")
    lines.append("```text")
    lines.append(comparison_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 6. No-accent focused results")
    lines.append("")
    lines.append("```text")
    lines.append(focus.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 7. Ghi chú")
    lines.append("")
    lines.append("- Stage này cải thiện cách chọn mô hình khi suy luận, không đổi trọng số PhoBERT.")
    lines.append("- Ngưỡng detector được chọn trên noisy validation.")
    lines.append("- Robust router là cấu hình dùng cho demo.")
    lines.append("- Oracle router chỉ dùng để tham khảo khi biết trước loại nhiễu.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
