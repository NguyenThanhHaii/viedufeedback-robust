from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


# =========================
# General utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_project_root() -> Path:
    """
    Kaggle-friendly root detection.

    Expected layouts:
    1. Local repo:
       project/
         data/
         configs/
         kaggle/stage6_train_phobert.py

    2. Kaggle after unzip:
       /kaggle/working/viedufeedback_stage6/
         data/
         configs/
         kaggle/stage6_train_phobert.py
    """
    cwd = Path.cwd()

    candidates = [
        cwd,
        Path("/kaggle/working"),
        cwd.parent,
        Path(__file__).resolve().parents[1] if "__file__" in globals() else cwd,
    ]

    for candidate in candidates:
        if (candidate / "data" / "processed" / "train_standardized.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find project root containing data/processed/train_standardized.csv"
    )


def unzip_first_stage6_input() -> Path:
    """
    Optional helper for Kaggle:
    If a zip package exists under /kaggle/input, unzip it to /kaggle/working/viedufeedback_stage6.
    """
    kaggle_input = Path("/kaggle/input")
    target = Path("/kaggle/working/viedufeedback_stage6")

    if target.exists() and (target / "data" / "processed" / "train_standardized.csv").exists():
        return target

    if not kaggle_input.exists():
        return Path.cwd()

    zip_candidates = list(kaggle_input.rglob("viedufeedback_phobert_stage6.zip"))
    if not zip_candidates:
        return Path.cwd()

    ensure_dir(target)
    zip_path = zip_candidates[0]
    print(f"Unzipping Kaggle input: {zip_path} -> {target}")

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(target)

    return target


def get_label_names(mapping_path: Path) -> List[str]:
    mapping = read_json(mapping_path)
    id2label = mapping["id2label"]
    return [id2label[str(i)] for i in sorted(int(k) for k in id2label.keys())]


# =========================
# Dataset and model
# =========================

class TextClsDataset(Dataset):
    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = [str(x) for x in texts]
        self.labels = [int(y) for y in labels]
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[idx], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TrainState:
    best_val_macro_f1: float
    best_epoch: int
    best_model_dir: Path
    history: List[Dict[str, float]]


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
    }


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()

    all_true: List[int] = []
    all_pred: List[int] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(all_true, all_pred)
    metrics["loss"] = round(total_loss / max(total_count, 1), 6)

    return metrics, all_true, all_pred


def train_one_task(
    root: Path,
    config_path: Path,
    output_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    config = load_yaml(config_path)

    seed = int(config["seed"])
    set_seed(seed)

    task_name = config["task"]["name"]
    label_col = config["task"]["label_col"]
    num_labels = int(config["task"]["num_labels"])
    model_name = config["model"]["name"]
    max_length = int(config["model"]["max_length"])

    train_batch_size = int(config["training"]["train_batch_size"])
    eval_batch_size = int(config["training"]["eval_batch_size"])
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    warmup_ratio = float(config["training"]["warmup_ratio"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    use_amp = bool(config["training"].get("fp16", True)) and device.type == "cuda"

    train_file = root / config["paths"]["train_file"]
    validation_file = root / config["paths"]["validation_file"]
    test_file = root / config["paths"]["test_file"]
    noisy_eval_file = root / config["paths"]["noisy_eval_file"]
    mapping_file = root / config["paths"]["mapping_file"]

    label_names = get_label_names(mapping_file)

    print("=" * 80)
    print(f"Task: {task_name}")
    print(f"Model: {model_name}")
    print(f"Labels: {label_names}")
    print(f"Max length: {max_length}")
    print("=" * 80)

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(validation_file)
    test_df = pd.read_csv(test_file)
    noisy_eval_df = pd.read_csv(noisy_eval_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_dataset = TextClsDataset(
        train_df["text"].fillna("").astype(str).tolist(),
        train_df[label_col].astype(int).tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = TextClsDataset(
        val_df["text"].fillna("").astype(str).tolist(),
        val_df[label_col].astype(int).tolist(),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = TextClsDataset(
        test_df["text"].fillna("").astype(str).tolist(),
        test_df[label_col].astype(int).tolist(),
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    task_output_dir = ensure_dir(output_root / "models" / "phobert" / task_name)
    best_model_dir = ensure_dir(task_output_dir / "best")

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"{task_name} epoch {epoch}/{epochs}", leave=False)

        for batch in progress:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix({"loss": running_loss / max(seen, 1)})

        train_loss = running_loss / max(seen, 1)
        val_metrics, _, _ = evaluate(model, val_loader, device=device, use_amp=use_amp)

        row = {
            "task": task_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        print(
            f"[{task_name}] epoch={epoch} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_macro_f1={row['val_macro_f1']:.6f} "
            f"val_acc={row['val_accuracy']:.6f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            save_json(
                {
                    "task": task_name,
                    "model_name": model_name,
                    "label_names": label_names,
                    "max_length": max_length,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "config": config,
                },
                best_model_dir / "training_metadata.json",
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[{task_name}] early stopping at epoch {epoch}")
                break

    # Reload best model for evaluation
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir, use_fast=False)

    test_metrics, y_true, y_pred = evaluate(model, test_loader, device=device, use_amp=use_amp)

    print(
        f"[{task_name}] TEST macro_f1={test_metrics['macro_f1']:.6f} "
        f"accuracy={test_metrics['accuracy']:.6f}"
    )

    # Clean test predictions
    clean_pred_df = test_df.copy()
    clean_pred_df["variant"] = "clean"
    clean_pred_df["task"] = task_name
    clean_pred_df["model"] = "phobert"
    clean_pred_df["y_true"] = y_true
    clean_pred_df["y_pred"] = y_pred
    clean_pred_df["y_true_name"] = clean_pred_df["y_true"].map({i: n for i, n in enumerate(label_names)})
    clean_pred_df["y_pred_name"] = clean_pred_df["y_pred"].map({i: n for i, n in enumerate(label_names)})
    clean_pred_df["is_correct"] = clean_pred_df["y_true"] == clean_pred_df["y_pred"]

    clean_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_labels)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Evaluate robustness on all variants
    robustness_rows = []
    robustness_reports = []
    robustness_predictions = []

    variant_order = [
        "clean",
        "typo_light",
        "typo_medium",
        "teencode_light",
        "mixed_light",
        "no_accent",
        "mixed_no_accent",
    ]

    for variant in variant_order:
        if variant not in set(noisy_eval_df["variant"].unique()):
            continue

        variant_df = noisy_eval_df[noisy_eval_df["variant"] == variant].copy()

        variant_dataset = TextClsDataset(
            variant_df["text"].fillna("").astype(str).tolist(),
            variant_df[label_col].astype(int).tolist(),
            tokenizer,
            max_length=max_length,
        )
        variant_loader = DataLoader(
            variant_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

        metrics, vt_true, vt_pred = evaluate(model, variant_loader, device=device, use_amp=use_amp)

        robustness_rows.append(
            {
                "task": task_name,
                "model": "phobert",
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
            vt_true,
            vt_pred,
            labels=list(range(num_labels)),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        robustness_reports.extend(
            flatten_report(report_dict, task_name=task_name, model_name="phobert", variant=variant)
        )

        pred_cols = [
            "id",
            "original_id",
            "variant",
            "noise_type",
            "noise_level",
            "original_text",
            "text",
            "sentiment_label",
            "sentiment_name",
            "topic_label",
            "topic_name",
        ]
        pred_cols = [c for c in pred_cols if c in variant_df.columns]
        pred_df = variant_df[pred_cols].copy()
        pred_df["task"] = task_name
        pred_df["model"] = "phobert"
        pred_df["y_true"] = vt_true
        pred_df["y_pred"] = vt_pred
        pred_df["y_true_name"] = pred_df["y_true"].map({i: n for i, n in enumerate(label_names)})
        pred_df["y_pred_name"] = pred_df["y_pred"].map({i: n for i, n in enumerate(label_names)})
        pred_df["is_correct"] = pred_df["y_true"] == pred_df["y_pred"]
        robustness_predictions.append(pred_df)

    robustness_df = pd.DataFrame(robustness_rows)
    drop_df = compute_drop_from_clean(robustness_df)
    robustness_report_df = pd.DataFrame(robustness_reports)
    robustness_predictions_df = pd.concat(robustness_predictions, ignore_index=True)

    elapsed = time.time() - start_time

    return {
        "task": task_name,
        "label_names": label_names,
        "history": pd.DataFrame(history),
        "clean_result": {
            "task": task_name,
            "model": "phobert",
            "variant": "clean",
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(float(best_val_macro_f1), 6),
            "train_seconds": round(float(elapsed), 2),
        },
        "clean_report": clean_report,
        "clean_predictions": clean_pred_df,
        "robustness_results": robustness_df,
        "robustness_drop": drop_df,
        "robustness_class_report": robustness_report_df,
        "robustness_predictions": robustness_predictions_df,
        "best_model_dir": str(best_model_dir),
    }


# =========================
# Reporting helpers
# =========================

def flatten_report(report_dict: Dict, task_name: str, model_name: str, variant: str) -> List[Dict]:
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


def compute_drop_from_clean(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (task, model), group_df in results_df.groupby(["task", "model"]):
        clean = group_df[group_df["variant"] == "clean"].iloc[0]

        for _, row in group_df.iterrows():
            macro_drop = float(clean["macro_f1"] - row["macro_f1"])
            acc_drop = float(clean["accuracy"] - row["accuracy"])
            weighted_drop = float(clean["weighted_f1"] - row["weighted_f1"])

            rows.append(
                {
                    "task": task,
                    "model": model,
                    "variant": row["variant"],
                    "noise_type": row["noise_type"],
                    "noise_level": row["noise_level"],
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

    return pd.DataFrame(rows)


def safe_relative_drop(clean_value: float, drop_value: float) -> float:
    if clean_value == 0:
        return 0.0
    return round(float(drop_value / clean_value * 100), 4)


def save_training_curve(history_df: pd.DataFrame, task_name: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], marker="o", label="val_macro_f1")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"PhoBERT validation scores - {task_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix(y_true: List[int], y_pred: List[int], label_names: List[str], title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=list(range(len(label_names))), labels=label_names, rotation=30, ha="right")
    plt.yticks(ticks=list(range(len(label_names))), labels=label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_phobert_report(report_path: Path, clean_df: pd.DataFrame, robustness_df: pd.DataFrame, drop_df: pd.DataFrame) -> None:
    ensure_dir(report_path.parent)

    lines = []
    lines.append("# PhoBERT Fine-tuning Report")
    lines.append("")
    lines.append("## 1. Clean test results")
    lines.append("")
    lines.append("```text")
    lines.append(clean_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 2. Robustness results")
    lines.append("")
    lines.append("```text")
    lines.append(robustness_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 3. Robustness drop from clean")
    lines.append("")
    lines.append("```text")
    lines.append(drop_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Notes")
    lines.append("")
    lines.append("- Macro-F1 is the primary metric because the dataset is imbalanced.")
    lines.append("- PhoBERT is fine-tuned on clean train data and evaluated on clean/noisy test variants.")
    lines.append("- Noisy variants are not used for training.")
    lines.append("- The same noisy test set is used for baseline and PhoBERT comparison.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def zip_outputs(output_root: Path) -> Path:
    zip_path = Path("/kaggle/working/phobert_stage6_outputs.zip") if Path("/kaggle/working").exists() else output_root.parent / "phobert_stage6_outputs.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for path in output_root.rglob("*"):
            if path.is_file():
                zipf.write(path, arcname=str(path.relative_to(output_root.parent)))

    return zip_path


# =========================
# Main
# =========================

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    maybe_root = unzip_first_stage6_input()
    os.chdir(maybe_root)

    root = find_project_root()
    print(f"Project root: {root}")

    output_root = ensure_dir(Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else root / "outputs")

    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")
    predictions_dir = ensure_dir(output_root / "predictions")
    reports_dir = ensure_dir(output_root / "reports")
    metrics_dir = ensure_dir(output_root / "metrics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config_paths = [
        root / "configs" / "phobert_sentiment.yaml",
        root / "configs" / "phobert_topic.yaml",
    ]

    all_clean_rows = []
    all_histories = []
    all_clean_reports = []
    all_clean_predictions = []
    all_robustness_results = []
    all_robustness_drop = []
    all_robustness_reports = []
    all_robustness_predictions = []

    task_outputs = []

    for config_path in config_paths:
        task_result = train_one_task(
            root=root,
            config_path=config_path,
            output_root=output_root,
            device=device,
        )
        task_outputs.append(task_result)

        task_name = task_result["task"]

        history_df = task_result["history"]
        all_histories.append(history_df)

        clean_row = task_result["clean_result"]
        all_clean_rows.append(clean_row)

        clean_report_rows = flatten_report(
            task_result["clean_report"],
            task_name=task_name,
            model_name="phobert",
            variant="clean",
        )
        all_clean_reports.extend(clean_report_rows)

        all_clean_predictions.append(task_result["clean_predictions"])

        all_robustness_results.append(task_result["robustness_results"])
        all_robustness_drop.append(task_result["robustness_drop"])
        all_robustness_reports.append(task_result["robustness_class_report"])
        all_robustness_predictions.append(task_result["robustness_predictions"])

        save_training_curve(
            history_df,
            task_name=task_name,
            output_path=figures_dir / f"phobert_training_curve_{task_name}.png",
        )

        # Clean confusion matrix
        pred_df = task_result["clean_predictions"]
        save_confusion_matrix(
            y_true=pred_df["y_true"].astype(int).tolist(),
            y_pred=pred_df["y_pred"].astype(int).tolist(),
            label_names=task_result["label_names"],
            title=f"PhoBERT clean confusion matrix - {task_name}",
            output_path=figures_dir / f"confusion_matrix_phobert_clean_{task_name}.png",
        )

    clean_results_df = pd.DataFrame(all_clean_rows)
    history_df = pd.concat(all_histories, ignore_index=True)
    clean_report_df = pd.DataFrame(all_clean_reports)
    clean_predictions_df = pd.concat(all_clean_predictions, ignore_index=True)

    robustness_results_df = pd.concat(all_robustness_results, ignore_index=True)
    robustness_drop_df = pd.concat(all_robustness_drop, ignore_index=True)
    robustness_report_df = pd.concat(all_robustness_reports, ignore_index=True)
    robustness_predictions_df = pd.concat(all_robustness_predictions, ignore_index=True)

    clean_results_df.to_csv(tables_dir / "phobert_clean_results.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(tables_dir / "phobert_training_history.csv", index=False, encoding="utf-8-sig")
    clean_report_df.to_csv(tables_dir / "phobert_clean_classification_report.csv", index=False, encoding="utf-8-sig")
    clean_predictions_df.to_csv(predictions_dir / "phobert_clean_predictions.csv", index=False, encoding="utf-8-sig")

    robustness_results_df.to_csv(tables_dir / "phobert_robustness_results.csv", index=False, encoding="utf-8-sig")
    robustness_drop_df.to_csv(tables_dir / "phobert_robustness_drop.csv", index=False, encoding="utf-8-sig")
    robustness_report_df.to_csv(tables_dir / "phobert_robustness_class_report.csv", index=False, encoding="utf-8-sig")
    robustness_predictions_df.to_csv(predictions_dir / "phobert_robustness_predictions.csv", index=False, encoding="utf-8-sig")

    save_json(
        {
            "clean_results": clean_results_df.to_dict(orient="records"),
            "robustness_results": robustness_results_df.to_dict(orient="records"),
            "robustness_drop": robustness_drop_df.to_dict(orient="records"),
        },
        metrics_dir / "phobert_results.json",
    )

    write_phobert_report(
        report_path=reports_dir / "phobert_report.md",
        clean_df=clean_results_df,
        robustness_df=robustness_results_df,
        drop_df=robustness_drop_df,
    )

    zip_path = zip_outputs(output_root)

    print("\nPhoBERT Stage 6 completed.")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Predictions: {predictions_dir}")
    print(f"Reports: {reports_dir}")
    print(f"Models: {output_root / 'models' / 'phobert'}")
    print(f"Output zip: {zip_path}")


if __name__ == "__main__":
    main()
