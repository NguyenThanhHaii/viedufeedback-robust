from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


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

SENTIMENT_ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

SENTIMENT_VI = {
    "negative": "Tiêu cực",
    "neutral": "Trung lập",
    "positive": "Tích cực",
}

TOPIC_ID2LABEL = {
    0: "lecturer",
    1: "training_program",
    2: "facility",
    3: "others",
}

TOPIC_VI = {
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
}


def project_root_from_demo_file(file_path: str | Path) -> Path:
    current = Path(file_path).resolve()
    return current.parents[1]


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
    min_alpha_chars: int = 12,
    min_words: int = 3,
) -> bool:
    if count_alpha_chars(text) < min_alpha_chars:
        return False
    if word_count(text) < min_words:
        return False
    return accented_ratio(text) <= threshold


def load_selected_threshold(root: Path, fallback: float = 0.10) -> float:
    path = root / "outputs" / "metrics" / "robust_inference_selected_threshold.json"
    if not path.exists():
        return fallback

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return float(payload.get("threshold", fallback))


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def phobert_model_candidates(root: Path, task: str) -> List[Path]:
    return [
        root / "outputs" / "models" / "phobert" / task / "best",
        root / "models" / "phobert" / task / "best",
        root / "outputs" / "models" / "phobert" / task,
        root / "models" / "phobert" / task,
    ]


def char_svm_candidates(root: Path, task: str) -> List[Path]:
    return [
        root / "outputs" / "models" / "baseline" / f"{task}_tfidf_char_svm.joblib",
        root / "outputs" / "models" / "baselines" / f"{task}_tfidf_char_svm.joblib",
        root / "models" / "baseline" / f"{task}_tfidf_char_svm.joblib",
        root / "models" / "baselines" / f"{task}_tfidf_char_svm.joblib",
        root / "outputs" / "models" / "baseline" / task / "tfidf_char_svm.joblib",
        root / "outputs" / "models" / "baselines" / task / "tfidf_char_svm.joblib",
        root / "models" / "baseline" / task / "tfidf_char_svm.joblib",
        root / "models" / "baselines" / task / "tfidf_char_svm.joblib",
    ]


def stable_softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(float(v) - max_value) for v in values]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


@dataclass
class PredictionResult:
    task: str
    model_used: str
    label_id: int
    label_en: str
    label_vi: str
    confidence: Optional[float]
    scores: Dict[str, float]


class PhoBERTPredictor:
    def __init__(self, model_dir: Path, task: str, max_length: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.task = task
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if task == "sentiment":
            self.id2label = SENTIMENT_ID2LABEL
            self.label_vi = SENTIMENT_VI
        elif task == "topic":
            self.id2label = TOPIC_ID2LABEL
            self.label_vi = TOPIC_VI
        else:
            raise ValueError(f"Unsupported task: {task}")

    def predict(self, text: str) -> PredictionResult:
        torch = self.torch

        inputs = self.tokenizer(
            str(text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        pred_id = int(max(range(len(probs)), key=lambda idx: probs[idx]))
        label_en = self.id2label.get(pred_id, str(pred_id))
        scores = {
            self.label_vi.get(self.id2label.get(idx, str(idx)), str(idx)): float(prob)
            for idx, prob in enumerate(probs)
        }

        return PredictionResult(
            task=self.task,
            model_used="PhoBERT",
            label_id=pred_id,
            label_en=label_en,
            label_vi=self.label_vi.get(label_en, label_en),
            confidence=float(probs[pred_id]),
            scores=scores,
        )


class CharSVMPredictor:
    def __init__(self, model_path: Path, task: str):
        self.task = task
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

        if task == "sentiment":
            self.id2label = SENTIMENT_ID2LABEL
            self.label_vi = SENTIMENT_VI
        elif task == "topic":
            self.id2label = TOPIC_ID2LABEL
            self.label_vi = TOPIC_VI
        else:
            raise ValueError(f"Unsupported task: {task}")

    def predict(self, text: str) -> PredictionResult:
        pred = self.pipeline.predict([str(text)])[0]
        pred_id = int(pred)
        label_en = self.id2label.get(pred_id, str(pred_id))

        confidence = None
        scores: Dict[str, float] = {}

        if hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function([str(text)])
            values = decision[0]
            if not hasattr(values, "__iter__"):
                values = [-float(values), float(values)]
            values = [float(v) for v in values]
            pseudo_probs = stable_softmax(values)
            scores = {
                self.label_vi.get(self.id2label.get(idx, str(idx)), str(idx)): float(score)
                for idx, score in enumerate(pseudo_probs)
            }
            if pred_id < len(pseudo_probs):
                confidence = float(pseudo_probs[pred_id])

        return PredictionResult(
            task=self.task,
            model_used="TF-IDF char SVM",
            label_id=pred_id,
            label_en=label_en,
            label_vi=self.label_vi.get(label_en, label_en),
            confidence=confidence,
            scores=scores,
        )


class RobustInferenceEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.threshold = load_selected_threshold(self.root, fallback=0.10)
        self.min_alpha_chars = 12
        self.min_words = 3

        self.phobert: Dict[str, PhoBERTPredictor] = {}
        self.char_svm: Dict[str, CharSVMPredictor] = {}
        self.load_errors: Dict[str, str] = {}

        self._load_models()

    def _load_models(self) -> None:
        for task in ["sentiment", "topic"]:
            phobert_dir = find_first_existing(phobert_model_candidates(self.root, task))
            if phobert_dir is not None:
                try:
                    self.phobert[task] = PhoBERTPredictor(phobert_dir, task=task)
                except Exception as exc:
                    self.load_errors[f"phobert_{task}"] = str(exc)
            else:
                self.load_errors[f"phobert_{task}"] = (
                    f"PhoBERT model directory not found for task={task}."
                )

            svm_path = find_first_existing(char_svm_candidates(self.root, task))
            if svm_path is not None:
                try:
                    self.char_svm[task] = CharSVMPredictor(svm_path, task=task)
                except Exception as exc:
                    self.load_errors[f"char_svm_{task}"] = str(exc)
            else:
                self.load_errors[f"char_svm_{task}"] = (
                    f"Char SVM model file not found for task={task}. "
                    "Run scripts/export_char_svm_models.py."
                )

    def detector_features(self, text: str) -> Dict[str, Any]:
        return {
            "alpha_chars": count_alpha_chars(text),
            "word_count": word_count(text),
            "marked_vietnamese_chars": count_marked_vietnamese_chars(text),
            "accented_ratio": accented_ratio(text),
            "threshold": self.threshold,
            "suspected_no_accent": detect_no_accent(
                text=text,
                threshold=self.threshold,
                min_alpha_chars=self.min_alpha_chars,
                min_words=self.min_words,
            ),
        }

    def predict_task(self, text: str, task: str, mode: str = "auto_router") -> PredictionResult:
        features = self.detector_features(text)
        suspected_no_accent = bool(features["suspected_no_accent"])

        if mode == "phobert_only":
            if task not in self.phobert:
                raise RuntimeError(f"PhoBERT model for {task} is not loaded.")
            return self.phobert[task].predict(text)

        if mode == "char_svm_only":
            if task not in self.char_svm:
                raise RuntimeError(f"TF-IDF char SVM model for {task} is not loaded.")
            return self.char_svm[task].predict(text)

        if mode != "auto_router":
            raise ValueError(f"Unsupported mode: {mode}")

        if suspected_no_accent and task in self.char_svm:
            return self.char_svm[task].predict(text)

        if task in self.phobert:
            return self.phobert[task].predict(text)

        if task in self.char_svm:
            return self.char_svm[task].predict(text)

        raise RuntimeError(f"No available model for task={task}.")

    def predict_both(self, text: str, mode: str = "auto_router") -> Dict[str, Any]:
        features = self.detector_features(text)
        sentiment = self.predict_task(text, "sentiment", mode=mode)
        topic = self.predict_task(text, "topic", mode=mode)

        return {
            "text": text,
            "mode": mode,
            "detector": features,
            "sentiment": sentiment,
            "topic": topic,
        }
