from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str] | None = None,
) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        label = label_names[i] if label_names and i < len(label_names) else str(i)
        metrics[f'{label}_precision'] = float(p)
        metrics[f'{label}_recall'] = float(r)
        metrics[f'{label}_f1'] = float(f)
        metrics[f'{label}_support'] = int(s)

    return metrics
