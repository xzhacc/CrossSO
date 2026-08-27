from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.no_grad()
def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute non-interpolated binary AP with threshold-consistent tie handling."""
    if scores.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {tuple(scores.shape)} != {tuple(targets.shape)}")
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional inputs, got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError("Average precision scores must be finite")
    labels = (targets > 0).to(torch.float64)
    positive_count = labels.sum()
    if int(positive_count) == 0:
        return scores.new_tensor(float("nan"), dtype=torch.float64)

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_ends = torch.ones_like(sorted_labels, dtype=torch.bool)
    threshold_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    true_positives = sorted_labels.cumsum(0)[threshold_ends]
    ranks = torch.arange(
        1, sorted_labels.numel() + 1, device=scores.device, dtype=torch.float64
    )[threshold_ends]
    precision = true_positives / ranks
    recall = true_positives / positive_count
    previous_recall = torch.cat([recall.new_zeros(1), recall[:-1]])
    return ((recall - previous_recall) * precision).sum()


@torch.no_grad()
def multilabel_average_precision(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | torch.Tensor | int]:
    """Return explicit macro, positive-support-weighted, and micro AP reductions.

    Classes without a positive example are excluded from all three reductions.
    """
    if scores.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {tuple(scores.shape)} != {tuple(targets.shape)}")
    if scores.ndim != 2:
        raise ValueError(f"Expected [samples, classes], got {tuple(scores.shape)}")
    positive_support = (targets > 0).sum(0).long()
    valid = positive_support > 0
    per_class = scores.new_full((scores.size(1),), float("nan"), dtype=torch.float64)
    for class_index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        per_class[class_index] = binary_average_precision(
            scores[:, class_index], targets[:, class_index]
        )

    if not valid.any():
        nan = float("nan")
        return {
            "macro_supported": nan,
            "support_weighted": nan,
            "micro": nan,
            "classes_with_positives": 0,
            "per_class_ap": per_class,
            "positive_support": positive_support,
        }

    supported_ap = per_class[valid]
    supported_weights = positive_support[valid].to(torch.float64)
    macro = supported_ap.mean()
    support_weighted = (supported_ap * supported_weights).sum() / supported_weights.sum()
    micro = binary_average_precision(scores[:, valid].reshape(-1), targets[:, valid].reshape(-1))
    return {
        "macro_supported": float(macro),
        "support_weighted": float(support_weighted),
        "micro": float(micro),
        "classes_with_positives": int(valid.sum()),
        "per_class_ap": per_class,
        "positive_support": positive_support,
    }


@torch.no_grad()
def per_class_retrieval_ap_at_ks(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return AP@k for every class and each class's positive count."""
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {tuple(preds.shape)} != {tuple(targets.shape)}")
    if preds.ndim != 2:
        raise ValueError(f"Expected [samples, classes], got {tuple(preds.shape)}")
    if not ks:
        raise ValueError("ks must not be empty")
    labels = (targets > 0).long()
    n, classes = labels.shape
    positives = labels.sum(0)
    if n == 0:
        empty = preds.new_full((classes,), float("nan"))
        return {f"AP@{int(k)}": empty.clone() for k in ks}, positives
    max_k = min(max(int(k) for k in ks), n)
    indices = torch.argsort(preds, dim=0, descending=True, stable=True)[:max_k]
    relevant = labels.gather(0, indices).float()
    precision = relevant.cumsum(0) / torch.arange(
        1, max_k + 1, device=preds.device
    )[:, None]
    precision = precision * relevant
    out: dict[str, torch.Tensor] = {}
    valid = positives > 0
    for k in ks:
        end = min(int(k), max_k)
        denom = torch.minimum(positives, torch.full_like(positives, end)).clamp_min(1)
        ap = precision[:end].sum(0) / denom
        out[f"AP@{int(k)}"] = torch.where(
            valid, ap, torch.full_like(ap, float("nan"))
        )
    return out, positives


@torch.no_grad()
def retrieval_map_at_ks(preds: torch.Tensor, targets: torch.Tensor, ks: Sequence[int]) -> dict[str, float]:
    per_class, positives = per_class_retrieval_ap_at_ks(preds, targets, ks)
    out: dict[str, float] = {}
    for k in ks:
        valid = positives > 0
        ap = per_class[f"AP@{int(k)}"]
        out[f"mAP@{k}"] = float(ap[valid].mean()) if valid.any() else float("nan")
    return out
