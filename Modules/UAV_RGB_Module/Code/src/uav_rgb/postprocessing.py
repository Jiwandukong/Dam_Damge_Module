"""SAM-3 mask postprocessing matching the August AUG512 handoff."""

from __future__ import annotations

import torch


def normalize_mask_probabilities(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize SAM-3 mask output shape to ``[N, H, W]``.

    This function deliberately does not apply sigmoid.  The locked evaluator
    thresholds the processor's raw ``masks_logits`` value directly.
    """
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise RuntimeError(f"unexpected masks_logits shape: {tuple(tensor.shape)}")
    return tensor


def union_selected_candidates(
    mask_probabilities: torch.Tensor,
    scores: torch.Tensor,
    *,
    confidence_threshold: float,
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by ``score >= threshold`` and OR raw masks ``> threshold``.

    There is intentionally no sigmoid, top-k selection, or class-wise argmax.
    """
    available = min(int(mask_probabilities.shape[0]), int(scores.numel()))
    probabilities = mask_probabilities[:available]
    scores = scores[:available]
    keep = (
        scores >= confidence_threshold
        if available
        else torch.zeros(0, dtype=torch.bool, device=mask_probabilities.device)
    )
    if int(keep.sum().item()) == 0:
        mask = torch.zeros(
            tuple(mask_probabilities.shape[-2:]),
            dtype=torch.bool,
            device=mask_probabilities.device,
        )
    else:
        mask = (probabilities[keep] > mask_threshold).any(dim=0)
    return mask, keep
