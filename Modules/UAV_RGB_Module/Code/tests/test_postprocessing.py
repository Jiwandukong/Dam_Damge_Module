import pytest
import torch

from uav_rgb.postprocessing import (
    normalize_mask_probabilities,
    union_selected_candidates,
)


def test_score_boundary_is_inclusive_but_raw_mask_boundary_is_exclusive():
    probabilities = torch.tensor(
        [
            [[0.5, 0.5001], [0.1, 0.9]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    scores = torch.tensor([0.5, 0.4999])

    mask, keep = union_selected_candidates(
        probabilities,
        scores,
        confidence_threshold=0.5,
        mask_threshold=0.5,
    )

    assert keep.tolist() == [True, False]
    assert mask.tolist() == [[False, True], [False, True]]


def test_selected_candidates_are_combined_with_binary_union():
    probabilities = torch.tensor(
        [
            [[0.9, 0.0], [0.0, 0.0]],
            [[0.0, 0.9], [0.0, 0.0]],
        ]
    )
    mask, keep = union_selected_candidates(
        probabilities,
        torch.tensor([0.8, 0.7]),
        confidence_threshold=0.5,
        mask_threshold=0.5,
    )

    assert keep.tolist() == [True, True]
    assert mask.tolist() == [[True, True], [False, False]]


def test_empty_candidate_set_returns_empty_bool_mask():
    mask, keep = union_selected_candidates(
        torch.empty((0, 2, 3)),
        torch.empty((0,)),
        confidence_threshold=0.5,
        mask_threshold=0.5,
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 3)
    assert not mask.any()
    assert keep.shape == (0,)


def test_normalization_only_changes_shape_and_never_applies_sigmoid():
    source = torch.tensor([[[[-2.0, 2.0]]]])
    normalized = normalize_mask_probabilities(source)
    assert normalized.shape == (1, 1, 2)
    assert normalized.tolist() == [[[-2.0, 2.0]]]

    with pytest.raises(RuntimeError, match="unexpected masks_logits shape"):
        normalize_mask_probabilities(torch.zeros((1, 2, 3, 4)))
