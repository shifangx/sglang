"""Validate the processor-expanded Nemotron 3.5 image/token contract."""

import math

import torch


def image_token_spans(
    input_ids, *, image_id, start_id, end_id, token_counts, unexpanded_ids=None
):
    """Return inclusive image spans; reject incomplete or unclaimed placeholders.

    Image boundaries matter even for NoPE: moving an image across text changes
    both the recurrent state and attention history.
    """
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] != image_id:
            i += 1
            continue
        start = i
        while i < len(ids) and ids[i] == image_id:
            i += 1
        if (
            start == 0
            or ids[start - 1] != start_id
            or i == len(ids)
            or ids[i] != end_id
        ):
            raise ValueError(
                "Nemotron image tokens must be enclosed by <img> and </img>"
            )
        spans.append((start, i - 1))
    lengths = [end - start + 1 for start, end in spans]
    if lengths != list(token_counts):
        raise ValueError(
            f"Nemotron image token/features mismatch: spans={lengths}, features={list(token_counts)}"
        )
    if unexpanded_ids is not None:
        collapsed = []
        cursor = 0
        for start, end in spans:
            collapsed.extend(ids[cursor : start - 1])
            collapsed.append(image_id)
            cursor = end + 2
        collapsed.extend(ids[cursor:])
        if collapsed != list(unexpanded_ids):
            raise ValueError(
                "Nemotron processor moved image placeholders or changed surrounding prompt tokens"
            )
    return spans


def image_token_counts(pixel_values, *, patch_size, downsample_ratio):
    """Normalize stacked/ragged images and count features after pixel shuffle."""
    if isinstance(pixel_values, torch.Tensor):
        images = list(pixel_values) if pixel_values.ndim == 4 else [pixel_values]
    elif isinstance(pixel_values, (list, tuple)):
        images = list(pixel_values)
    else:
        raise ValueError("Nemotron processor must return image tensors in pixel_values")
    factor = round(1 / downsample_ratio) if downsample_ratio > 0 else 0
    if factor < 1 or not math.isclose(factor * downsample_ratio, 1.0):
        raise ValueError(
            "Nemotron downsample_ratio must be the reciprocal of an integer"
        )
    multiple = patch_size * factor
    counts = []
    normalized = []
    for image in images:
        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Nemotron images must have shape (3, H, W)")
        h, w = image.shape[-2:]
        if not h or not w or h % multiple or w % multiple:
            raise ValueError(
                f"Nemotron image dimensions must be positive multiples of {multiple}"
            )
        normalized.append(image)
        counts.append((h // multiple) * (w // multiple))
    return normalized, counts
