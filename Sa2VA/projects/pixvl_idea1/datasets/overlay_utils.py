from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


def _compute_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _expand(binary: np.ndarray, px: int) -> np.ndarray:
    expanded = binary.copy()
    for _ in range(max(px - 1, 0)):
        nxt = expanded.copy()
        nxt[1:, :] |= expanded[:-1, :]
        nxt[:-1, :] |= expanded[1:, :]
        nxt[:, 1:] |= expanded[:, :-1]
        nxt[:, :-1] |= expanded[:, 1:]
        expanded = nxt
    return expanded


def build_overlay_image(
    image: Image.Image,
    mask: np.ndarray,
    darken_alpha: float = 0.4,
    boundary_px: int = 2,
    boundary_color: Iterable[int] = (255, 64, 64),
) -> Image.Image:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32)
    mask = mask.astype(bool)
    overlay = image_np.copy()

    overlay[~mask] *= darken_alpha

    boundary = _compute_boundary(mask)
    if boundary_px > 1:
        boundary = _expand(boundary, boundary_px)
    overlay[boundary] = np.asarray(list(boundary_color), dtype=np.float32)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

