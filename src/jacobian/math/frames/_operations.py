"""Domain functions for frame operations."""

from __future__ import annotations

import math

from jacobian.math.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    FrameRequest,
    GramResult,
)


def compute_gram(request: FrameRequest) -> GramResult:
    """Compute the Gram matrix G = <v_i, v_j>."""
    vectors = request.vectors
    n = len(vectors)
    gram = []
    for i in range(n):
        row = []
        for j in range(n):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
            row.append(dot)
        gram.append(tuple(row))
    return GramResult(gram=tuple(gram), dimension=len(vectors[0]))


def compute_coherence(request: FrameRequest) -> CoherenceResult:
    """Compute frame coherence: max off-diagonal |<v_i, v_j>| / (||v_i|| ||v_j||)."""
    vectors = request.vectors
    n = len(vectors)
    if n < 2:
        return CoherenceResult(coherence=0.0)
    max_off = 0.0
    for i in range(n):
        norm_i = math.sqrt(sum(x * x for x in vectors[i]))
        if norm_i == 0:
            continue
        for j in range(i + 1, n):
            norm_j = math.sqrt(sum(x * x for x in vectors[j]))
            if norm_j == 0:
                continue
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
            val = abs(dot) / (norm_i * norm_j)
            if val > max_off:
                max_off = val
    return CoherenceResult(coherence=max_off)


def compute_frame_potential(request: FrameRequest) -> FramePotentialResult:
    """Compute frame potential: sum_{i,j} |<v_i, v_j>|^2."""
    vectors = request.vectors
    n = len(vectors)
    total = 0.0
    for i in range(n):
        for j in range(n):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
            total += dot * dot
    return FramePotentialResult(potential=total)
