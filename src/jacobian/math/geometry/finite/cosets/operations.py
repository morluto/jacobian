"""Exact occupied-coset partition by RREF quotient reduction."""

import time
from itertools import groupby
from operator import itemgetter

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.finite._models import LinearSubspace
from jacobian.math.geometry.finite.cosets._models import (
    CosetIntersection,
    CosetIntersectionProfile,
    CosetIntersectionSource,
)
from jacobian.math.geometry.finite.operations import _admit_linear_subspace
from jacobian.math.geometry.finite.values import PrimeFieldVectorSpace


def coset_intersection_profile(
    space: PrimeFieldVectorSpace,
    subspace: LinearSubspace,
    subset: tuple[tuple[int, ...], ...],
) -> CosetIntersectionProfile:
    """Partition a finite subset by cosets, retaining its ordered parent.

    A representative has zero in every RREF pivot coordinate. Every reduction
    subtracts a subspace vector, and these normal representatives are unique.
    """
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return coset_intersection_profile(space, subspace, subset)
    deadline = execution.started_at + 60.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before coset partition admission")
    source = CosetIntersectionSource(space=space, subspace=subspace, subset=subset)
    size, dimension, rank = len(subset), len(space.axis), len(subspace.basis)
    # At most min(N, p**(n-r)) occupied cosets. Retained input, members and
    # representatives are bounded separately, plus the retained RREF basis. Coordinates
    # and modular intermediates fit 14 and 28 bits respectively (p <= 10000).
    occupied = min(size, space.field_order ** max(0, dimension - rank))
    coordinates = (2 * size + occupied + rank) * dimension
    # RREF recognition, quotient reduction and comparison sorting of the
    # representatives. Source canonical ordering already orders each fibre.
    reduction_rank = rank if rank < dimension else 0
    work = rank * rank * dimension + size * dimension * reduction_rank
    if rank < dimension:
        work += 2 * size * dimension * max(1, size.bit_length())
    if coordinates > 1_048_576 or work > 67_108_864:
        raise OperationResourceAdmissionError(
            location=("subset",),
            code="finite_geometry.coset_partition_budget",
            message="coset partition exceeds coordinate allocation or reduction work",
        )
    # Supplied structural values need their mathematical claims recognized
    # once, after admission; result parsing never repeats this computation.
    basis = _admit_linear_subspace(subspace)
    pivots = tuple(next(i for i, value in enumerate(row) if value) for row in basis)
    modulus = space.field_order
    rows: list[CosetIntersection] = []
    if rank == dimension:
        if subset:
            rows.append(
                CosetIntersection(
                    representative=(0,) * dimension, members=subset, cardinality=size
                )
            )
    else:
        reductions = []
        for index, vector in enumerate(subset):
            if index % 256 == 0:
                request_checkpoint("during coset quotient reduction")
            word = list(vector)
            for pivot, row in zip(pivots, basis, strict=True):
                factor = word[pivot]
                if factor:
                    for j, value in enumerate(row):
                        word[j] = (word[j] - factor * value) % modulus
            reductions.append((tuple(word), vector))
        # Comparison sorting supplies a deterministic worst-case work bound,
        # independent of hash collisions. Stability preserves source ordering.
        reductions.sort(key=itemgetter(0))
        for representative, group in groupby(reductions, key=itemgetter(0)):
            members = tuple(vector for _, vector in group)
            rows.append(
                CosetIntersection(
                    representative=representative,
                    members=members,
                    cardinality=len(members),
                )
            )
    result = CosetIntersectionProfile(
        space=source.space,
        subspace=source.subspace,
        subset=source.subset,
        rows=tuple(rows),
    )
    request_checkpoint("after coset partition construction")
    return result
