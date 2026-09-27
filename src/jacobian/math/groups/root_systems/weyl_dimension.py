"""Exact Weyl dimension formula on bounded finite root systems."""

from __future__ import annotations

from math import gcd

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    MAX_HIGHEST_WEIGHT_BITS,
    MAX_POSITIVE_ROOTS,
    MAX_ROOT_COORDINATE,
    MAX_WEYL_DIMENSION_BITS,
    CartanMatrix,
    WeylDimensionFactor,
    WeylDimensionResult,
)
from jacobian.math.groups.root_systems.operations import (
    _admit_cartan_finite_type,
    _as_cartan,
    _cartan_datum_from_admitted,
    _positive_coroots_from_admitted,
)

_MAX_DIMENSION_OUTPUT_CELLS = 64_000
_MAX_FORMULA_WORK = 100_000_000


def weyl_dimension(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    highest_weight: tuple[int, ...],
) -> WeylDimensionResult:
    """Return ``dim(V_λ)`` and its complete positive-root factor profile.

    Weight coordinates are pairings with the ordered simple coroots, hence
    coordinates in the fundamental-weight basis. Components are handled in
    the same product: this is the dimension of the irreducible representation
    of the direct-sum semisimple Lie algebra with the supplied highest weight.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    rank = len(cartan)
    if (
        not isinstance(highest_weight, (tuple, list))
        or len(highest_weight) != rank
        or any(type(value) is not int or value < 0 for value in highest_weight)
    ):
        raise OperationDomainValidationError(
            location=("highest_weight",),
            code="root_system.invalid_dominant_weight",
            message="highest weight must be a nonnegative integer vector on the Cartan axis",
        )
    weight = tuple(highest_weight)
    max_bits = max((value.bit_length() for value in weight), default=0)
    if max_bits > MAX_HIGHEST_WEIGHT_BITS:
        raise OperationResourceAdmissionError(
            location=("highest_weight",),
            code="root_system.weyl_dimension_bounds",
            message=(
                "highest-weight coordinates exceed the admitted "
                f"{MAX_HIGHEST_WEIGHT_BITS}-bit bound"
            ),
        )

    # Every supported finite type has at most 120 positive roots. Reserve the
    # full root family and bound all pairings, product growth, and JSON output
    # before asking the root backend to enumerate any roots.
    root_count_bound = MAX_POSITIVE_ROOTS
    pairing_bits_bound = max_bits + (rank * MAX_ROOT_COORDINATE + 1).bit_length() + 2
    dimension_bits_bound = root_count_bound * pairing_bits_bound
    work_bound = root_count_bound * (4 * rank + 8) + (
        root_count_bound * pairing_bits_bound * dimension_bits_bound
    )
    output_bytes_bound = (
        4_096
        + root_count_bound * (128 + 16 * rank + 2 * (pairing_bits_bound // 3 + 8))
        + (dimension_bits_bound + 7) // 8
    )
    if (
        root_count_bound > MAX_POSITIVE_ROOTS
        or dimension_bits_bound > MAX_WEYL_DIMENSION_BITS
        or work_bound > _MAX_FORMULA_WORK
        or output_bytes_bound > _MAX_DIMENSION_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("highest_weight",),
            code="root_system.weyl_dimension_bounds",
            message="Weyl dimension formula exceeds its admitted work or output envelope",
        )

    # Finite-type recognition runs once, after resource admission.
    _admit_cartan_finite_type(rows)
    datum = _cartan_datum_from_admitted(cartan)
    coroot_data = _positive_coroots_from_admitted(datum)
    pairs = coroot_data.positive_root_coroot_pairs
    if not 1 <= len(pairs) <= MAX_POSITIVE_ROOTS:
        raise RuntimeError("finite root family exceeded its admitted root-count bound")

    dimension_numerator = 1
    dimension_denominator = 1
    profile: list[WeylDimensionFactor] = []
    for pair in pairs:
        denominator = sum(pair.coroot_coefficients)
        numerator = sum(
            (coordinate + 1) * coroot_coordinate
            for coordinate, coroot_coordinate in zip(
                weight, pair.coroot_coefficients, strict=True
            )
        )
        if denominator <= 0 or numerator <= 0:
            raise RuntimeError("positive coroot produced a nonpositive Weyl factor")
        profile.append(
            WeylDimensionFactor(
                positive_root=pair.root_coefficients,
                positive_coroot=pair.coroot_coefficients,
                numerator_pairing=numerator,
                denominator_pairing=denominator,
            )
        )
        dimension_numerator *= numerator
        dimension_denominator *= denominator
        common = gcd(dimension_numerator, dimension_denominator)
        dimension_numerator //= common
        dimension_denominator //= common
        if dimension_numerator.bit_length() > MAX_WEYL_DIMENSION_BITS:
            raise RuntimeError("pre-admitted Weyl dimension bound was exceeded")

    if dimension_denominator != 1:
        raise RuntimeError("Weyl dimension product did not yield an integer")
    return WeylDimensionResult._from_kernel(
        cartan, weight, dimension_numerator, tuple(profile)
    )
