"""Product representation profile kernel."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.product_representation._models import (
    ProductRepresentationResult,
    RepresentationEntry,
)
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet

__all__ = ["compute_product_representation_profile"]

MAX_PRODUCT_REPRESENTATION_PAIRS = 100_000
# Parsing and multiplying decimal integers is work proportional to their
# operand widths.  Bound that work independently of the pair and wire limits:
# a single pair must not turn a large-but-serializable value into an
# effectively unbounded native call.
MAX_PRODUCT_REPRESENTATION_DIGIT_WORK = 10_000_000_000


def _admit_product_representation(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> None:
    pair_count = len(left.elements) * len(right.elements)
    if pair_count > MAX_PRODUCT_REPRESENTATION_PAIRS:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="additive.product_representation_pair_work_exceeded",
            message=(
                "product representation exceeds the "
                f"{MAX_PRODUCT_REPRESENTATION_PAIRS}-pair work bound"
            ),
        )

    maximum_product_digits = max(
        (len(value.lstrip("-")) for value in left.elements),
        default=1,
    ) + max(
        (len(value.lstrip("-")) for value in right.elements),
        default=1,
    )
    # Decimal multiplication is quadratic in operand width for the native
    # integers used here.  Apply this independent work bound before parsing
    # any caller-supplied value into a Python integer.
    digit_work = pair_count * maximum_product_digits**2
    if digit_work > MAX_PRODUCT_REPRESENTATION_DIGIT_WORK:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="additive.product_representation_digit_work_exceeded",
            message=(
                "product representation arithmetic exceeds the "
                f"{MAX_PRODUCT_REPRESENTATION_DIGIT_WORK}-unit digit work bound"
            ),
        )


def compute_product_representation_profile(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> ProductRepresentationResult:
    """Return the complete exact product representation profile.

    For every x, the multiplicity r(x) = |{(a,b) in A x B : a*b = x}|.
    """
    _admit_product_representation(left, right)
    counts: dict[int, int] = {}
    for a in left.elements:
        for b in right.elements:
            product = parse_canonical_integer(a) * parse_canonical_integer(b)
            counts[product] = counts.get(product, 0) + 1

    entries = tuple(
        RepresentationEntry(product=format_canonical_integer(p), multiplicity=m)
        for p, m in sorted(counts.items())
    )

    return ProductRepresentationResult(
        left=left,
        right=right,
        entries=entries,
        support_cardinality=len(counts),
    )
