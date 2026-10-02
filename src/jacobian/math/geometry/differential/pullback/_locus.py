"""Bound scalar planning and monic nonvanishing predicates independently."""

from collections import Counter
from fractions import Fraction

from jacobian.math.geometry.differential.metrics._dag import ZERO, Dag, Expression
from jacobian.math.polynomials.rational_functions._bounds import (
    _polynomial_backend_conversion_work_units,
)

_MAX_SCALAR_BITS = 65_536
_MAX_PLANNED_ALLOCATION_BITS = 268_435_456


def _bits(value: Fraction) -> tuple[int, int]:
    return abs(value.numerator).bit_length(), value.denominator.bit_length()


class PullbackDag(Dag):
    """Charge private scalar work even when its eventual locus is the unit."""

    def _scalar_work(self, left: Fraction, right: Fraction, *, add: bool) -> None:
        a, b = _bits(left)
        c, d = _bits(right)
        if not left or not right or (not add and (left in (-1, 1) or right in (-1, 1))):
            numerator, denominator = max(a, c), max(b, d)
            work = 1 + (a + b + c + d + 63) // 64
        else:
            numerator, denominator = (
                (max(a, c) + 1, b)
                if add and left.denominator == right.denominator
                else ((max(a + d, c + b) + 1, b + d) if add else (a + c, b + d))
            )
            # A squared machine-word bound covers products and Euclidean
            # reduction, including unreduced cross-products during addition.
            work = (1 + (a + b + c + d + 63) // 64) ** 2
        if max(numerator, denominator) > _MAX_SCALAR_BITS:
            self.ledger.limits.reject(
                "scalar_height",
                "private pullback scalars exceed the admitted bit envelope",
            )
        self.ledger.charge("normalization", work)
        self.ledger.allocation_bits += numerator + denominator
        if self.ledger.allocation_bits > _MAX_PLANNED_ALLOCATION_BITS:
            self.ledger.limits.reject(
                "allocation", "pullback scalar planning exceeds its allocation budget"
            )

    def multiply_checked(self, *values: Expression) -> Expression:
        numerator: Counter[int] = Counter()
        denominator: Counter[int] = Counter()
        scalar = Fraction(1)
        for value in values:
            self._scalar_work(scalar, value.scalar, add=False)
            scalar *= value.scalar
            if not scalar:
                return ZERO
            self.ledger.charge(
                "multiplication", 1 + len(value.numerator) + len(value.denominator)
            )
            numerator.update(value.numerator)
            denominator.update(value.denominator)
        common = numerator & denominator
        return Expression(
            scalar,
            tuple(sorted((numerator - common).elements())),
            tuple(sorted((denominator - common).elements())),
        )

    def add(self, *values: Expression) -> Expression:
        grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], Fraction] = {}
        for value in values:
            key = (value.numerator, value.denominator)
            current = grouped.get(key, Fraction(0))
            self._scalar_work(current, value.scalar, add=True)
            self.ledger.charge(
                "addition", 1 + len(value.numerator) + len(value.denominator)
            )
            grouped[key] = current + value.scalar
        # The shared DAG receives one term per factor identity; its scalar
        # grouping only copies these already bounded coefficients from zero.
        return super().add(
            *(Expression(scalar, *key) for key, scalar in grouped.items() if scalar)
        )


def admit_locus_factor(dag: Dag, index: int) -> tuple[int, int, int]:
    """Bound monic polynomial delivery, without charging its discarded content."""

    bound = dag.nodes[index].bound
    if (
        bound.terms > 256
        or max(bound.degrees, default=0) > 64
        or bound.coefficient_digits > 128
    ):
        dag.ledger.limits.reject(
            "locus_factor", "pullback locus factors exceed the monic polynomial carrier"
        )
    # The exact polynomial is q*A for an integral A of bounded coefficient
    # height. Monic normalization returns A/LC(A), so both components of each
    # reduced coefficient are bounded by A's height; q cancels entirely.
    content_bits = sum(_bits(bound.rational_content))
    raw_bits = content_bits + 8 * bound.coefficient_digits
    dag.ledger.charge(
        "source_conversion", _polynomial_backend_conversion_work_units(bound)
    )
    dag.ledger.charge(
        "normalization",
        max(bound.terms, dag.dense(bound.degrees)) * (1 + (raw_bits + 63) // 64) ** 2,
    )
    return (
        bound.terms,
        8 * bound.coefficient_digits * bound.terms,
        dag.dimension * (bound.terms + 1),
    )


def locus_presentation(dag: Dag, factors: tuple[int, ...]) -> tuple[int, ...]:
    """Keep a bounded whole predicate, splitting only when its product is too big.

    Multiplicities are irrelevant to nonvanishing. Retaining a safely bounded
    product prevents splitting an already legal family past its guard-count cap.
    This estimates before adding any multiplication node to the executable DAG.
    """

    unique = tuple(sorted(set(factors)))
    if len(unique) < 2:
        return unique
    degrees = (0,) * dag.dimension
    terms, digits = 1, 0
    content_numerator_bits = content_denominator_bits = 0
    allocation_bits = product_work = 0
    for position, index in enumerate(unique):
        bound = dag.nodes[index].bound
        dag.ledger.charge("normalization", 1 + dag.dimension)
        degrees = tuple(a + b for a, b in zip(degrees, bound.degrees, strict=True))
        collision = min(terms, bound.terms)
        digits += bound.coefficient_digits + (
            len(str(collision)) if position and collision > 1 else 0
        )
        product_work += terms * bound.terms
        terms = min(terms * bound.terms, dag.dense(degrees))
        numerator_bits, denominator_bits = _bits(bound.rational_content)
        content_numerator_bits += numerator_bits
        content_denominator_bits += denominator_bits
        content_digits = content_numerator_bits // 3 + content_denominator_bits // 3 + 2
        allocation_bits += max(terms, dag.dense(degrees)) * (
            16 + 4 * (digits + content_digits) + 64 * dag.dimension
        )
        if (
            max(degrees, default=0) > 64
            or terms > 256
            or digits > 128
            # 2^(3*d) < 10^d gives a sufficient raw-content admission.
            or content_numerator_bits + 3 * digits > 3 * dag.ledger.limits.raw_digits
            or content_denominator_bits > 3 * dag.ledger.limits.raw_digits
            or dag.ledger.allocation_bits + allocation_bits
            > _MAX_PLANNED_ALLOCATION_BITS
            or len(dag.nodes) + len(unique) >= 16384
        ):
            return unique
    scalar_work = (
        len(unique)
        * (1 + (content_numerator_bits + content_denominator_bits + 63) // 64) ** 2
    )
    normalization_work = (
        dag.dense(degrees)
        * (
            1
            + (content_numerator_bits + content_denominator_bits + 8 * digits + 63)
            // 64
        )
        ** 2
    )
    conversion_work = terms * (1 + (digits + content_digits + 31) // 32) + dag.dense(
        degrees
    )
    if (
        dag.ledger.work
        + product_work
        + scalar_work
        + normalization_work
        + conversion_work
        > 50_000_000
    ):
        return unique
    dag.ledger.charge("normalization", scalar_work)
    return (dag.polynomial(unique),)


def select_locus_factors(
    dag: Dag,
    guards: tuple[Expression, ...],
    determinant: Expression,
    undefined_factors: tuple[int, ...],
    output_denominator_count: int,
) -> tuple[int, ...]:
    """Choose a bounded conjunction, retaining zero only as a failure sentinel."""

    retained = set(undefined_factors) | set(determinant.numerator)
    if not determinant.scalar:
        retained.add(0)
    if len(retained - {0}) + output_denominator_count > 768:
        # Only compress when factor splitting would exceed the family cap.
        # Otherwise preserve the cheaper factored execution and dense boxes.
        for predicate in guards:
            for factors in (predicate.numerator, predicate.denominator):
                selected = set(factors)
                if len(selected) < 2 or not selected <= retained:
                    continue
                presentation = locus_presentation(dag, factors)
                if len(presentation) < len(selected):
                    retained.difference_update(selected)
                    retained.update(presentation)
                if len(retained - {0}) + output_denominator_count <= 768:
                    break
            if len(retained - {0}) + output_denominator_count <= 768:
                break
    if len(retained - {0}) + output_denominator_count > 768:
        dag.ledger.limits.reject("locus", "complete pullback locus exceeds 768 guards")
    return tuple(sorted(retained))
