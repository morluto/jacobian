"""Native exact operations for affine maps of standard real tori."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.geometry.affine_tori._bounds import (
    begin_affine_torus_deadline,
    build_affine_torus_plan,
    require_affine_torus_deadline,
)
from jacobian.math.geometry.affine_tori._flint_process import compute_fixed_locus_kernel
from jacobian.math.geometry.affine_tori._kernel_types import (
    EmptyFixedLocusKernel,
    NonemptyFixedLocusKernel,
)
from jacobian.math.geometry.affine_tori._models import (
    AffineTorusFixedLocusResult,
    EmptyAffineTorusFixedLocus,
    NonemptyAffineTorusFixedLocus,
)
from jacobian.math.geometry.affine_tori.values import (
    ConnectedSubtorusParameterization,
    FiniteTorusComponentPresentation,
    IntegralTorusCharacter,
    RationalAffineTorusMap,
    RationalTorusCosetFamily,
    RationalTorusPoint,
)
from jacobian.math.matrices.values import IntegerMatrix


def verify_integral_torus_character(character: IntegralTorusCharacter) -> bool:
    """Verify the primitive-domain claim of a source-bound character."""
    try:
        values = tuple(
            parse_canonical_integer(value) for value in character.coefficients
        )
        if len(values) != character.torus.dimension:
            return False
        divisor = 0
        for value in values:
            divisor = gcd(divisor, abs(value))
        return divisor == 1
    except (TypeError, ValueError):
        return False


def _integer_matrix(
    entries: tuple[tuple[int, ...], ...], *, rows: int, columns: int
) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(
            tuple(format_canonical_integer(value) for value in row) for row in entries
        ),
    )


def _point(
    source: RationalAffineTorusMap, coordinates: tuple[Fraction, ...]
) -> RationalTorusPoint:
    return RationalTorusPoint(
        torus=source.torus,
        coordinates=tuple(
            CanonicalRational.from_fraction(coordinate) for coordinate in coordinates
        ),
    )


def _nonempty_result(
    source: RationalAffineTorusMap, kernel: NonemptyFixedLocusKernel
) -> AffineTorusFixedLocusResult:
    dimension = source.torus.dimension
    identity_dimension = (
        len(kernel.identity_embedding[0]) if kernel.identity_embedding else 0
    )
    generator_count = len(kernel.component_generators)
    family = RationalTorusCosetFamily.model_construct(
        ambient_torus=source.torus,
        base_point=_point(source, kernel.base_point),
        identity_component=ConnectedSubtorusParameterization.model_construct(
            ambient_torus=source.torus,
            parameter_dimension=identity_dimension,
            embedding=_integer_matrix(
                kernel.identity_embedding,
                rows=dimension,
                columns=identity_dimension,
            ),
        ),
        component_generators=tuple(
            _point(source, coordinates) for coordinates in kernel.component_generators
        ),
        finite_components=FiniteTorusComponentPresentation.model_construct(
            generator_count=generator_count,
            relation_matrix=_integer_matrix(
                kernel.relation_matrix,
                rows=generator_count,
                columns=generator_count,
            ),
            generator_orders=tuple(
                format_canonical_integer(value) for value in kernel.generator_orders
            ),
            invariant_factors=tuple(
                format_canonical_integer(value) for value in kernel.invariant_factors
            ),
            component_count=format_canonical_integer(kernel.component_count),
        ),
    )
    return AffineTorusFixedLocusResult(
        source=source,
        outcome=NonemptyAffineTorusFixedLocus(fixed_locus=family),
    )


def _empty_result(
    source: RationalAffineTorusMap, kernel: EmptyFixedLocusKernel
) -> AffineTorusFixedLocusResult:
    character = tuple(format_canonical_integer(value) for value in kernel.character)
    candidate = IntegralTorusCharacter.model_construct(
        torus=source.torus,
        coefficients=character,
    )
    if not verify_integral_torus_character(candidate):
        raise RuntimeError("fixed-locus kernel returned a nonprimitive character")
    return AffineTorusFixedLocusResult(
        source=source,
        outcome=EmptyAffineTorusFixedLocus(
            obstruction=candidate,
            obstruction_pairing=CanonicalRational.from_fraction(kernel.pairing),
        ),
    )


def affine_torus_fixed_locus(
    source: RationalAffineTorusMap,
) -> AffineTorusFixedLocusResult:
    """Return the exact fixed locus of ``x |-> A x + b`` on ``R^n/Z^n``.

    A nonempty result presents all components as a base point, one primitive
    connected subtorus, and the finite group ``Z^r/CZ^r`` of rational
    translates.  An empty result returns a primitive invariant character
    whose nonzero pairing with ``b`` proves inconsistency.
    """

    deadline = begin_affine_torus_deadline()
    plan = build_affine_torus_plan(source, deadline=deadline)
    kernel = compute_fixed_locus_kernel(source, plan)
    require_affine_torus_deadline(deadline, "before result construction")
    result = (
        _empty_result(source, kernel)
        if isinstance(kernel, EmptyFixedLocusKernel)
        else _nonempty_result(source, kernel)
    )
    # Dispatch owns the one canonical serialization. The source-derived plan
    # has already proved this result shape fits that concrete transport limit.
    require_affine_torus_deadline(deadline, "before result handoff")
    return result


__all__ = ["affine_torus_fixed_locus", "verify_integral_torus_character"]
