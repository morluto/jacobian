"""Element invariant operations for numerical semigroups."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian._exact import format_canonical_rational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    catenary_degree_from_factorizations,
    factorization_length_extrema,
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)
from jacobian.math.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)


def _minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    """Normalize an admitted presentation to its minimal atom axis."""

    return minimal_generating_system(
        tuple(sorted({parse_canonical_integer(generator) for generator in generators}))
    )


def compute_element_delta_set(
    request: ElementDeltaSetRequest,
) -> ElementDeltaSetResult:
    """Compute the complete factorization-length delta set of one element."""

    atoms = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    lengths = factorization_lengths(atoms, value)
    return ElementDeltaSetResult._from_kernel(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        factorization_lengths=lengths,
        delta_set=tuple(sorted({right - left for left, right in pairwise(lengths)})),
    )


def compute_element_elasticity(
    request: ElementElasticityRequest,
) -> ElementElasticityResult:
    """Compute the exact factorization-length elasticity of one element."""

    atoms = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    minimum_length, maximum_length = factorization_length_extrema(atoms, value)
    return ElementElasticityResult._from_kernel(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        minimum_length=minimum_length,
        maximum_length=maximum_length,
        elasticity=format_canonical_rational(Fraction(maximum_length, minimum_length)),
    )


def compute_element_catenary_degree(
    request: ElementCatenaryDegreeRequest,
) -> ElementCatenaryDegreeResult:
    """Compute the catenary degree from the complete factorization family."""

    atoms = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    family = factorizations(atoms, value)
    return ElementCatenaryDegreeResult._from_kernel(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        factorization_count=len(family),
        catenary_degree=catenary_degree_from_factorizations(family),
    )


def verify_element_catenary_degree_result(
    result: ElementCatenaryDegreeResult,
) -> bool:
    """Replay the admitted complete factorization family for a supplied claim."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    family = factorizations(atoms, parse_canonical_integer(result.value))
    return result.factorization_count == len(
        family
    ) and result.catenary_degree == catenary_degree_from_factorizations(family)


def verify_element_delta_set_result(result: ElementDeltaSetResult) -> bool:
    """Replay one supplied element delta-set claim within its admitted bounds."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    lengths = factorization_lengths(atoms, parse_canonical_integer(result.value))
    delta_set = tuple(sorted({right - left for left, right in pairwise(lengths)}))
    return result.factorization_lengths == lengths and result.delta_set == delta_set


def verify_element_elasticity_result(result: ElementElasticityResult) -> bool:
    """Replay one supplied element elasticity claim within its admitted bounds."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    minimum, maximum = factorization_length_extrema(
        atoms, parse_canonical_integer(result.value)
    )
    return (result.minimum_length, result.maximum_length) == (
        minimum,
        maximum,
    ) and result.elasticity == format_canonical_rational(Fraction(maximum, minimum))


__all__ = [
    "compute_element_catenary_degree",
    "compute_element_delta_set",
    "compute_element_elasticity",
    "verify_element_delta_set_result",
    "verify_element_elasticity_result",
]
