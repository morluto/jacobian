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
    return ElementDeltaSetResult(
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
    return ElementElasticityResult(
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
    return ElementCatenaryDegreeResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        factorization_count=len(family),
        catenary_degree=catenary_degree_from_factorizations(family),
    )


__all__ = [
    "compute_element_catenary_degree",
    "compute_element_delta_set",
    "compute_element_elasticity",
]
