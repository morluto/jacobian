"""Element invariant operations for numerical semigroups."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian._exact import format_canonical_rational
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.numerical_semigroups._algorithms import (
    catenary_degree_from_factorizations,
    factorization_length_extrema,
    factorization_lengths,
    factorizations,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._models import (
    MAX_GRAPH_FACTORIZATIONS,
    _require_bounded_value,
    _require_materializable_factorizations,
    _require_member,
    _require_minimal_generators,
    _run_admission,
)


def compute_element_delta_set(
    request: ElementDeltaSetRequest,
) -> ElementDeltaSetResult:
    """Compute the complete factorization-length delta set of one element."""

    atoms = _run_admission(
        "element_delta_set",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    value = _run_admission(
        "element_delta_set",
        ("value",),
        lambda: _require_bounded_value(atoms, request.value),
    )
    _run_admission(
        "element_delta_set", ("value",), lambda: _require_member(atoms, value)
    )
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

    atoms = _run_admission(
        "element_elasticity",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    value = _run_admission(
        "element_elasticity",
        ("value",),
        lambda: _require_bounded_value(atoms, request.value),
    )

    def admit_element() -> None:
        if value <= 0:
            raise ValueError("elasticity is defined here only for positive elements")
        _require_member(atoms, value)

    _run_admission("element_elasticity", ("value",), admit_element)
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

    atoms = _run_admission(
        "element_catenary_degree",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    value = _run_admission(
        "element_catenary_degree",
        ("value",),
        lambda: _require_bounded_value(atoms, request.value),
    )
    _run_admission(
        "element_catenary_degree",
        ("value",),
        lambda: _require_member(atoms, value),
    )
    _run_admission(
        "element_catenary_degree",
        ("value",),
        lambda: _require_materializable_factorizations(
            atoms, value, MAX_GRAPH_FACTORIZATIONS
        ),
    )
    family = factorizations(atoms, value)
    return ElementCatenaryDegreeResult._from_kernel(
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
