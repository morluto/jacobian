"""Exact Stanley-Reisner ideals and explicit source-variable transport."""

from __future__ import annotations

from jacobian.math.polynomials.ideals import monomial_ideal_graded_betti_table
from jacobian.math.topology._simplicial_kernel import canonicalize
from jacobian.math.topology._structural import (
    StanleyReisnerIdealRequest,
    StanleyReisnerIdealResult,
    compute_stanley_reisner_ideal,
)


def _complex(vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]):
    return canonicalize(vertices, facets).complex


def test_triangle_boundary_ideal_has_exact_generator_and_composes_to_betti() -> None:
    source = _complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    result = compute_stanley_reisner_ideal(StanleyReisnerIdealRequest(complex=source))

    assert result.source == source
    assert tuple((item.vertex, item.variable) for item in result.vertex_variables) == (
        ("a", "v0"),
        ("b", "v1"),
        ("c", "v2"),
    )
    assert result.ideal.variables == ("v0", "v1", "v2")
    assert tuple(
        term.exponents
        for generator in result.ideal.generators
        for term in generator.polynomial.terms
    ) == ((1, 1, 1),)

    betti = monomial_ideal_graded_betti_table(result.ideal)
    assert betti.ideal == result.ideal
    assert betti.regularity == 3
    assert tuple(
        (entry.homological_degree, entry.internal_degree, entry.value)
        for entry in betti.graded_betti_numbers
    ) == ((0, 3, 1),)


def test_awkward_vertex_labels_keep_an_injective_explicit_binding() -> None:
    source = _complex(("a-b", "a:b", "C"), (("a-b", "a:b"), ("a-b", "C"), ("a:b", "C")))
    result = compute_stanley_reisner_ideal(StanleyReisnerIdealRequest(complex=source))

    assert tuple(item.vertex for item in result.vertex_variables) == source.vertices
    assert len({item.variable for item in result.vertex_variables}) == len(
        source.vertices
    )
    assert result.ideal.generators[0].polynomial.terms[0].exponents == (1, 1, 1)


def test_simplex_uses_the_existing_singleton_zero_ideal_and_round_trips() -> None:
    source = _complex(("a", "b", "c"), (("a", "b", "c"),))
    result = compute_stanley_reisner_ideal(StanleyReisnerIdealRequest(complex=source))

    assert len(result.ideal.generators) == 1
    assert result.ideal.generators[0].polynomial.terms == ()
    assert (
        StanleyReisnerIdealResult.model_validate_json(result.model_dump_json())
        == result
    )
    betti = monomial_ideal_graded_betti_table(result.ideal)
    assert betti.lcm_lattice_homology == ()
    assert betti.multigraded_betti_numbers == ()
    assert betti.graded_betti_numbers == ()
    assert betti.regularity is None
    assert not betti.has_linear_resolution
    assert type(betti).model_validate_json(betti.model_dump_json()) == betti
