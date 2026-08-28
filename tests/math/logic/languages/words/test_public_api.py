"""Exact public API contract for jacobian.math.logic.languages.words."""

from __future__ import annotations

from jacobian.math.logic.languages import words


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the words public API."""
    expected = (
        "FactorAnalysis",
        "FiniteWord",
        "FixedPointPrefixAnalysis",
        "PeriodAnalysis",
        "PrimitivityAnalysis",
        "ProlongableSubstitution",
        "Substitution",
        "SubstitutionDependencyEdge",
        "SubstitutionDependencyGraph",
        "WordMorphism",
        "apply_morphism",
        "compose_morphisms",
        "conjugates",
        "factor_occurrences",
        "factors_of_length",
        "fixed_point_prefix",
        "incidence_matrix",
        "parikh_vector",
        "periods",
        "prefix_function",
        "primitive_root",
        "substitution_dependency_graph",
        "substitution_primitivity_profile",
    )
    assert tuple(words.__all__) == expected
    assert len(words.__all__) == len(set(words.__all__))
    assert all(not name.startswith("_") for name in words.__all__)
    assert all(hasattr(words, name) for name in words.__all__)
