"""Exact public API contract for jacobian.math.formal_concept_analysis."""

from __future__ import annotations

from jacobian.math import formal_concept_analysis


def test_exact_public_api_symbols() -> None:
    expected = (
        "AttributeImplication",
        "FiniteAttributeImplicationSystem",
        "FormalContext",
        "ImplicationClosureResult",
        "ImplicationClosureWork",
        "ImplicationDerivation",
        "attribute_closure",
        "attribute_derivation",
        "concept_from_attributes",
        "concept_from_objects",
        "concept_lattice",
        "enumerate_concepts",
        "implication_closure",
        "object_closure",
        "object_derivation",
    )
    assert tuple(formal_concept_analysis.__all__) == expected
    assert len(formal_concept_analysis.__all__) == len(
        set(formal_concept_analysis.__all__)
    )
    assert all(not name.startswith("_") for name in formal_concept_analysis.__all__)
    assert all(
        hasattr(formal_concept_analysis, name)
        for name in formal_concept_analysis.__all__
    )
