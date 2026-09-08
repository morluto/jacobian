"""Coloring witnesses need only their relation, while noncolorability needs search."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring import (
    _models,
    operations,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def _cycle(n: int) -> FiniteHypergraph:
    return FiniteHypergraph(
        vertices=tuple(str(i) for i in range(n)),
        edges=tuple((str(i), (str(i), str((i + 1) % n))) for i in range(n)),
    )


def test_coloring_witness_does_not_require_exhaustive_search_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = operations.decide_nonmonochromatic_coloring(_cycle(4), 2)
    monkeypatch.setattr(_models, "MAX_COLORING_WORK", 0)
    assert operations.verify_coloring_witness(claim)


def test_noncolorability_resource_refusal_is_not_a_false_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = operations.decide_nonmonochromatic_coloring(_cycle(3), 2)
    assert operations.verify_non_colorable(claim)
    monkeypatch.setattr(_models, "MAX_COLORING_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError):
        operations.verify_non_colorable(claim)


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_noncolorability_verifier_preserves_failed_search(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    claim = operations.decide_nonmonochromatic_coloring(_cycle(3), 2)

    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("backend failure")

    monkeypatch.setattr(operations, "decide_nonmonochromatic_coloring", fail)
    with pytest.raises(error_type, match="backend failure"):
        operations.verify_non_colorable(claim)
