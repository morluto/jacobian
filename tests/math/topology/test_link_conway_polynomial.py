"""Conway polynomial conversion from the exact knot Alexander value."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    ConwayPolynomialResult,
    OrientedLinkDiagram,
    braid_closure,
    link_conway_polynomial,
)
from jacobian.math.topology.links.extensions_tools import TOOLS


def _two_braid(*exponents: int) -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=tuple(
            BraidLetter(generator=1, exponent=exponent) for exponent in exponents
        ),
    )


def _figure_eight() -> OrientedLinkDiagram:
    word = BraidWord(
        strand_count=3,
        letters=(
            BraidLetter(generator=1, exponent=1),
            BraidLetter(generator=2, exponent=-1),
            BraidLetter(generator=1, exponent=1),
            BraidLetter(generator=2, exponent=-1),
        ),
    )
    return braid_closure(word).diagram


def _terms(result: ConwayPolynomialResult) -> dict[int, int]:
    return {term.exponents[0]: term.coefficient.num for term in result.polynomial.terms}


def test_trefoil_matches_independent_seifert_matrix_determinant() -> None:
    diagram = braid_closure(_two_braid(1, 1, 1)).diagram
    result = link_conway_polynomial(diagram)

    # For V=[[1,0],[-1,1]], det(uV-u^-1 V^T)=(u-u^-1)^2+1.
    assert _terms(result) == {2: 1, 0: 1}
    assert result.alexander.polynomial.variables == ("t",)
    assert result.alexander.diagram == diagram
    assert (
        ConwayPolynomialResult.model_validate_json(result.model_dump_json()) == result
    )


def test_figure_eight_matches_independent_seifert_matrix_determinant() -> None:
    result = link_conway_polynomial(_figure_eight())

    # For V=[[1,0],[-1,-1]], det(uV-u^-1 V^T)=1-(u-u^-1)^2.
    assert _terms(result) == {2: -1, 0: 1}


def test_unknot_and_mirror_normalizations() -> None:
    unknot = link_conway_polynomial(OrientedLinkDiagram(free_loops=1))
    assert _terms(unknot) == {0: 1}

    trefoil = braid_closure(_two_braid(1, 1, 1)).diagram
    mirrored = trefoil.model_copy(
        update={
            "crossings": tuple(
                crossing.model_copy(
                    update={
                        "over_pair": crossing.under_pair,
                        "under_pair": crossing.over_pair,
                        "sign": -crossing.sign,
                    }
                )
                for crossing in trefoil.crossings
            )
        }
    )
    assert _terms(link_conway_polynomial(mirrored)) == {2: 1, 0: 1}


def test_conway_is_knot_only_and_keeps_alexander_bound() -> None:
    hopf = braid_closure(_two_braid(1, 1)).diagram
    with pytest.raises(OperationDomainValidationError, match="exactly one component"):
        link_conway_polynomial(hopf)

    over_bound_knot = braid_closure(_two_braid(*([1] * 9))).diagram
    with pytest.raises(OperationResourceAdmissionError, match="eight crossings"):
        link_conway_polynomial(over_bound_knot)


def test_result_rejects_non_conway_exponent_lattice() -> None:
    result = link_conway_polynomial(OrientedLinkDiagram(free_loops=1))
    payload = result.model_dump(mode="json")
    payload["polynomial"]["terms"] = [
        {
            "coefficient": {"num": "1", "den": "1"},
            "exponents": [1],
        },
        {
            "coefficient": {"num": "1", "den": "1"},
            "exponents": [0],
        },
    ]
    with pytest.raises(ValidationError, match="nonnegative even exponents"):
        ConwayPolynomialResult.model_validate_json(json.dumps(payload))


def test_conway_operation_is_published_in_link_manifest() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "link_diagram.conway_polynomial.compute"
    )
    result = tool.run(
        tool.request_type.model_validate({"diagram": OrientedLinkDiagram(free_loops=1)})
    )
    assert _terms(result) == {0: 1}
