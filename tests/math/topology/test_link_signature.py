"""Exact Goeritz signature and correction for bounded classical diagrams."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    LinkSignatureResult,
    OrientedLinkDiagram,
    braid_closure,
    link_signature,
)
from jacobian.math.topology.links.extensions_tools import TOOLS


def _braid(strands: int, letters: tuple[int, ...]) -> OrientedLinkDiagram:
    return braid_closure(
        BraidWord(
            strand_count=strands,
            letters=tuple(
                BraidLetter(generator=abs(letter), exponent=1 if letter > 0 else -1)
                for letter in letters
            ),
        )
    ).diagram


def _two_by_two_signature(matrix: tuple[tuple[int, int], tuple[int, int]]) -> int:
    """Independent Sylvester sign check for tiny symmetrized Seifert forms."""
    trace = matrix[0][0] + matrix[1][1]
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if determinant < 0:
        return 0
    assert determinant > 0 and trace != 0
    return 2 if trace > 0 else -2


@pytest.mark.parametrize(
    ("letters", "seifert_matrix"),
    [
        ((1, 1, 1), ((1, 0), (-1, 1))),
        ((-1, -1, -1), ((-1, 0), (1, -1))),
        ((1, -2, 1, -2), ((1, 0), (-1, -1))),
    ],
)
def test_goeritz_signature_matches_independent_seifert_form(
    letters: tuple[int, ...],
    seifert_matrix: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    diagram = _braid(2 if len(letters) == 3 else 3, letters)
    result = link_signature(diagram)
    symmetric_seifert_form = tuple(
        tuple(
            seifert_matrix[row][column] + seifert_matrix[column][row]
            for column in range(2)
        )
        for row in range(2)
    )

    assert result.signature == _two_by_two_signature(symmetric_seifert_form)
    assert result.goeritz_data.blackboard_graph.diagram == diagram
    assert (
        result.goeritz_inertia.matrix.row_count
        == result.goeritz_data.reduced_matrix.row_count
    )
    assert result.signature == (
        result.goeritz_inertia.n_positive
        - result.goeritz_inertia.n_negative
        - result.correction_term
    )
    assert LinkSignatureResult.model_validate_json(result.model_dump_json()) == result


def test_type_ii_correction_is_source_bound_and_changes_figure_eight_signature() -> (
    None
):
    result = link_signature(_braid(3, (1, -2, 1, -2)))

    assert result.signature == 0
    assert result.goeritz_inertia.n_positive == 2
    assert result.goeritz_inertia.n_negative == 0
    assert result.correction_term == 2
    assert tuple(row.crossing_type for row in result.correction_contributions) == (
        "TYPE_I",
        "TYPE_II",
        "TYPE_I",
        "TYPE_II",
    )
    assert tuple(
        row.correction_contribution for row in result.correction_contributions
    ) == (0, 1, 0, 1)

    payload = result.model_dump(mode="python")
    payload["correction_term"] = 0
    with pytest.raises(ValidationError, match="correction term must sum"):
        LinkSignatureResult.model_validate(payload)


def test_signature_requires_its_bounded_goeritz_projection() -> None:
    unlink = link_signature(OrientedLinkDiagram(free_loops=2))
    assert unlink.signature == 0
    assert unlink.goeritz_data is None
    assert unlink.correction_contributions == ()

    at_bound = link_signature(_braid(2, (1,) * 32))
    assert len(at_bound.correction_contributions) == 32

    over_bound = _braid(2, (1,) * 33)
    with pytest.raises(OperationResourceAdmissionError, match="at most 32 crossings"):
        link_signature(over_bound)


def test_reidemeister_one_curl_uses_zero_dimensional_goeritz_inertia() -> None:
    curl = _braid(2, (1,))
    result = link_signature(curl)

    assert result.goeritz_data.reduced_matrix.entries == ()
    assert result.goeritz_inertia.n_positive == 0
    assert result.goeritz_inertia.n_negative == 0
    assert result.goeritz_inertia.n_zero == 0
    assert result.signature == 0


def test_signature_is_published_as_one_typed_link_operation() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "link_diagram.signature.compute"
    )
    assert tool.request_type.__name__ == "LinkSignatureRequest"
    assert tool.result_type is LinkSignatureResult
    assert tool.examples
