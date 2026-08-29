from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
    BinaryCodeDistanceGraphRequest,
)
from jacobian.math.combinatorics.codes.nonlinear.distance_graph.operations import (
    compute_distance_graph,
)
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode


def _code(length, codewords):
    return ExplicitBinaryCode(
        length=length, codewords=tuple(tuple(c) for c in codewords)
    )


def test_basic_distances() -> None:
    """Code {000, 011, 110}: all pairs at distance 2."""
    code = _code(3, [[0, 0, 0], [0, 1, 1], [1, 1, 0]])
    result_2 = compute_distance_graph(code, 2)
    assert result_2.edge_count == 3
    result_1 = compute_distance_graph(code, 1)
    assert result_1.edge_count == 0
    result_3 = compute_distance_graph(code, 3)
    assert result_3.edge_count == 0


def test_distance_1() -> None:
    """Code {000, 100, 010}: 000 vs 100 = 1, 000 vs 010 = 1, 100 vs 010 = 2."""
    code = _code(3, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    result = compute_distance_graph(code, 1)
    assert result.edge_count == 2
    edges = set(result.graph.edges)
    assert (0, 1) in edges
    assert (0, 2) in edges


def test_distance_0_distinct_words() -> None:
    """Distance 0 on distinct words: no edges."""
    code = _code(2, [[0, 0], [1, 1]])
    result = compute_distance_graph(code, 0)
    assert result.edge_count == 0


def test_vertex_count() -> None:
    """Graph has one vertex per codeword."""
    code = _code(3, [[0, 0, 0], [0, 1, 1], [1, 1, 0]])
    result = compute_distance_graph(code, 1)
    assert result.graph.vertex_count == 3


def test_replay_hamming() -> None:
    """Every edge connects codewords at the claimed Hamming distance."""
    code = _code(4, [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 1]])
    result = compute_distance_graph(code, 2)
    for i, j in result.graph.edges:
        w_i = code.codewords[i]
        w_j = code.codewords[j]
        dist = sum(1 for a, b in zip(w_i, w_j, strict=True) if a != b)
        assert dist == 2


def test_empty_code() -> None:
    """Empty code has zero vertices."""
    code = ExplicitBinaryCode(length=0, codewords=())
    result = compute_distance_graph(code, 0)
    assert result.graph.vertex_count == 0
    assert result.edge_count == 0


def test_rejects_distance_exceeds_length() -> None:
    code = _code(2, [[0, 0], [1, 1]])
    with pytest.raises(ValidationError):
        BinaryCodeDistanceGraphRequest(source=code, target_distance=3)


def test_request_schema_inlines_binary_word_definition() -> None:
    schema = BinaryCodeDistanceGraphRequest.model_json_schema()
    codewords = schema["properties"]["source"]["properties"]["codewords"]
    assert "$defs" not in schema
    assert codewords["items"]["type"] == "array"
    assert codewords["maxItems"] == 256


@pytest.mark.parametrize("distance", [True, 1.0])
def test_request_rejects_non_strict_target_distance(distance: object) -> None:
    code = _code(2, [[0, 0], [1, 1]])
    with pytest.raises(ValidationError):
        BinaryCodeDistanceGraphRequest(source=code, target_distance=distance)


def test_native_admission_rejects_invalid_distance() -> None:
    code = _code(2, [[0, 0], [1, 1]])
    with pytest.raises(OperationDomainValidationError) as error:
        compute_distance_graph(code, -1)
    assert error.value.errors()[0]["type"] == "code.distance_must_be_nonnegative"


def test_result_rejects_inconsistent_edge_count() -> None:
    code = _code(2, [[0, 0], [1, 1]])
    result = compute_distance_graph(code, 2)
    payload = result.model_dump(mode="json")
    payload["edge_count"] = 0
    from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
        BinaryCodeDistanceGraphResult,
    )

    with pytest.raises(ValidationError, match="edge_count"):
        BinaryCodeDistanceGraphResult.model_validate(payload)


def test_result_preserves_source() -> None:
    code = _code(2, [[0, 0], [1, 1]])
    result = compute_distance_graph(code, 1)
    assert result.source == code
    assert result.target_distance == 1
