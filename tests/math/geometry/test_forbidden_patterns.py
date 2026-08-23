"""Source-binding tests for forbidden-pattern screening results."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
)
from jacobian.math.geometry._operations import forbidden_patterns
from jacobian.math.geometry._profiles import PROFILE_OPERATIONS


def _point(label, num_x, den_x, num_y, den_y):
    return {
        "label": label,
        "point": {
            "x": {"num": num_x, "den": den_x},
            "y": {"num": num_y, "den": den_y},
        },
    }


def _config(*points):
    return {"configuration": {"points": list(points)}}


COLLINEAR_CONFIG = _config(
    _point("A", "0", "1", "0", "1"),
    _point("B", "1", "1", "0", "1"),
    _point("C", "2", "1", "0", "1"),
)

CONCYCLIC_CONFIG = _config(
    _point("A", "0", "1", "0", "1"),
    _point("B", "1", "1", "0", "1"),
    _point("C", "0", "1", "1", "1"),
    _point("D", "1", "1", "1", "1"),
)

CLEAN_CONFIG = _config(
    _point("A", "0", "1", "0", "1"),
    _point("B", "1", "1", "0", "1"),
    _point("C", "0", "1", "1", "1"),
    _point("D", "2", "1", "3", "1"),
)


class TestKnownAnswers:
    def test_collinear_triple_witness(self):
        request = ForbiddenPatternsRequest.model_validate(COLLINEAR_CONFIG)
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)
        assert result.checked_triples == 1

    def test_unit_square_is_concyclic(self):
        request = ForbiddenPatternsRequest.model_validate(CONCYCLIC_CONFIG)
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is True
        assert (
            result.concyclic_quadruple.first,
            result.concyclic_quadruple.second,
            result.concyclic_quadruple.third,
            result.concyclic_quadruple.fourth,
        ) == (0, 1, 2, 3)

    def test_clean_configuration_exhausts_enumeration(self):
        request = ForbiddenPatternsRequest.model_validate(CLEAN_CONFIG)
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        from itertools import combinations

        n = len(request.configuration.points)
        assert result.checked_triples == len(list(combinations(range(n), 3)))
        assert result.checked_quadruples == len(list(combinations(range(n), 4)))

    def test_collinear_quadruple_is_not_concyclic(self):
        four_collinear = _config(
            _point("A", "0", "1", "0", "1"),
            _point("B", "1", "1", "0", "1"),
            _point("C", "2", "1", "0", "1"),
            _point("D", "3", "1", "0", "1"),
        )
        request = ForbiddenPatternsRequest.model_validate(four_collinear)
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        assert result.has_concyclic_quadruple is False


class TestResultBinding:
    def test_forged_negative_decision_rejected(self):
        request = ForbiddenPatternsRequest.model_validate(COLLINEAR_CONFIG)
        payload = {
            "configuration": request.configuration.model_dump(),
            "point_count": 3,
            "has_collinear_triple": False,
            "has_concyclic_quadruple": False,
            "collinear_triple": None,
            "concyclic_quadruple": None,
            "checked_triples": 0,
            "checked_quadruples": 0,
        }
        with pytest.raises(ValidationError, match="collinear-triple flag"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_tampered_witness_indices_rejected(self):
        config = _config(
            _point("A", "0", "1", "0", "1"),
            _point("B", "1", "1", "0", "1"),
            _point("C", "2", "1", "0", "1"),
            _point("D", "5", "1", "7", "1"),
        )
        result = forbidden_patterns(ForbiddenPatternsRequest.model_validate(config))
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)
        payload = result.model_dump()
        payload["collinear_triple"] = {"first": 0, "second": 1, "third": 3}
        with pytest.raises(ValidationError, match="collinear witness"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_wrong_checked_counts_rejected(self):
        result = forbidden_patterns(
            ForbiddenPatternsRequest.model_validate(COLLINEAR_CONFIG)
        )
        payload = result.model_dump()
        payload["checked_triples"] = 0
        with pytest.raises(ValidationError, match="checked counts"):
            ForbiddenPatternsResult.model_validate(payload)


class TestToolsAndExamples:
    @pytest.mark.parametrize(
        "tool",
        [
            tool
            for tool in PROFILE_OPERATIONS
            if tool.operation_id.endswith("forbidden_patterns.check")
        ],
        ids=lambda tool: tool.operation_id,
    )
    def test_example_runs_and_reports_no_patterns(self, tool):
        for ex in tool.examples:
            request = tool.request_type.model_validate(ex.input)
            result = tool.run(request)
            assert result.has_collinear_triple is False
            assert result.has_concyclic_quadruple is False
