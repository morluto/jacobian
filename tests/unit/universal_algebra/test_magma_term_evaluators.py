from __future__ import annotations

import pytest

from jacobian.contracts.universal_algebra import MagmaTerm
from jacobian.universal_algebra_capabilities import (
    _evaluate_term,
    _z3_evaluate_term,
)


def test_magma_evaluators_reject_malformed_constructed_terms() -> None:
    missing_variable = MagmaTerm.model_construct(kind="VARIABLE")
    missing_children = MagmaTerm.model_construct(kind="PRODUCT")

    with pytest.raises(ValueError, match="only a variable name"):
        _evaluate_term(missing_variable, (), {})
    with pytest.raises(ValueError, match="exactly two child terms"):
        _evaluate_term(missing_children, (), {})
    with pytest.raises(ValueError, match="only a variable name"):
        _z3_evaluate_term(missing_variable, (), {}, 0, None)
    with pytest.raises(ValueError, match="exactly two child terms"):
        _z3_evaluate_term(missing_children, (), {}, 0, None)
