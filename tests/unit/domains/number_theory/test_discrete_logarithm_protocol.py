from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.domains.number_theory.discrete_logarithm_protocol import (
    PROTOCOL,
    DiscreteLogarithmWorkerResult,
)


def test_discrete_logarithm_worker_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DiscreteLogarithmWorkerResult.model_validate(
            {
                "protocol": PROTOCOL,
                "result": {
                    "status": "SOLVED",
                    "base": 2,
                    "target": 1,
                    "modulus": 3,
                    "discrete_log": 0,
                },
                "diagnostic": "untrusted",
            }
        )


def test_discrete_logarithm_worker_result_rejects_wrong_protocol() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        DiscreteLogarithmWorkerResult.model_validate(
            {
                "protocol": "different.protocol/v1",
                "result": {
                    "status": "UNSOLVABLE",
                    "base": 2,
                    "target": 3,
                    "modulus": 8,
                },
            }
        )
