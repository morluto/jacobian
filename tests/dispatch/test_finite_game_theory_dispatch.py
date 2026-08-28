"""Published-envelope checks for finite game-theory dispatch."""

import pytest
from jsonschema.validators import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.games.finite._models import (
    MAX_EXACT_EQUILIBRIUM_WORK,
    ZeroSumGameRequest,
)
from jacobian.math.logic.games.finite._tools import TOOLS


def _small_game() -> dict[str, object]:
    return {
        "payoff_matrix": {
            "n_rows": 2,
            "n_cols": 2,
            "entries": [
                {"num": "3", "den": "1"},
                {"num": "0", "den": "1"},
                {"num": "0", "den": "1"},
                {"num": "2", "den": "1"},
            ],
        }
    }


def test_zero_sum_schema_publishes_coupled_admission_and_dispatches() -> None:
    """The catalog schema explains the exact envelope enforced by ``math.run``."""

    operation = Catalog.open().operation("game_theory.nash_equilibrium.compute")
    assert operation is not None
    schema = operation.request_type.model_json_schema()
    matrix_schema = schema["$defs"]["PayoffMatrix"]

    assert "n_rows * n_cols" in matrix_schema["properties"]["entries"]["description"]
    assert (
        "(max(n_rows, n_cols) + 2) * (sum of payoff denominator decimal digits "
        "+ maximum payoff numerator decimal digits)"
    ) in schema["description"]
    assert schema["x-jacobian-bounds"]["max_exact_equilibrium_work"] == (
        MAX_EXACT_EQUILIBRIUM_WORK
    )
    assert str(MAX_EXACT_EQUILIBRIUM_WORK) in operation.description

    payload = _small_game()
    assert not list(Draft202012Validator(schema).iter_errors(payload))
    output = invoke_operation(operation.operation_id, payload, Catalog.open()).output
    assert set(output) == {"row_strategy", "col_strategy", "value"}
    assert len(output["row_strategy"]) == 2
    assert len(output["col_strategy"]) == 2


def test_zero_sum_schema_explains_structurally_valid_exact_work_rejection() -> None:
    """A cross-field envelope remains discoverable before its typed rejection."""

    operation = Catalog.open().operation("game_theory.nash_equilibrium.compute")
    assert operation is not None
    schema = operation.request_type.model_json_schema()
    denominator = "1" * (MAX_EXACT_EQUILIBRIUM_WORK // 4 + 1)
    payload = {
        "payoff_matrix": {
            "n_rows": 2,
            "n_cols": 2,
            "entries": [{"num": "1", "den": denominator}] * 4,
        }
    }

    assert not list(Draft202012Validator(schema).iter_errors(payload))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        invoke_operation(operation.operation_id, payload, Catalog.open())
    assert exc_info.value.errors()[0]["type"] == "finite_game.exact_equilibrium_budget"


def test_best_response_declaration_publishes_an_operation_neutral_schema() -> None:
    """The exact-equilibrium envelope appears only on the Nash operation."""

    tools = {tool.operation_id: tool for tool in TOOLS}
    best_response_schema = ZeroSumGameRequest.model_json_schema()
    equilibrium_schema = tools[
        "game_theory.nash_equilibrium.compute"
    ].request_type.model_json_schema()
    assert "game_theory.best_response.compute" in tools

    assert "linear program" not in best_response_schema["description"]
    assert "x-jacobian-bounds" not in best_response_schema
    assert (
        "(max(n_rows, n_cols) + 2) * (sum of payoff denominator decimal digits "
        "+ maximum payoff numerator decimal digits)"
    ) not in best_response_schema["description"]
    assert str(MAX_EXACT_EQUILIBRIUM_WORK) not in str(best_response_schema)

    assert "linear program" in equilibrium_schema["description"]
    assert equilibrium_schema["x-jacobian-bounds"]["max_exact_equilibrium_work"] == (
        MAX_EXACT_EQUILIBRIUM_WORK
    )

    payload = _small_game()
    assert not list(Draft202012Validator(best_response_schema).iter_errors(payload))
