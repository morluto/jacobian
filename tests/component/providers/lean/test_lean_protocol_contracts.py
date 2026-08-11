from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.lean_metavariable_fields import LeanElaborationContext
from jacobian.lean_frontend.declaration_protocol import (
    LeanDeclarationResultEnvelope,
    LeanDeclarationSearchQuery,
)
from jacobian.lean_frontend.helper_protocol import LeanTypedGoalsHelperResult
from jacobian.lean_frontend.repl_protocol import LeanReplProofStepResponse


@pytest.mark.parametrize(
    "response",
    (
        {"proofState": 1, "goals": []},
        {"proofState": True, "goals": [], "proofStatus": "Completed"},
        {
            "proofState": 1,
            "goals": [],
            "proofStatus": "Completed",
            "unexpected": "field",
        },
    ),
)
def test_repl_proof_response_rejects_incomplete_or_coerced_shapes(
    response: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        LeanReplProofStepResponse.model_validate(response)


def test_declaration_search_catalog_requires_bound_scan_positions() -> None:
    with pytest.raises(ValidationError, match="scanned total"):
        LeanDeclarationSearchQuery(
            name_contains="Nat",
            type_constants=(),
            namespace_prefixes=(),
            target_module_prefixes=("Init",),
            kinds=(),
            limit=10,
            candidate_names=("Nat.add",),
            candidate_scan_positions=(1,),
        )


def test_declaration_search_catalog_requires_unique_candidates() -> None:
    with pytest.raises(ValidationError, match="candidate names must be unique"):
        LeanDeclarationSearchQuery(
            name_contains="Nat",
            type_constants=(),
            namespace_prefixes=(),
            target_module_prefixes=("Init",),
            kinds=(),
            limit=10,
            candidate_names=("Nat.add", "Nat.add"),
            candidate_scan_positions=(1, 2),
            scanned_declarations_total=2,
        )


def test_declaration_result_discriminator_rejects_cross_operation_payload() -> None:
    with pytest.raises(ValidationError):
        LeanDeclarationResultEnvelope.model_validate(
            {
                "request_id": "request",
                "payload": {
                    "operation": "inspect",
                    "declarations": [],
                    "scanned_declarations": 0,
                    "stop_reason": "EXHAUSTED",
                },
            }
        )


def test_typed_goal_result_rejects_metavariable_payload() -> None:
    with pytest.raises(ValidationError):
        LeanTypedGoalsHelperResult.model_validate(
            {
                "request_id": "request",
                "payload": {
                    "expression_serialization": "LEAN_PRETTY_PRINTED_EXPR",
                    "structured_metavariables": [],
                    "elaboration_context": {},
                    "coercion_provenance": "UNAVAILABLE",
                    "coercion_provenance_basis": "not retained",
                },
            }
        )


def test_lean_helper_boolean_fields_do_not_coerce_strings() -> None:
    with pytest.raises(ValidationError):
        LeanElaborationContext.model_validate(
            {
                "decl_name": "",
                "may_postpone": "false",
                "err_to_sorry": False,
                "auto_bound_implicit": False,
                "implicit_lambda": False,
                "is_noncomputable_section": False,
                "ignore_tc_failures": False,
                "in_pattern": False,
                "save_rec_app_syntax": False,
                "holes_as_synthetic_opaque": False,
            }
        )
