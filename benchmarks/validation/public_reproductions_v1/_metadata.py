"""Task-local verifier contract metadata loading and validation.

Task-specific diagnostic behavior (input-binding decoupling, scope-assurance
independence) lives in each task's ``tests/verifier_contract.json`` rather than
global name registries so renames or removals cannot leave stale entries. This
module owns the closed schema validation and the predicate helpers consumed by
generic verifier tests.
"""

from __future__ import annotations

import json

from benchmarks.validation.public_reproductions_v1._paths import TASKS

_TASK_CONTRACT_KEYS = frozenset(
    {"schema_version", "input_binding_decoupled", "scope_independent_assurance"}
)


def load_task_contract_metadata(task_name: str) -> dict[str, object]:
    """Load task-local verifier contract metadata from the task's tests/ dir."""

    path = TASKS / task_name / "tests" / "verifier_contract.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{task_name}: invalid verifier_contract.json: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{task_name}: verifier_contract.json must be an object")
    unknown = set(value) - _TASK_CONTRACT_KEYS
    if unknown:
        raise ValueError(
            f"{task_name}: unknown verifier contract fields: {sorted(unknown)}"
        )
    if value.get("schema_version") != "1":
        raise ValueError(f"{task_name}: verifier contract schema_version must be '1'")
    for field in ("input_binding_decoupled", "scope_independent_assurance"):
        if field in value and type(value[field]) is not bool:
            raise ValueError(f"{task_name}: {field} must be a boolean")
    return value


def is_input_binding_decoupled(task_name: str) -> bool:
    """Whether correctness is reported independently of workspace input binding.

    Task-specific diagnostic behavior is declared solely in the task's
    ``tests/verifier_contract.json``. Missing metadata yields the generic
    (non-decoupled) behavior; malformed metadata fails closed in
    ``load_task_contract_metadata``.
    """

    metadata = load_task_contract_metadata(task_name)
    return metadata.get("input_binding_decoupled") is True


def is_scope_independent_assurance(task_name: str) -> bool:
    """Whether scope is reported independently of assurance typing.

    Task-specific diagnostic behavior is declared solely in the task's
    ``tests/verifier_contract.json``. Missing metadata yields the generic
    (scope-coupled) behavior; malformed metadata fails closed in
    ``load_task_contract_metadata``.
    """

    metadata = load_task_contract_metadata(task_name)
    return metadata.get("scope_independent_assurance") is True
