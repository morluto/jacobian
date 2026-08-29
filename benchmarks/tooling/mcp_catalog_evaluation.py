"""Deterministic catalog-scale controls for the direct MCP migration.

This evaluator proves only locally observable availability, schema, discovery,
execution, and composition facts.  Agent selection and deferred-loading claims
remain external evidence and are reported as unmeasured rather than inferred.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from tools.command_runner import git_head_sha

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.mcp.server import create_server
from mcp import Client

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUITE = _ROOT / "benchmarks/config/direct-mcp-catalog-evaluation-v1.json"
_FIXED_TOOLS = frozenset({"math.find", "math.run"})


class TaskCategory(StrEnum):
    """Frozen prompt families required by issue #2982."""

    STRAIGHTFORWARD = "STRAIGHTFORWARD"
    ALTERNATE_TERMINOLOGY = "ALTERNATE_TERMINOLOGY"
    POSTCONDITION_DISTINCTION = "POSTCONDITION_DISTINCTION"
    STRUCTURAL_AMBIGUITY = "STRUCTURAL_AMBIGUITY"
    MULTI_OPERATION = "MULTI_OPERATION"


class DiscoveryProbe(StrictModel):
    query: str = Field(min_length=1)
    namespace: str | None = None
    limit: int = Field(ge=1, le=20, strict=True)
    required_operation_ids: tuple[str, ...] = Field(min_length=1)
    maximum_rank: int = Field(ge=1, le=20, strict=True)


class EvaluationStep(StrictModel):
    step_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    operation_id: str = Field(min_length=3)
    input: dict[str, Any]
    expected_output_fields: dict[str, Any] = Field(min_length=1)


class EvaluationTask(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    category: TaskCategory
    prompt: str = Field(min_length=1)
    discovery_probes: tuple[DiscoveryProbe, ...] = Field(min_length=1)
    steps: tuple[EvaluationStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_coherent_steps(self) -> Self:
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("evaluation step IDs must be unique within a task")
        if self.category is TaskCategory.MULTI_OPERATION and len(self.steps) < 2:
            raise ValueError("multi-operation cases require at least two steps")
        return self


class SemanticDiscoveryCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    family: TaskCategory
    prompt: str = Field(min_length=1)
    query: str = Field(min_length=1)
    namespace: str | None = None
    limit: int = Field(ge=1, le=20, strict=True)
    required_operation_ids: tuple[str, ...] = Field(min_length=1)
    maximum_rank: int = Field(ge=1, le=20, strict=True)


class DecisionPolicy(StrictModel):
    minimum_external_repetitions_per_execution_case: int = Field(ge=2)
    require_complete_catalog_schema_coverage: Literal[True]
    require_all_local_execution_parity: Literal[True]
    require_all_local_compositions: Literal[True]
    require_observed_deferred_client_discovery: Literal[True]
    require_exact_loaded_definition_bytes: Literal[True]
    maximum_direct_to_legacy_loaded_definition_bytes_ratio: float = Field(
        gt=0.0, le=1.0
    )
    minimum_direct_end_to_end_success_rate_per_case: float = Field(ge=0.0, le=1.0)
    maximum_direct_minus_legacy_failure_rate_per_case: float = Field(ge=0.0, le=1.0)
    minimum_math_find_unique_improvement_families: int = Field(ge=1)


class CatalogEvaluationSuite(StrictModel):
    schema_version: Literal["1"]
    suite_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    question: str = Field(min_length=1)
    required_categories: tuple[TaskCategory, ...] = Field(min_length=1)
    decision_policy: DecisionPolicy
    tasks: tuple[EvaluationTask, ...] = Field(min_length=1)
    semantic_discovery_cases: tuple[SemanticDiscoveryCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_frozen_coverage(self) -> Self:
        case_ids = [task.case_id for task in self.tasks]
        case_ids.extend(case.case_id for case in self.semantic_discovery_cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case IDs must be unique")
        if set(self.required_categories) != set(TaskCategory):
            raise ValueError("required_categories must freeze every issue #2982 family")
        observed = {task.category for task in self.tasks}
        if observed != set(self.required_categories):
            raise ValueError("tasks must cover every required category")
        return self


def load_suite(path: Path) -> CatalogEvaluationSuite:
    """Load and fail closed on the complete frozen local evaluation contract."""

    return CatalogEvaluationSuite.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _field(value: object, path: str) -> tuple[bool, object]:
    current = value
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif (
            isinstance(current, Sequence)
            and not isinstance(current, str | bytes)
            and component.isdigit()
            and int(component) < len(current)
        ):
            current = current[int(component)]
        else:
            return False, None
    return True, current


def _expected_fields(output: object, expected: Mapping[str, Any]) -> bool:
    for path, expected_value in expected.items():
        present, observed = _field(output, path)
        if not present or observed != expected_value:
            return False
    return True


def _resolve_references(
    value: object, outputs: Mapping[str, Mapping[str, Any]]
) -> object:
    if isinstance(value, Mapping):
        if set(value) == {"$from_step", "path"}:
            step_id = value["$from_step"]
            path = value["path"]
            if not isinstance(step_id, str) or not isinstance(path, str):
                raise ValueError("step references require string step IDs and paths")
            if step_id not in outputs:
                raise ValueError(f"step reference is unavailable: {step_id}")
            present, selected = _field(outputs[step_id], path)
            if not present:
                raise ValueError(
                    f"step reference path is unavailable: {step_id}.{path}"
                )
            return selected
        return {key: _resolve_references(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, outputs) for item in value]
    return value


def _reference_count(value: object) -> int:
    if isinstance(value, Mapping):
        if set(value) == {"$from_step", "path"}:
            return 1
        return sum(_reference_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_reference_count(item) for item in value)
    return 0


def _percentile_95(samples: Sequence[float]) -> float:
    ordered = sorted(samples)
    index = max(0, int(len(ordered) * 0.95 + 0.999999) - 1)
    return ordered[index]


def _latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    rounded = [round(sample * 1_000, 6) for sample in samples]
    return {
        "samples_ms": rounded,
        "median_ms": round(statistics.median(samples) * 1_000, 6),
        "p95_ms": round(_percentile_95(samples) * 1_000, 6),
    }


def _tool_record(tool: Any) -> dict[str, Any]:
    record = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not isinstance(record, dict):
        raise TypeError("MCP tool model did not serialize to an object")
    return record


def _tool_bytes(records: Sequence[Mapping[str, Any]]) -> int:
    return len(_json_bytes(list(records)))


def _match_ids(payload: Mapping[str, Any]) -> list[str]:
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return []
    return [
        match["operation_id"]
        for match in matches
        if isinstance(match, Mapping) and isinstance(match.get("operation_id"), str)
    ]


async def _run_discovery(
    client: Client,
    *,
    query: str,
    namespace: str | None,
    limit: int,
    required_operation_ids: Sequence[str],
    maximum_rank: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {"op": "search", "query": query, "limit": limit}
    if namespace is not None:
        request["namespace"] = namespace
    started = time.perf_counter()
    result = await client.call_tool("math.find", {"request": request})
    elapsed = time.perf_counter() - started
    payload = result.structured_content
    if not isinstance(payload, dict):
        raise RuntimeError("math.find omitted structured discovery output")
    match_ids = _match_ids(payload)
    ranks = {
        operation_id: (
            match_ids.index(operation_id) + 1 if operation_id in match_ids else None
        )
        for operation_id in required_operation_ids
    }
    return {
        "query": query,
        "namespace": namespace,
        "limit": limit,
        "required_operation_ids": list(required_operation_ids),
        "match_ids": match_ids,
        "ranks": ranks,
        "maximum_rank": maximum_rank,
        "success": all(
            rank is not None and rank <= maximum_rank for rank in ranks.values()
        ),
        "elapsed_ms": round(elapsed * 1_000, 6),
        "response_bytes": len(_json_bytes(payload)),
    }


async def _call_direct(
    client: Client, operation_id: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = await client.call_tool(operation_id, payload)
    elapsed = time.perf_counter() - started
    output = result.structured_content
    if not isinstance(output, dict):
        raise RuntimeError(
            f"direct operation omitted structured output: {operation_id}"
        )
    return output, elapsed


async def _call_legacy(
    client: Client, operation_id: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = await client.call_tool(
        "math.run", {"operation_id": operation_id, "payload": payload}
    )
    elapsed = time.perf_counter() - started
    envelope = result.structured_content
    output = envelope.get("output") if isinstance(envelope, Mapping) else None
    if not isinstance(output, dict):
        raise RuntimeError(f"math.run omitted owner output: {operation_id}")
    return output, elapsed


async def _run_task(
    client: Client,
    task: EvaluationTask,
    tool_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    discovery = [
        await _run_discovery(
            client,
            query=probe.query,
            namespace=probe.namespace,
            limit=probe.limit,
            required_operation_ids=probe.required_operation_ids,
            maximum_rank=probe.maximum_rank,
        )
        for probe in task.discovery_probes
    ]
    direct_outputs: dict[str, dict[str, Any]] = {}
    legacy_outputs: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    for step in task.steps:
        direct_input = _resolve_references(step.input, direct_outputs)
        legacy_input = _resolve_references(step.input, legacy_outputs)
        if not isinstance(direct_input, dict) or not isinstance(legacy_input, dict):
            raise RuntimeError("MCP evaluation step inputs must have object roots")
        direct_output, direct_elapsed = await _call_direct(
            client, step.operation_id, direct_input
        )
        legacy_output, legacy_elapsed = await _call_legacy(
            client, step.operation_id, legacy_input
        )
        direct_outputs[step.step_id] = direct_output
        legacy_outputs[step.step_id] = legacy_output
        steps.append(
            {
                "step_id": step.step_id,
                "operation_id": step.operation_id,
                "reference_count": _reference_count(step.input),
                "direct": {
                    "success": _expected_fields(
                        direct_output, step.expected_output_fields
                    ),
                    "elapsed_ms": round(direct_elapsed * 1_000, 6),
                    "output_sha256": _sha256(_json_bytes(direct_output)),
                },
                "legacy": {
                    "success": _expected_fields(
                        legacy_output, step.expected_output_fields
                    ),
                    "elapsed_ms": round(legacy_elapsed * 1_000, 6),
                    "output_sha256": _sha256(_json_bytes(legacy_output)),
                },
                "exact_output_parity": _json_bytes(direct_output)
                == _json_bytes(legacy_output),
            }
        )
    selected_records = [tool_records[step.operation_id] for step in task.steps]
    direct_success = all(step["direct"]["success"] for step in steps)
    legacy_success = all(step["legacy"]["success"] for step in steps)
    parity = all(step["exact_output_parity"] for step in steps)
    references = sum(step["reference_count"] for step in steps)
    composition_success = task.category is not TaskCategory.MULTI_OPERATION or (
        direct_success and legacy_success and parity and references > 0
    )
    return {
        "case_id": task.case_id,
        "category": task.category,
        "prompt_sha256": _sha256(task.prompt.encode("utf-8")),
        "discovery": discovery,
        "steps": steps,
        "direct_success": direct_success,
        "legacy_success": legacy_success,
        "exact_output_parity": parity,
        "composition_success": composition_success,
        "call_counts": {
            "legacy_find_then_run": len(discovery) + len(steps),
            "direct_execution": len(steps),
            "semantic_find_only": len(discovery),
        },
        "required_direct_definition_bytes_estimate": _tool_bytes(selected_records),
    }


def _decision(
    suite: CatalogEvaluationSuite,
    *,
    schema_coverage: bool,
    tasks: Sequence[Mapping[str, Any]],
    semantic_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parity = all(task["exact_output_parity"] for task in tasks)
    composition = all(task["composition_success"] for task in tasks)
    local_execution = all(
        task["direct_success"] and task["legacy_success"] for task in tasks
    )
    semantic_reachability = all(case["success"] for case in semantic_cases)
    removal_gates = {
        "complete_catalog_schema_coverage": "PASS" if schema_coverage else "FAIL",
        "local_typed_execution_parity": "PASS"
        if parity and local_execution
        else "FAIL",
        "local_multi_operation_composition": "PASS" if composition else "FAIL",
        "repeated_real_client_noninferiority": "UNMEASURED",
        "deferred_client_discovery_observed": "UNMEASURED",
        "exact_per_task_loaded_definition_bytes": "UNMEASURED",
    }
    find_gates = {
        "local_semantic_vocabulary_reachability": "PASS"
        if semantic_reachability
        else "FAIL",
        "unique_improvement_over_client_tool_discovery": "UNMEASURED",
        "independent_improvement_family_count": 0,
        "required_improvement_family_count": (
            suite.decision_policy.minimum_math_find_unique_improvement_families
        ),
    }
    return {
        "math_run": {
            "gates": removal_gates,
            "removal_supported": all(
                value == "PASS" for value in removal_gates.values()
            ),
            "recommendation": "RETAIN_PENDING_EXTERNAL_CLIENT_EVIDENCE",
        },
        "math_find": {
            "gates": find_gates,
            "removal_supported": False,
            "recommendation": "RETAIN_AS_SEMANTIC_DISCOVERY_ONLY",
        },
    }


async def run_evaluation(
    suite: CatalogEvaluationSuite,
    *,
    list_repetitions: int,
) -> dict[str, Any]:
    """Run all locally observable controls against the production MCP server."""

    catalog = Catalog.open()
    catalog_snapshot = catalog.snapshot()
    catalog_ids = tuple(
        descriptor.operation_id for descriptor in catalog_snapshot.operations
    )
    construction_started = time.perf_counter()
    server = create_server()
    construction_elapsed = time.perf_counter() - construction_started
    client_started = time.perf_counter()
    async with Client(server, raise_exceptions=True) as client:
        client_elapsed = time.perf_counter() - client_started
        listings = []
        listing_latencies = []
        for _ in range(list_repetitions):
            started = time.perf_counter()
            listings.append(await client.list_tools())
            listing_latencies.append(time.perf_counter() - started)
        records = sorted(
            (_tool_record(tool) for tool in listings[0].tools),
            key=lambda record: record["name"],
        )
        if any(
            [_tool_record(tool) for tool in listing.tools]
            != [_tool_record(tool) for tool in listings[0].tools]
            for listing in listings[1:]
        ):
            raise RuntimeError("MCP tool listing changed during the frozen run")
        records_by_name = {record["name"]: record for record in records}
        advertised = frozenset(records_by_name)
        direct_ids = advertised - _FIXED_TOOLS
        schema_coverage = direct_ids == frozenset(catalog_ids)
        tasks = [await _run_task(client, task, records_by_name) for task in suite.tasks]
        semantic_cases = [
            {
                "case_id": case.case_id,
                "family": case.family,
                "prompt_sha256": _sha256(case.prompt.encode("utf-8")),
                **await _run_discovery(
                    client,
                    query=case.query,
                    namespace=case.namespace,
                    limit=case.limit,
                    required_operation_ids=case.required_operation_ids,
                    maximum_rank=case.maximum_rank,
                ),
            }
            for case in suite.semantic_discovery_cases
        ]

    direct_records = [records_by_name[operation_id] for operation_id in catalog_ids]
    fixed_records = [records_by_name[name] for name in sorted(_FIXED_TOOLS)]
    schema_bytes = sum(
        len(_json_bytes(record.get("inputSchema", {})))
        + len(_json_bytes(record.get("outputSchema", {})))
        for record in direct_records
    )
    suite_payload = suite.model_dump(mode="json")
    decision = _decision(
        suite,
        schema_coverage=schema_coverage,
        tasks=tasks,
        semantic_cases=semantic_cases,
    )
    return {
        "schema_version": "1",
        "suite": {
            "suite_id": suite.suite_id,
            "digest": _sha256(_json_bytes(suite_payload)),
            "question": suite.question,
            "task_count": len(suite.tasks),
            "semantic_discovery_case_count": len(suite.semantic_discovery_cases),
            "decision_policy": suite.decision_policy.model_dump(mode="json"),
        },
        "environment": {
            "repository_revision": git_head_sha(_ROOT),
            "evaluator_sha256": _sha256(Path(__file__).read_bytes()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mcp_sdk": importlib.metadata.version("mcp"),
        },
        "capability_boundary": {
            "ordinary_mcp_tool_listing": "OBSERVED",
            "typed_direct_invocation": "OBSERVED",
            "generic_find_then_run": "OBSERVED",
            "semantic_math_find": "OBSERVED",
            "deferred_client_tool_search": "UNMEASURED_BY_LOCAL_MCP_CLIENT",
            "model_tool_selection": "UNMEASURED_BY_LOCAL_MCP_CLIENT",
            "exact_loaded_tool_definition_bytes": None,
        },
        "surface": {
            "catalog_operation_count": len(catalog_ids),
            "advertised_tool_count": len(records),
            "advertised_direct_operation_count": len(direct_records),
            "fixed_tool_count": len(fixed_records),
            "complete_catalog_schema_coverage": schema_coverage,
            "tool_names_sha256": _sha256(_json_bytes(sorted(advertised))),
            "tool_definitions_sha256": _sha256(_json_bytes(records)),
            "all_tool_definition_bytes": _tool_bytes(records),
            "direct_tool_definition_bytes": _tool_bytes(direct_records),
            "fixed_tool_definition_bytes": _tool_bytes(fixed_records),
            "direct_input_output_schema_bytes": schema_bytes,
        },
        "latency": {
            "server_construction_ms": round(construction_elapsed * 1_000, 6),
            "client_initialization_ms": round(client_elapsed * 1_000, 6),
            "tools_list": _latency_summary(listing_latencies),
        },
        "tasks": tasks,
        "semantic_discovery": semantic_cases,
        "summary": {
            "discovery_success_count": sum(
                probe["success"] for task in tasks for probe in task["discovery"]
            ),
            "discovery_probe_count": sum(len(task["discovery"]) for task in tasks),
            "direct_task_success_count": sum(task["direct_success"] for task in tasks),
            "legacy_task_success_count": sum(task["legacy_success"] for task in tasks),
            "exact_parity_task_count": sum(
                task["exact_output_parity"] for task in tasks
            ),
            "composition_success_count": sum(
                task["composition_success"]
                for task in tasks
                if task["category"] == TaskCategory.MULTI_OPERATION
            ),
            "composition_case_count": sum(
                task["category"] == TaskCategory.MULTI_OPERATION for task in tasks
            ),
            "semantic_discovery_success_count": sum(
                case["success"] for case in semantic_cases
            ),
        },
        "decision": decision,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic catalog-scale direct MCP migration controls."
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list-repetitions", type=int, default=7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.list_repetitions <= 100:
        raise SystemExit("--list-repetitions must be between 1 and 100")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    suite = load_suite(args.suite.resolve(strict=True))
    report = asyncio.run(run_evaluation(suite, list_repetitions=args.list_repetitions))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    local_failures = (
        any(gate == "FAIL" for gate in report["decision"]["math_run"]["gates"].values())
        or report["decision"]["math_find"]["gates"][
            "local_semantic_vocabulary_reachability"
        ]
        == "FAIL"
    )
    if local_failures:
        raise SystemExit("one or more deterministic evaluation controls failed")


if __name__ == "__main__":
    main()


__all__ = [
    "CatalogEvaluationSuite",
    "DecisionPolicy",
    "TaskCategory",
    "load_suite",
    "run_evaluation",
]
