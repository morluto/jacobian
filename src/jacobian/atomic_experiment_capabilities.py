"""Atomic structure and experiment capability registrations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from jacobian.atomic_capability_builders import AdapterFactory, SchemaBuilder
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
)
from jacobian.contracts.discovery import (
    ExperimentCancelResult,
    ExperimentHandle,
    ExperimentSnapshot,
    StructureCanonicalizationResult,
)
from jacobian.contracts.search import (
    ExperimentControlResult,
    SearchExperimentSnapshot,
)
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    from jacobian.atomic_capabilities import AtomicServiceAdapter
    from jacobian.runtime.services import ApplicationServices


def build_experiment_adapters(
    application: ApplicationServices,
    *,
    adapter: AdapterFactory,
    schema: SchemaBuilder,
    artifact_uri: dict[str, Any],
    experiment_uri: dict[str, Any],
    enumeration_budget_schema: Callable[[], dict[str, Any]],
) -> tuple[AtomicServiceAdapter, ...]:
    """Build structure computation and durable experiment lifecycle adapters."""

    return (
        adapter(
            capability_id="structure.canonicalize",
            title="Canonicalize one structure",
            description=(
                "Compute a plugin-defined canonical representative without "
                "self-certification."
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=schema(
                {
                    "structure_uri": artifact_uri,
                    "plugin_id": artifact_uri,
                    "wall_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 86400,
                    },
                },
                required=("structure_uri", "plugin_id", "wall_seconds"),
            ),
            output_schema=model_schema(StructureCanonicalizationResult),
            invoke=lambda p: application.structures.canonicalize(**p),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="plugin canonicalization is not independently verified",
            tags=("structure", "canonicalization"),
        ),
        adapter(
            capability_id="search.enumerate",
            title="Start a bounded enumeration",
            description=(
                "Start one durable candidate-enumeration experiment; it cannot "
                "self-certify."
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=schema(
                {
                    "claim_uri": artifact_uri,
                    "plugin_id": artifact_uri,
                    "bounds": {"type": "object"},
                    "quotient_by_isomorphism": {"type": "boolean"},
                    "profile": {"enum": ["FAST", "EXACT_CANDIDATE"]},
                    "seed": {"type": "integer"},
                    "budget": enumeration_budget_schema(),
                },
                required=("claim_uri", "plugin_id", "bounds", "budget"),
            ),
            output_schema=model_schema(ExperimentHandle),
            invoke=lambda p: application.experiments.start_enumeration(p),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis=(
                "enumeration lifecycle state cannot certify a mathematical conclusion"
            ),
            tags=("search", "enumeration", "experiment"),
        ),
        adapter(
            capability_id="experiment.inspect",
            title="Inspect one experiment",
            description=(
                "Read the durable state and accounting of one enumeration or "
                "search experiment."
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=schema(
                {"experiment_uri": experiment_uri},
                required=("experiment_uri",),
            ),
            output_schema=TypeAdapter(
                ExperimentSnapshot | SearchExperimentSnapshot
            ).json_schema(),
            invoke=lambda p: application.experiment_router.inspect(p["experiment_uri"]),
            read_only=True,
            tags=("experiment",),
        ),
        adapter(
            capability_id="experiment.wait",
            title="Wait for an experiment update",
            description=(
                "Wait for a bounded interval and return the latest experiment snapshot."
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=schema(
                {
                    "experiment_uri": experiment_uri,
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 86400,
                    },
                },
                required=("experiment_uri",),
            ),
            output_schema=TypeAdapter(
                ExperimentSnapshot | SearchExperimentSnapshot
            ).json_schema(),
            invoke=lambda p: application.experiment_router.wait(
                p["experiment_uri"],
                timeout_seconds=p.get("timeout_seconds", 30),
            ),
            read_only=True,
            tags=("experiment",),
        ),
        adapter(
            capability_id="experiment.cancel",
            title="Request experiment cancellation",
            description=(
                "Request cancellation of one running enumeration or search experiment."
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=schema(
                {"experiment_uri": experiment_uri},
                required=("experiment_uri",),
            ),
            output_schema=TypeAdapter(
                ExperimentCancelResult | ExperimentControlResult
            ).json_schema(),
            invoke=lambda p: application.experiment_router.cancel(p["experiment_uri"]),
            tags=("experiment", "control"),
        ),
    )
