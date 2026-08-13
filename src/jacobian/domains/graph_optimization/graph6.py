"""Typed graph6 operation and checker declaration."""

from __future__ import annotations

from pydantic import Field, StrictStr

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.base import ContractModel
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.domains._examples import example
from jacobian.math.graphs.graph6 import Graph6DecodeValue, decode_graph6
from jacobian.operation_bindings import inline_operation
from jacobian.operations import OperationRefusalError, OperationSpec
from jacobian.provider_runtime import source_provider_runtime


class Graph6DecodeRequest(ContractModel):
    graph6: StrictStr = Field(min_length=1, max_length=352)


def _graph6_runtime(*, checker_ids: tuple[str, ...] = ()) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.graph6-checker",
        version="1",
        entrypoint="jacobian_checkers.graph6:check_graph6_decode",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-graph6-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


def _decode(request: Graph6DecodeRequest) -> Graph6DecodeValue:
    try:
        return decode_graph6(request.graph6)
    except ValueError as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="GRAPH6_DECODE_REFUSED",
                stage="graph6_decoding",
                message=str(exc),
                hint=(
                    "Supply standard small-order graph6 (orders 0-62), optionally "
                    "prefixed by >>graph6<<; sparse6, digraph6, extended headers, "
                    "invalid lengths, characters, and padding are rejected."
                ),
            )
        ) from exc


GRAPH6_CAPABILITIES = (
    inline_operation(
        OperationSpec(
            operation_id="graph.encoding.graph6.decode.compute",
            version="1",
            title="Decode canonical small-order graph6",
            description=(
                "Decode a headerless or standard-header graph6 string of order at "
                "most 62 using the column-major upper-triangle bit convention, "
                "returning sorted edges, degrees, and a canonical graph digest."
            ),
            request_type=Graph6DecodeRequest,
            result_type=Graph6DecodeValue,
            execute=_decode,
            tags=("graph", "encoding", "graph6", "deterministic", "exact"),
            invocation_examples=(
                example(
                    "triangle_graph6",
                    "Decode the graph6 representation of the triangle graph.",
                    {"graph6": "Bw"},
                ),
            ),
        )
    ),
)

GRAPH6_CHECKER_DECLARATIONS = (
    ExactReplayCheckerDeclaration(
        "graph.encoding.graph6.decode.compute",
        Graph6DecodeRequest,
        "check_graph6_decode",
        "graph.graph6-decode.standard-library-v1",
        entrypoint_module="jacobian_checkers.graph6",
        provider_runtime_factory=_graph6_runtime,
        replay_method="standard-library graph6 bitstream replay",
        reason=(
            "operator-authorized standard-library checker independently decodes "
            "the graph6 bitstream without importing the producer"
        ),
        verification_capability_id="graph.encoding.graph6.decode.verify",
        verification_title="Verify a canonical graph6 decode",
        verification_description=(
            "Independently replay the small-order graph6 header, upper-triangle "
            "bits, padding, sorted edges, degrees, and canonical graph digest."
        ),
        verification_tags=("verification", "exact", "graph", "encoding", "graph6"),
    ),
)

__all__ = ["GRAPH6_CAPABILITIES", "GRAPH6_CHECKER_DECLARATIONS"]
