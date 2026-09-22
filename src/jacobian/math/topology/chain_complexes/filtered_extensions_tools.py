# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.chain_complexes.filtered_extensions import *


def _map(r: Any) -> Any:
    return filtered_map(r)


def _pages(r: Any) -> Any:
    return pages_through(r)


def _abut(r: Any) -> Any:
    return abutment(r)


_C = {
    "coefficient_ring": "QQ",
    "degree_min": 0,
    "degree_max": 1,
    "basis_sizes": [1, 1],
    "differential_matrices": [[["1"]]],
}
_F = [
    {"subspaces": [{"vectors": [["1"]]}, {"vectors": []}]},
    {"subspaces": [{"vectors": [["1"]]}, {"vectors": [["1"]]}]},
]
TOOLS = (
    MathTool(
        operation_id="homological.filtered_chain_map.compute",
        title="Compute a filtered chain-map profile",
        description="Check an exact degreewise chain map between filtered complexes and return its chain-map and filtration-preservation decisions with all source and target axes retained.",
        request_type=FilteredChainMapRequest,
        result_type=FilteredChainMapResult,
        run=_map,
        tags=("homological", "filtered", "chain-map", "exact"),
        examples=(
            OperationExample(
                name="identity_filtered_map",
                description="Check the identity map of a two-term filtered complex; source and target filtrations must have matching levels.",
                input={
                    "source": _C,
                    "target": _C,
                    "source_filtration": _F,
                    "target_filtration": _F,
                    "maps": [[["1"]], [["1"]]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.spectral_sequence.pages.compute",
        title="Compute a finite spectral-sequence page profile",
        description="Compute pages E^0 through a requested finite page, retaining representative transport and distinguishing explicit stabilization from an unproven truncation.",
        request_type=SpectralPagesRequest,
        result_type=SpectralPagesResult,
        run=_pages,
        tags=("homological", "spectral-sequence", "stabilization", "exact"),
        examples=(
            OperationExample(
                name="pages_two_step",
                description="Compute the bounded page profile of a two-term filtered complex; page results retain the finite filtration and source axes.",
                input={"complex": _C, "filtration": _F, "through_page": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="homological.spectral_sequence.abutment.compute",
        title="Return an explicitly stabilized spectral-sequence abutment",
        description="Return the last computed page only when the admitted finite window proves stabilization; active or truncated pages are not promoted to an abutment.",
        request_type=SpectralAbutmentRequest,
        result_type=SpectralAbutmentResult,
        run=_abut,
        tags=("homological", "spectral-sequence", "abutment", "exact"),
        examples=(
            OperationExample(
                name="zero_differential_abutment",
                description="Return the stabilized abutment of a one-level zero-differential filtered complex; stabilization is explicit rather than inferred from truncation.",
                input={
                    "complex": {
                        "coefficient_ring": "QQ",
                        "degree_min": 0,
                        "degree_max": 0,
                        "basis_sizes": [1],
                        "differential_matrices": [],
                    },
                    "filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
