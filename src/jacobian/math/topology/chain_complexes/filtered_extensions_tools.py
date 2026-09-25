# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredChainComplexRequest,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import *


def _map(r: Any) -> Any:
    return filtered_map(r)


def _page_zero_map(r: Any) -> Any:
    return filtered_chain_map_page_zero(r)


def _page_map(r: Any) -> Any:
    return filtered_chain_map_page(r)


def _compose_map(r: Any) -> Any:
    return filtered_chain_map_compose(r)


def _pages(r: Any) -> Any:
    return pages_through(r)


def _abut(r: Any) -> Any:
    return abutment(r)


def _homology_filtration(r: Any) -> Any:
    return filtered_homology_filtration(r)


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
        operation_id="homological.filtered_chain_complex.homology_filtration.compute",
        title="Compute the induced filtration on homology",
        description=(
            "Compute exact homology bases over a bounded prime field and the "
            "nested image of each retained chain filtration level in homology. "
            "Each image basis class includes a filtered cycle representative and "
            "an incoming chain whose boundary relates it to the retained global "
            "homology representative."
        ),
        request_type=FilteredChainComplexRequest,
        result_type=FilteredHomologyResult,
        run=_homology_filtration,
        tags=("homological", "filtered", "homology", "exact"),
        examples=(
            OperationExample(
                name="diagonal_line_in_homology",
                description=(
                    "The first homology filtration level is the diagonal line "
                    "inside a two-dimensional homology space."
                ),
                input={
                    "complex": {
                        "coefficient_ring": "GF_p",
                        "prime": 2,
                        "degree_min": 0,
                        "degree_max": 0,
                        "basis_sizes": [2],
                        "differential_matrices": [],
                    },
                    "filtration": [
                        {"subspaces": [{"vectors": [["1", "1"]]}]},
                        {"subspaces": [{"vectors": [["1", "0"], ["0", "1"]]}]},
                    ],
                },
            ),
        ),
    ),
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
        operation_id="homological.filtered_chain_map.page_zero.compute",
        title="Induce a map on the associated graded",
        description=(
            "For a filtration-preserving chain map, compute its exact E0 map "
            "in the quotient bases of the source and target associated-graded "
            "complexes. The result retains both original complexes and "
            "filtrations, and the induced maps are checked against the E0 "
            "differentials."
        ),
        request_type=FilteredChainMapRequest,
        result_type=FilteredChainMapPageZeroResult,
        run=_page_zero_map,
        tags=("homological", "filtered", "spectral-sequence", "exact"),
        examples=(
            OperationExample(
                name="identity_on_associated_graded",
                description="The identity filtered chain map induces an identity on each associated-graded summand.",
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
        operation_id="homological.filtered_chain_map.page.compute",
        title="Induce a map on a spectral-sequence page",
        description=(
            "For an exact filtration-preserving chain map, compute the map "
            "on a requested bounded E^r page. The result retains both "
            "source-bound page values and quotient-coordinate matrices; "
            "representative transport and commutation with d^r are checked."
        ),
        request_type=FilteredChainMapPageRequest,
        result_type=FilteredChainMapPageResult,
        run=_page_map,
        tags=("homological", "filtered", "spectral-sequence", "chain-map", "exact"),
        examples=(
            OperationExample(
                name="identity_on_e1",
                description="The identity map on a one-term zero-differential filtered complex induces the identity on E1.",
                input={
                    "map": {
                        "source": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "target": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "source_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "target_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "maps": [[["1"]]],
                        "filtration_preserving": True,
                        "chain_map": True,
                    },
                    "page": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.filtered_chain_map.compose.compute",
        title="Compose two filtered chain maps",
        description=(
            "Compose two exact filtration-preserving chain maps whose middle "
            "complex and filtration agree. The result is a degreewise matrix "
            "map from the first source to the second target."
        ),
        request_type=FilteredChainMapCompositionRequest,
        result_type=FilteredChainMapResult,
        run=_compose_map,
        tags=("homological", "filtered", "chain-map", "composition", "exact"),
        examples=(
            OperationExample(
                name="compose_two_scalar_maps",
                description=(
                    "Compose multiplication by two and three on a one-term "
                    "complex with its degenerate one-level filtration."
                ),
                input={
                    "first": {
                        "source": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "source_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "target": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "target_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "maps": [[["2"]]],
                        "filtration_preserving": True,
                        "chain_map": True,
                    },
                    "second": {
                        "source": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "source_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "target": {
                            "coefficient_ring": "QQ",
                            "degree_min": 0,
                            "degree_max": 0,
                            "basis_sizes": [1],
                            "differential_matrices": [],
                        },
                        "target_filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
                        "maps": [[["3"]]],
                        "filtration_preserving": True,
                        "chain_map": True,
                    },
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
        description=(
            "Return a stable E-infinity page only after the finite filtration "
            "proves all later differentials vanish, together with the exact "
            "comparison isomorphism to each quotient F_p H_n/F_(p-1) H_n. "
            "The matrices use one chosen basis for each graded homology "
            "quotient and do not assert a splitting of filtered homology. "
            "Currently admitted over bounded GF(p) coefficients."
        ),
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
