"""Filtered-map transport and explicit finite stabilization profiles."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_SPECTRAL_PAGE,
    FiltrationLevel,
    SpectralPageResult,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    _in_span,
    _mat_vec,
    _parse_entry,
    admit_filtered,
    spectral_page,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    ChainMapValue,
)


class FilteredChainMapRequest(StrictModel):
    chain_map: ChainMapValue
    source_filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    target_filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)


class FilteredChainMapResult(StrictModel):
    chain_map: ChainMapValue
    source_filtration: tuple[FiltrationLevel, ...]
    target_filtration: tuple[FiltrationLevel, ...]
    filtration_preserving: bool
    is_chain_map: bool


class SpectralPagesRequest(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    through_page: int = Field(ge=0, le=MAX_SPECTRAL_PAGE, default=MAX_SPECTRAL_PAGE)


class SpectralPagesResult(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...]
    pages: tuple[SpectralPageResult, ...]
    stabilized_page: int | None = None
    status: str


class SpectralAbutmentRequest(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)


class SpectralAbutmentResult(StrictModel):
    complex: ChainComplexValue
    page: SpectralPageResult
    status: str
    associated_graded_dimensions: tuple[tuple[int, ...], ...]


def filtered_map(request: FilteredChainMapRequest) -> FilteredChainMapResult:
    chain_map = request.chain_map
    source, target, matrices = (
        chain_map.source,
        chain_map.target,
        chain_map.map_matrices,
    )
    if (
        source.coefficient_ring != target.coefficient_ring
        or source.prime != target.prime
        or source.degree_min != target.degree_min
        or source.degree_max != target.degree_max
    ):
        raise OperationDomainValidationError(
            location=("target",),
            code="filtered_chain_map.parent_mismatch",
            message="source and target complexes must share coefficient and degree parents",
        )
    admit_filtered(source, request.source_filtration)
    admit_filtered(target, request.target_filtration)
    if len(request.source_filtration) != len(request.target_filtration) or len(
        matrices
    ) != len(source.basis_sizes):
        raise OperationDomainValidationError(
            location=("maps",),
            code="filtered_chain_map.axis_mismatch",
            message="map and filtration degree axes must agree",
        )
    p = source.prime
    parsed = []
    for degree, matrix in enumerate(matrices):
        rows = target.basis_sizes[degree]
        cols = source.basis_sizes[degree]
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise OperationDomainValidationError(
                location=("maps", degree),
                code="filtered_chain_map.shape_invalid",
                message="each map must have target-by-source chain axes",
            )
        try:
            parsed.append([[_parse_entry(v, p) for v in row] for row in matrix])
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise OperationDomainValidationError(
                location=("maps", degree),
                code="filtered_chain_map.entry_invalid",
                message="map entries must use the retained canonical coefficient grammar",
            ) from exc
    # f d = d f, with matrix convention d rows lower x upper
    chain_ok = True
    for degree in range(len(parsed) - 1):
        output_width = source.basis_sizes[degree + 1]
        left = _mul(
            parsed[degree],
            [
                [_parse_entry(v, p) for v in row]
                for row in source.differential_matrices[degree]
            ],
            p,
            output_width=output_width,
        )
        right = _mul(
            [
                [_parse_entry(v, p) for v in row]
                for row in target.differential_matrices[degree]
            ],
            parsed[degree + 1],
            p,
            output_width=output_width,
        )
        chain_ok = chain_ok and left == right
    preserving = True
    for _level, (sl, tl) in enumerate(
        zip(request.source_filtration, request.target_filtration, strict=True)
    ):
        for degree, (ss, ts) in enumerate(zip(sl.subspaces, tl.subspaces, strict=True)):
            target_basis = [[_parse_entry(v, p) for v in row] for row in ts.vectors]
            for vector in ss.vectors:
                image = _mat_vec(
                    parsed[degree], [_parse_entry(v, p) for v in vector], p
                )
                if not _in_span(target_basis, image, p):
                    preserving = False
    return FilteredChainMapResult(
        chain_map=chain_map,
        source_filtration=request.source_filtration,
        target_filtration=request.target_filtration,
        filtration_preserving=preserving,
        is_chain_map=chain_ok,
    )


def _mul(left: Any, right: Any, prime: int | None, *, output_width: int) -> Any:
    if not left:
        return []
    if not right:
        return [[0] * output_width for _ in left]
    cols = list(zip(*right, strict=False))
    result = []
    for row in left:
        output = []
        for col in cols:
            value = sum(a * b for a, b in zip(row, col, strict=False))
            output.append(value % prime if prime is not None else value)
        result.append(output)
    return result


def pages_through(request: SpectralPagesRequest) -> SpectralPagesResult:
    admit_filtered(request.complex, request.filtration)
    pages = []
    stable = None
    for page in range(request.through_page + 1):
        value = spectral_page(request.complex, request.filtration, page)
        pages.append(value)
        if value.page_status.value == "STABILIZED":
            stable = page
            break
    return SpectralPagesResult(
        complex=request.complex,
        filtration=request.filtration,
        pages=tuple(pages),
        stabilized_page=stable,
        status="STABILIZED"
        if stable is not None
        else ("TRUNCATED" if len(pages) == request.through_page + 1 else "ACTIVE"),
    )


def abutment(request: SpectralAbutmentRequest) -> SpectralAbutmentResult:
    profile = pages_through(
        SpectralPagesRequest(complex=request.complex, filtration=request.filtration)
    )
    if profile.stabilized_page is None:
        raise OperationDomainValidationError(
            location=("filtration",),
            code="spectral_sequence.not_stabilized",
            message="the admitted page window did not establish an abutment",
        )
    page = profile.pages[-1]
    return SpectralAbutmentResult(
        complex=request.complex,
        page=page,
        status="STABILIZED",
        associated_graded_dimensions=page.page_dimensions,
    )


__all__ = [
    "FilteredChainMapRequest",
    "FilteredChainMapResult",
    "SpectralAbutmentRequest",
    "SpectralAbutmentResult",
    "SpectralPagesRequest",
    "SpectralPagesResult",
    "abutment",
    "filtered_map",
    "pages_through",
]
