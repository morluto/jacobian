"""Functorial maps on the exact homology of finite-module Koszul complexes."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    ModuleKoszulHomologyMap,
    ModuleKoszulHomologyMapRequest,
    ModuleKoszulMapRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_homology,
    module_koszul_map,
)

MAX_KOSZUL_HOMOLOGY_MAP_BASIS_CELLS = 8
MAX_KOSZUL_HOMOLOGY_MAP_COEFFICIENT_DIGITS = 8
MAX_KOSZUL_HOMOLOGY_MAP_OUTPUT_BYTES = 8 * 1024 * 1024


def _fraction(value: CanonicalRational) -> Fraction:
    return Fraction(value.num, value.den)


def _dense_map(matrix: Any) -> list[list[Fraction]]:
    result = [
        [Fraction(0) for _ in range(matrix.column_count)]
        for _ in range(matrix.row_count)
    ]
    for row, column, coefficient in matrix.entries:
        result[row][column] = _fraction(coefficient)
    return result


def _coordinates_mod_boundaries(
    cycle: list[Fraction],
    boundary_basis: tuple[tuple[CanonicalRational, ...], ...],
    homology_basis: tuple[tuple[CanonicalRational, ...], ...],
) -> list[Fraction]:
    """Express a cycle in the quotient basis modulo the boundary subspace."""

    columns = [
        [_fraction(value) for value in vector]
        for vector in (*boundary_basis, *homology_basis)
    ]
    unknowns = len(columns)
    if unknowns == 0:
        if any(cycle):
            raise ArithmeticError("a chain map sent a cycle outside target cycles")
        return []

    # Exact RREF of [basis | cycle].  The basis vectors are independent by the
    # homology producer's construction, so every target cycle has unique
    # coordinates in boundary-plus-homology basis.
    rows = [
        [columns[column][row] for column in range(unknowns)] + [cycle[row]]
        for row in range(len(cycle))
    ]
    pivot = 0
    for column in range(unknowns):
        source = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if source is None:
            raise ArithmeticError("target homology basis is not independent")
        rows[pivot], rows[source] = rows[source], rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [value / scale for value in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot and rows[i][column]:
                factor = rows[i][column]
                rows[i] = [
                    left - factor * right
                    for left, right in zip(rows[i], rows[pivot], strict=True)
                ]
        pivot += 1
    if any(row[-1] for row in rows[pivot:]):
        raise ArithmeticError("chain-map image is not a target cycle")
    boundary_count = len(boundary_basis)
    return [rows[boundary_count + index][-1] for index in range(len(homology_basis))]


def koszul_homology_map(
    request: ModuleKoszulHomologyMapRequest | Mapping[str, Any],
) -> ModuleKoszulHomologyMap:
    """Compute the induced map on every homology group of a typed chain map.

    The input is treated as caller-supplied data: module-linearity and all
    chain-map squares are reconstructed before its induced maps are used.
    Homology coordinates are relative to the exact canonical bases returned
    alongside the matrices.
    """

    try:
        value = (
            request
            if isinstance(request, ModuleKoszulHomologyMapRequest)
            else ModuleKoszulHomologyMapRequest.model_validate(request)
        )
        supplied = value.chain_map
        if (
            sum(supplied.source_complex.basis_sizes)
            + sum(supplied.target_complex.basis_sizes)
            > MAX_KOSZUL_HOMOLOGY_MAP_BASIS_CELLS
        ):
            raise OperationResourceAdmissionError(
                location=("chain_map",),
                code="koszul.module.homology_map_basis_budget",
                message="combined chain bases exceed the induced homology-map envelope",
            )
        coefficients = [
            coefficient
            for complex_value in (
                supplied.source_complex,
                supplied.target_complex,
            )
            for differential in complex_value.differentials
            for _, _, coefficient in differential.entries
        ]
        coefficients.extend(
            coefficient
            for degree_map in supplied.degree_maps
            for _, _, coefficient in degree_map.entries
        )
        maximum_input_digits = max(
            (
                canonical_rational_component_digits(coefficient)
                for coefficient in coefficients
            ),
            default=1,
        )
        if any(
            canonical_rational_component_digits(coefficient)
            > MAX_KOSZUL_HOMOLOGY_MAP_COEFFICIENT_DIGITS
            for coefficient in coefficients
        ):
            raise OperationResourceAdmissionError(
                location=("chain_map",),
                code="koszul.module.homology_map_coefficient_budget",
                message="chain-map coefficients exceed the induced homology-map envelope",
            )
        max_degree_size = max(
            (
                *supplied.source_complex.basis_sizes,
                *supplied.target_complex.basis_sizes,
            ),
            default=0,
        )
        homology_component_digits = (
            4 * max_degree_size * (maximum_input_digits + 4) + 32
        )
        # Homology representatives are ratios of minors. The induced coordinate
        # solve uses at most max_degree_size such rational entries in each minor;
        # this deliberately generous estimate is checked before exact expansion.
        induced_component_digits = max_degree_size * (
            2 * homology_component_digits + maximum_input_digits + 8
        )
        degree_map_cells = sum(
            source_size * target_size
            for source_size, target_size in zip(
                supplied.source_complex.basis_sizes,
                supplied.target_complex.basis_sizes,
                strict=True,
            )
        )
        if degree_map_cells * (2 * induced_component_digits + 64) > (
            MAX_KOSZUL_HOMOLOGY_MAP_OUTPUT_BYTES
        ):
            raise OperationResourceAdmissionError(
                location=("degree_maps",),
                code="koszul.module.homology_map_output_budget",
                message="induced homology-map output exceeds its preflight byte bound",
            )
    except OperationResourceAdmissionError:
        raise
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.homology_map_request",
            message="the supplied Koszul chain map is not canonical",
        ) from exc

    # Reconstruct from the original module map, so serialized producer claims
    # about module-linearity or chain-map equations are never trusted here.
    verified = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=supplied.algebra,
            source=supplied.source,
            target=supplied.target,
            sequence=supplied.sequence,
            map_matrix=supplied.module_map,
        )
    )
    source_homology = module_koszul_homology(verified.source_complex)
    target_homology = module_koszul_homology(verified.target_complex)
    induced: list[tuple[tuple[CanonicalRational, ...], ...]] = []
    for degree, matrix in enumerate(verified.degree_maps):
        dense = _dense_map(matrix)
        source_classes = source_homology.degrees[degree].homology_basis
        target_degree = target_homology.degrees[degree]
        target_dimension = target_homology.dimensions[degree]
        columns: list[list[Fraction]] = []
        for source_class in source_classes:
            vector = [_fraction(value) for value in source_class]
            image = [
                sum(
                    (row[column] * vector[column] for column in range(len(vector))),
                    Fraction(0),
                )
                for row in dense
            ]
            columns.append(
                _coordinates_mod_boundaries(
                    image,
                    target_degree.boundary_basis,
                    target_degree.homology_basis,
                )
            )
        induced.append(
            tuple(
                tuple(
                    CanonicalRational.from_fraction(columns[column][row])
                    for column in range(len(columns))
                )
                for row in range(target_dimension)
            )
        )

    result = ModuleKoszulHomologyMap(
        chain_map=verified,
        source_homology=source_homology,
        target_homology=target_homology,
        degree_maps=tuple(induced),
    )
    output_bytes = sum(
        (abs(value.num).bit_length() + 7) // 8 + (value.den.bit_length() + 7) // 8 + 32
        for degree_map in result.degree_maps
        for row in degree_map
        for value in row
    )
    if output_bytes > MAX_KOSZUL_HOMOLOGY_MAP_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("degree_maps",),
            code="koszul.module.homology_map_output_budget",
            message="induced homology-map matrices exceed the output envelope",
        )
    return result
