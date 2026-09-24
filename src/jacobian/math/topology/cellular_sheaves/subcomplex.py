"""Restriction of finite cellular sheaves to included subcomplexes."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_RESTRICTION_CELLS,
    MAX_SHEAF_RESTRICTION_OUTPUT_CHARS,
    MAX_SHEAF_SIMPLICES,
    FiniteCellularSheaf,
    SheafSubcomplexResult,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)


def _domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("subcomplex",),
        code=f"topology.cellular_sheaf.subcomplex.{code}",
        message=message,
    )


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("sheaf",),
        code=f"topology.cellular_sheaf.subcomplex.{code}",
        message=message,
    )


def restrict_to_subcomplex(
    sheaf: FiniteCellularSheaf, subcomplex: FiniteSimplicialComplex
) -> SheafSubcomplexResult:
    """Return the exact sheaf diagram induced on an included subcomplex."""
    source_cells = sheaf.canonical_face_order
    selected_cells = tuple(
        face for group in subcomplex.faces_by_dimension for face in group.faces
    )
    source_set = set(source_cells)
    if not set(subcomplex.vertices).issubset(sheaf.complex.vertices) or not set(
        selected_cells
    ).issubset(source_set):
        raise _domain(
            "not_included",
            "every subcomplex vertex and simplex must belong to the sheaf complex",
        )

    if (
        len(source_cells) > MAX_SHEAF_SIMPLICES
        or len(selected_cells) > MAX_SHEAF_SIMPLICES
    ):
        raise _resource(
            "simplex_bound",
            f"restriction admits at most {MAX_SHEAF_SIMPLICES} cells in either complex",
        )
    selected = set(selected_cells)
    if tuple(stalk.simplex for stalk in sheaf.stalks) != source_cells:
        raise _domain(
            "source_stalk_order",
            "the source sheaf must bind exactly one stalk to each canonical cell",
        )
    basis_for = {stalk.simplex: stalk.basis for stalk in sheaf.stalks}
    field = _admit_field(sheaf.coefficient_field, sheaf.prime)
    filtered_stalks = tuple(
        stalk for stalk in sheaf.stalks if stalk.simplex in selected
    )
    filtered_cover = tuple(
        item
        for item in sheaf.cover_restrictions
        if item.source in selected and item.target in selected
    )
    filtered_derived = tuple(
        item
        for item in sheaf.derived_restrictions
        if item.source in selected and item.target in selected
    )
    restriction_count = len(filtered_cover) + len(filtered_derived)
    matrix_cells = sum(
        len(item.row_basis) * len(item.column_basis)
        for item in (*filtered_cover, *filtered_derived)
    )
    if (
        restriction_count > MAX_SHEAF_DERIVED_RESTRICTIONS
        or matrix_cells > MAX_SHEAF_RESTRICTION_CELLS
    ):
        raise _resource(
            "restriction_bound",
            "the filtered comparable-cell diagram exceeds its admitted bound",
        )

    expected_pairs = tuple(
        (source, target)
        for source in selected_cells
        for target in selected_cells
        if len(source) < len(target) and set(source).issubset(target)
    )
    retained_pairs = {
        (item.source, item.target) for item in (*filtered_cover, *filtered_derived)
    }
    if retained_pairs != set(expected_pairs):
        raise _domain(
            "incomplete_source_diagram",
            "the source sheaf must carry every restriction between included cells",
        )
    for item in (*filtered_cover, *filtered_derived):
        if (
            basis_for.get(item.source) != item.column_basis
            or basis_for.get(item.target) != item.row_basis
            or len(item.entries) != len(item.row_basis)
            or any(len(row) != len(item.column_basis) for row in item.entries)
        ):
            raise _domain(
                "source_restriction_axes",
                "each retained restriction must match its source and target bases",
            )
        for row in item.entries:
            for scalar in row:
                try:
                    if (
                        sheaf_scalar_digits(scalar) > 64
                        or field.typed(field.parse(scalar)) != scalar
                    ):
                        raise ValueError("noncanonical scalar")
                except (ValueError, ZeroDivisionError) as error:
                    raise _domain(
                        "source_scalar_invalid",
                        "retained restrictions must use canonical exact field scalars",
                    ) from error

    # Admission precedes construction.  The source is included in the returned
    # source-bound result, and the target repeats retained exact maps.
    output_chars = len(sheaf.model_dump_json()) + len(subcomplex.model_dump_json())
    output_chars += (
        sum(
            len(item.model_dump_json())
            for item in (*filtered_stalks, *filtered_cover, *filtered_derived)
        )
        + 1024
    )
    output_scalar_count = sum(
        len(row)
        for item in (*filtered_cover, *filtered_derived)
        for row in item.entries
    )
    output_chars += sheaf_scalar_json_bound(output_scalar_count)
    if output_chars > MAX_SHEAF_RESTRICTION_OUTPUT_CHARS:
        raise _resource(
            "output_bound",
            f"the source-bound result exceeds the {MAX_SHEAF_RESTRICTION_OUTPUT_CHARS}-character bound",
        )

    stalk_for = {stalk.simplex: stalk for stalk in filtered_stalks}
    target = FiniteCellularSheaf._from_kernel(
        complex=subcomplex,
        coefficient_field=sheaf.coefficient_field,
        prime=sheaf.prime,
        stalks=tuple(stalk_for[cell] for cell in selected_cells),
        cover_restrictions=filtered_cover,
        derived_restrictions=filtered_derived,
        diamonds=sum(
            1
            for source, target_cell in expected_pairs
            if len(target_cell) - len(source) == 2
        ),
        comparable_pairs=len(expected_pairs),
    )
    return SheafSubcomplexResult._from_kernel(source=sheaf, subcomplex=target)


__all__ = ["restrict_to_subcomplex"]
