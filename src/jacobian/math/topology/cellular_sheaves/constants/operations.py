"""Exact bounded construction of constant cellular sheaves."""

from __future__ import annotations

import re
from math import comb

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex, Simplex
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field, _ExactField
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_DIAMONDS,
    MAX_SHEAF_RESTRICTION_CELLS,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    MAX_SHEAF_TOTAL_STALK_RANK,
    BasisLabel,
    FiniteCellularSheaf,
    SheafRestriction,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.constants._models import (
    ConstantSheafRequest,
)

MAX_CONSTANT_SHEAF_WORK = 1_000_000
MAX_CONSTANT_SHEAF_OUTPUT_BYTES = 8_000_000
_BASIS_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$")
type Pair = tuple[Simplex, Simplex]


def _domain(reason: str, message: str, location: tuple[str | int, ...]) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"topology.cellular_sheaf.constant.{reason}",
        message=message,
    )


def _resource(reason: str, message: str, location: tuple[str | int, ...]) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"topology.cellular_sheaf.constant.{reason}",
        message=message,
    )


def _cells(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


def _canonical_path(source: Simplex, target: Simplex) -> tuple[Simplex, ...]:
    path = [source]
    current = source
    for vertex in target:
        if vertex not in source:
            current = tuple(sorted((*current, vertex)))
            path.append(current)
    return tuple(path)


def _simplex_json_width(simplex: Simplex) -> int:
    # Vertex labels are restricted to ASCII alphanumerics and _ . : -.
    return 2 + sum(len(vertex) + 2 for vertex in simplex) + max(0, len(simplex) - 1)


def _basis_json_width(basis: tuple[BasisLabel, ...]) -> int:
    return 2 + sum(len(label) + 2 for label in basis) + max(0, len(basis) - 1)


def _estimate_result_bytes(
    complex_: FiniteSimplicialComplex,
    cells: tuple[Simplex, ...],
    comparable: tuple[Pair, ...],
    basis: tuple[BasisLabel, ...],
) -> int:
    complex_bytes = len(complex_.model_dump_json().encode("utf-8"))
    basis_bytes = _basis_json_width(basis)
    stalk_bytes = sum(192 + _simplex_json_width(cell) + basis_bytes for cell in cells)
    scalar_width = 64
    restriction_bytes = sum(
        384
        + _simplex_json_width(source)
        + _simplex_json_width(target)
        + 2 * basis_bytes
        + len(basis) ** 2 * scalar_width
        + sum(_simplex_json_width(face) for face in _canonical_path(source, target))
        for source, target in comparable
    )
    return complex_bytes + stalk_bytes + restriction_bytes + 1024


def _admit(
    request: ConstantSheafRequest,
) -> tuple[_ExactField, tuple[Simplex, ...], tuple[Pair, ...], tuple[Pair, ...], int]:
    if not isinstance(request, ConstantSheafRequest):
        _domain("request_type", "request must be a constant-sheaf request", ())
    if not isinstance(request.complex, FiniteSimplicialComplex):
        _domain(
            "complex_type",
            "complex must be a canonical finite simplicial complex",
            ("complex",),
        )
    if not isinstance(request.basis, tuple):
        _domain("basis_invalid", "basis identifiers must be an ordered tuple", ("basis",))
    if len(request.basis) > MAX_SHEAF_STALK_RANK:
        _resource(
            "basis_rank_bound",
            f"the common stalk basis exceeds the {MAX_SHEAF_STALK_RANK}-dimension limit",
            ("basis",),
        )
    if any(
        not isinstance(label, str) or _BASIS_LABEL.fullmatch(label) is None
        for label in request.basis
    ) or len(set(request.basis)) != len(request.basis):
        _domain(
            "basis_invalid",
            "basis identifiers must be distinct canonical labels",
            ("basis",),
        )
    field = _admit_field(request.coefficient_field, request.prime)
    try:
        require_canonical_complex_admission(request.complex)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.cellular_sheaf.constant.complex_not_canonical",
            message=str(exc),
        ) from exc

    cells = _cells(request.complex)
    cell_count = len(cells)
    rank = len(request.basis)
    if cell_count > MAX_SHEAF_SIMPLICES:
        _resource(
            "simplex_bound",
            f"the complex has {cell_count} nonempty simplices, above the {MAX_SHEAF_SIMPLICES}-simplex limit",
            ("complex",),
        )
    if cell_count * rank > MAX_SHEAF_TOTAL_STALK_RANK:
        _resource(
            "total_rank_bound",
            "the copied stalk bases exceed the aggregate stalk-rank limit",
            ("basis",),
        )

    comparable, covers, derived = _relation_pairs(cells)
    diamonds = _admit_diagram(request, cells, comparable, covers, derived)
    return field, cells, covers, derived, diamonds


def _relation_pairs(
    cells: tuple[Simplex, ...],
) -> tuple[tuple[Pair, ...], tuple[Pair, ...], tuple[Pair, ...]]:
    comparable = tuple(
        sorted(
            (source, target)
            for source in cells
            for target in cells
            if len(source) < len(target) and set(source) < set(target)
        )
    )
    covers = tuple(pair for pair in comparable if len(pair[1]) == len(pair[0]) + 1)
    derived = tuple(pair for pair in comparable if len(pair[1]) >= len(pair[0]) + 2)
    return comparable, covers, derived


def _count_diamonds(cells: tuple[Simplex, ...], comparable: tuple[Pair, ...]) -> int:
    cell_sets = {cell: frozenset(cell) for cell in cells}
    diamonds = 0
    for source, target in comparable:
        if len(target) - len(source) != 2:
            continue
        source_set = cell_sets[source]
        target_set = cell_sets[target]
        middle_count = sum(
            1
            for middle in cells
            if len(middle) == len(source) + 1
            and source_set < cell_sets[middle] < target_set
        )
        diamonds += comb(middle_count, 2)
    return diamonds


def _admit_diagram(
    request: ConstantSheafRequest,
    cells: tuple[Simplex, ...],
    comparable: tuple[Pair, ...],
    covers: tuple[Pair, ...],
    derived: tuple[Pair, ...],
) -> int:
    cell_count = len(cells)
    rank = len(request.basis)
    if len(covers) > MAX_SHEAF_COVER_MAPS:
        _resource(
            "cover_bound",
            f"the complex has {len(covers)} cover inclusions, above the {MAX_SHEAF_COVER_MAPS}-map limit",
            ("complex",),
        )
    if len(derived) > MAX_SHEAF_DERIVED_RESTRICTIONS:
        _resource(
            "derived_restriction_bound",
            f"the complete sheaf has {len(derived)} derived inclusions, above the {MAX_SHEAF_DERIVED_RESTRICTIONS}-map limit",
            ("complex",),
        )
    restriction_cells = len(comparable) * rank * rank
    if restriction_cells > MAX_SHEAF_RESTRICTION_CELLS:
        _resource(
            "restriction_cells_bound",
            f"the complete restriction diagram has {restriction_cells} matrix cells, above the {MAX_SHEAF_RESTRICTION_CELLS}-cell limit",
            ("complex",),
        )

    diamonds = _count_diamonds(cells, comparable)
    if diamonds > MAX_SHEAF_DIAMONDS:
        _resource(
            "diamond_bound",
            f"the face poset has {diamonds} length-two diamonds, above the {MAX_SHEAF_DIAMONDS}-diamond limit",
            ("complex",),
        )

    work = (
        cell_count**3
        + restriction_cells
        + sum(len(_canonical_path(source, target)) for source, target in derived)
    )
    if work > MAX_CONSTANT_SHEAF_WORK:
        _resource(
            "work_bound",
            f"constant-sheaf construction requires {work} bounded units, above {MAX_CONSTANT_SHEAF_WORK}",
            ("complex",),
        )
    result_bytes = _estimate_result_bytes(
        request.complex, cells, comparable, request.basis
    )
    if result_bytes > MAX_CONSTANT_SHEAF_OUTPUT_BYTES:
        _resource(
            "result_size_bound",
            f"the estimated exact sheaf value is {result_bytes} bytes, above the {MAX_CONSTANT_SHEAF_OUTPUT_BYTES}-byte limit",
            ("complex",),
        )
    return diamonds


def constant_sheaf(request: ConstantSheafRequest) -> FiniteCellularSheaf:
    """Copy one based exact vector space to every nonempty simplex.

    Every face-to-coface restriction is the identity under the copied basis
    identifications.  The result is the canonical finite cellular-sheaf value
    and composes directly with section and cohomology operations.
    """

    field, cells, covers, derived, diamonds = _admit(request)
    basis = request.basis
    identity = field.render(field.identity(len(basis)))
    stalks = tuple(SheafStalk(simplex=cell, basis=basis) for cell in cells)
    cover_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=basis,
            column_basis=basis,
            entries=identity,
            cover_path=(source, target),
        )
        for source, target in covers
    )
    derived_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=basis,
            column_basis=basis,
            entries=identity,
            cover_path=_canonical_path(source, target),
        )
        for source, target in derived
    )
    return FiniteCellularSheaf(
        complex=request.complex,
        coefficient_field=request.coefficient_field,
        prime=request.prime,
        stalks=stalks,
        cover_restrictions=cover_restrictions,
        derived_restrictions=derived_restrictions,
        diamonds=diamonds,
        comparable_pairs=len(covers) + len(derived),
    )


__all__ = ["constant_sheaf"]
