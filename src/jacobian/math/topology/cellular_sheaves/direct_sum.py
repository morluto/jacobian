"""Pointwise direct sums of bounded exact cellular sheaves."""

from __future__ import annotations

from typing import Any, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import Simplex
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field, _ExactField
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_RESTRICTION_CELLS,
    MAX_SHEAF_RESTRICTION_OUTPUT_CHARS,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    MAX_SHEAF_TOTAL_STALK_RANK,
    FiniteCellularSheaf,
    SheafRestriction,
    SheafScalar,
    SheafStalk,
    _require_field_scalars,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)

RestrictionKey = tuple[Simplex, Simplex]
RestrictionMap = dict[RestrictionKey, SheafRestriction]
StalkMap = dict[Simplex, SheafStalk]


class SheafDirectSumRequest(StrictModel):
    left: FiniteCellularSheaf
    right: FiniteCellularSheaf


class SheafDirectSumStalkInclusion(StrictModel):
    simplex: tuple[str, ...]
    left: tuple[tuple[SheafScalar, ...], ...]
    right: tuple[tuple[SheafScalar, ...], ...]


class SheafDirectSumResult(StrictModel):
    """Binary direct sum with injections represented on each stalk axis."""

    left_source: FiniteCellularSheaf
    right_source: FiniteCellularSheaf
    direct_sum: FiniteCellularSheaf
    stalk_inclusions: tuple[SheafDirectSumStalkInclusion, ...]

    @model_validator(mode="after")
    def require_inclusion_axes(self) -> Self:
        if self.left_source.complex != self.right_source.complex:
            raise ValueError("direct sum parents must be the same simplicial complex")
        if (
            self.left_source.coefficient_field != self.right_source.coefficient_field
            or self.left_source.prime != self.right_source.prime
            or self.direct_sum.complex != self.left_source.complex
            or self.direct_sum.coefficient_field != self.left_source.coefficient_field
            or self.direct_sum.prime != self.left_source.prime
        ):
            raise ValueError("direct sum must preserve its exact field and complex")
        cells = self.direct_sum.canonical_face_order
        if tuple(entry.simplex for entry in self.stalk_inclusions) != cells:
            raise ValueError("direct sum inclusions must cover canonical stalk axes")
        left = {item.simplex: item for item in self.left_source.stalks}
        right = {item.simplex: item for item in self.right_source.stalks}
        summed = {item.simplex: item for item in self.direct_sum.stalks}
        for entry in self.stalk_inclusions:
            lrank = len(left[entry.simplex].basis)
            rrank = len(right[entry.simplex].basis)
            rank = len(summed[entry.simplex].basis)
            if rank != lrank + rrank:
                raise ValueError(
                    "direct sum stalk rank must be the sum of source ranks"
                )
            if len(entry.left) != rank or any(len(row) != lrank for row in entry.left):
                raise ValueError("left inclusion matrix has incompatible stalk axes")
            if len(entry.right) != rank or any(
                len(row) != rrank for row in entry.right
            ):
                raise ValueError("right inclusion matrix has incompatible stalk axes")
            _require_field_scalars(
                (entry.left, entry.right),
                self.left_source.coefficient_field,
                self.left_source.prime,
                label="direct-sum inclusion",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("left",),
        code=f"topology.cellular_sheaf.direct_sum.{code}",
        message=message,
    )


def _domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("left",),
        code=f"topology.cellular_sheaf.direct_sum.{code}",
        message=message,
    )


def _admit_diagram(
    sheaf: FiniteCellularSheaf, field: _ExactField
) -> tuple[StalkMap, RestrictionMap]:
    cells = sheaf.canonical_face_order
    if len(cells) > MAX_SHEAF_SIMPLICES:
        raise _resource(
            "simplex_bound", f"at most {MAX_SHEAF_SIMPLICES} simplices are admitted"
        )
    stalks = {stalk.simplex: stalk for stalk in sheaf.stalks}
    if len(stalks) != len(cells) or tuple(stalks) != cells:
        raise _domain(
            "stalk_axis", "each canonical simplex must have exactly one stalk"
        )
    total_rank = 0
    for stalk in sheaf.stalks:
        if len(stalk.basis) > MAX_SHEAF_STALK_RANK:
            raise _resource("stalk_rank_bound", "an input stalk exceeds the rank bound")
        total_rank += len(stalk.basis)
    if total_rank > MAX_SHEAF_TOTAL_STALK_RANK:
        raise _resource(
            "total_rank_bound", "an input sheaf exceeds the total stalk rank bound"
        )
    covers = {(item.source, item.target): item for item in sheaf.cover_restrictions}
    derived = {(item.source, item.target): item for item in sheaf.derived_restrictions}
    strict_pairs = tuple(
        (source, target)
        for source in cells
        for target in cells
        if set(source) < set(target)
    )
    cover_pairs = tuple(
        (source, target)
        for source, target in strict_pairs
        if len(target) == len(source) + 1
    )
    derived_pairs = tuple(pair for pair in strict_pairs if pair not in set(cover_pairs))
    cover_pairs = tuple(
        (source, target)
        for source, target in strict_pairs
        if len(target) == len(source) + 1
    )
    derived_pairs = tuple(pair for pair in strict_pairs if pair not in set(cover_pairs))
    if (
        len(covers) != len(sheaf.cover_restrictions)
        or set(covers) != set(cover_pairs)
        or len(derived) != len(sheaf.derived_restrictions)
        or set(derived) != set(derived_pairs)
    ):
        raise _domain(
            "restriction_diagram", "each strict face inclusion needs one canonical map"
        )
    scalar_cells = 0
    for pair, restriction in (*covers.items(), *derived.items()):
        source, target = pair
        if (
            restriction.column_basis != stalks[source].basis
            or restriction.row_basis != stalks[target].basis
        ):
            raise _domain(
                "restriction_axes", "restriction axes must match their source stalks"
            )
        scalar_cells += len(restriction.row_basis) * len(restriction.column_basis)
        for row in restriction.entries:
            for scalar in row:
                if sheaf_scalar_digits(scalar) > MAX_SHEAF_ENTRY_DIGITS:
                    raise _resource(
                        "scalar_bound",
                        "an input restriction scalar exceeds its digit bound",
                    )
                try:
                    if field.typed(field.parse(scalar)) != scalar:
                        raise ValueError
                except (ValueError, ZeroDivisionError):
                    raise _domain(
                        "scalar_encoding", "input restriction scalars must be canonical"
                    ) from None
    if scalar_cells > MAX_SHEAF_RESTRICTION_CELLS:
        raise _resource(
            "matrix_cells_bound", "input restriction matrices exceed their cell bound"
        )
    return stalks, {**covers, **derived}


def direct_sum(  # noqa: C901
    left: FiniteCellularSheaf, right: FiniteCellularSheaf
) -> SheafDirectSumResult:
    """Construct the pointwise direct sum, retaining exact stalk injections."""
    if left.complex != right.complex:
        raise _domain(
            "complex_mismatch", "direct summands must use the same simplicial complex"
        )
    if left.coefficient_field != right.coefficient_field or left.prime != right.prime:
        raise _domain(
            "field_mismatch",
            "direct summands must use the same exact coefficient field",
        )
    field = _admit_field(left.coefficient_field, left.prime)
    left_stalks, left_maps = _admit_diagram(left, field)
    right_stalks, right_maps = _admit_diagram(right, field)
    cells = left.canonical_face_order
    summed_rank = 0
    for simplex in cells:
        rank = len(left_stalks[simplex].basis) + len(right_stalks[simplex].basis)
        if rank > MAX_SHEAF_STALK_RANK:
            raise _resource(
                "stalk_rank_bound", "a direct-sum stalk exceeds the rank bound"
            )
        summed_rank += rank
    if summed_rank > MAX_SHEAF_TOTAL_STALK_RANK:
        raise _resource(
            "total_rank_bound", "the direct sum exceeds the total stalk rank bound"
        )
    cover_count = len(left.cover_restrictions) + len(right.cover_restrictions)
    derived_count = len(left.derived_restrictions) + len(right.derived_restrictions)
    if (
        cover_count > MAX_SHEAF_COVER_MAPS
        or derived_count > MAX_SHEAF_DERIVED_RESTRICTIONS
    ):
        raise _resource(
            "restriction_count_bound", "direct-sum restriction count exceeds its bound"
        )
    pair_order = tuple(left_maps)
    output_matrix_cells = 0
    cover_pairs = tuple((item.source, item.target) for item in left.cover_restrictions)
    derived_pairs = tuple(
        (item.source, item.target) for item in left.derived_restrictions
    )
    for pair in pair_order:
        ls, lt = pair
        output_matrix_cells += (
            len(left_stalks[lt].basis) + len(right_stalks[lt].basis)
        ) * (len(left_stalks[ls].basis) + len(right_stalks[ls].basis))
    if output_matrix_cells > MAX_SHEAF_RESTRICTION_CELLS:
        raise _resource(
            "matrix_cells_bound",
            "direct-sum restriction matrices exceed their cell bound",
        )
    estimated_chars = (
        len(left.model_dump_json())
        + len(right.model_dump_json())
        + sheaf_scalar_json_bound(output_matrix_cells)
        + summed_rank * summed_rank * 3
        + 256 * len(cells)
        + 1024 * len(left_maps)
    )
    if estimated_chars > MAX_SHEAF_RESTRICTION_OUTPUT_CHARS:
        raise _resource(
            "output_bound",
            "direct-sum result is predicted to exceed the result-size bound",
        )

    sum_stalks: list[SheafStalk] = []
    inclusions: list[SheafDirectSumStalkInclusion] = []
    for simplex in cells:
        lb = left_stalks[simplex].basis
        rb = right_stalks[simplex].basis
        basis = tuple(f"L{i}" for i in range(len(lb))) + tuple(
            f"R{i}" for i in range(len(rb))
        )
        sum_stalks.append(SheafStalk(simplex=simplex, basis=basis))
        lmat = tuple(
            tuple(
                field.typed(field.one() if row == col else field.zero())
                for col in range(len(lb))
            )
            for row in range(len(lb) + len(rb))
        )
        rmat = tuple(
            tuple(
                field.typed(field.one() if row - len(lb) == col else field.zero())
                for col in range(len(rb))
            )
            for row in range(len(lb) + len(rb))
        )
        inclusions.append(
            SheafDirectSumStalkInclusion(simplex=simplex, left=lmat, right=rmat)
        )

    def sum_maps(
        mapping_left: RestrictionMap,
        mapping_right: RestrictionMap,
        pairs: tuple[RestrictionKey, ...],
    ) -> tuple[
        tuple[
            Simplex, Simplex, tuple[tuple[SheafScalar, ...], ...], tuple[Simplex, ...]
        ],
        ...,
    ]:
        out = []
        for source, target in pairs:
            ml = mapping_left[(source, target)]
            mr = mapping_right[(source, target)]
            rows = len(ml.row_basis) + len(mr.row_basis)
            columns = len(ml.column_basis) + len(mr.column_basis)
            matrix = [
                [field.typed(field.zero()) for _ in range(columns)] for _ in range(rows)
            ]
            for i, row in enumerate(ml.entries):
                for j, scalar in enumerate(row):
                    matrix[i][j] = field.typed(field.parse(scalar))
            for i, row in enumerate(mr.entries):
                for j, scalar in enumerate(row):
                    matrix[i + len(ml.row_basis)][j + len(ml.column_basis)] = (
                        field.typed(field.parse(scalar))
                    )
            out.append(
                (source, target, tuple(tuple(row) for row in matrix), ml.cover_path)
            )
        return tuple(out)

    cover_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=next(s.basis for s in sum_stalks if s.simplex == target),
            column_basis=next(s.basis for s in sum_stalks if s.simplex == source),
            entries=entries,
            cover_path=path,
        )
        for source, target, entries, path in sum_maps(
            left_maps, right_maps, cover_pairs
        )
    )
    derived_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=next(s.basis for s in sum_stalks if s.simplex == target),
            column_basis=next(s.basis for s in sum_stalks if s.simplex == source),
            entries=entries,
            cover_path=path,
        )
        for source, target, entries, path in sum_maps(
            left_maps, right_maps, derived_pairs
        )
    )
    out = FiniteCellularSheaf._from_kernel(
        complex=left.complex,
        coefficient_field=left.coefficient_field,
        prime=left.prime,
        stalks=tuple(sum_stalks),
        cover_restrictions=cover_restrictions,
        derived_restrictions=derived_restrictions,
        diamonds=left.diamonds,
        comparable_pairs=left.comparable_pairs,
    )
    result = SheafDirectSumResult(
        left_source=left,
        right_source=right,
        direct_sum=out,
        stalk_inclusions=tuple(inclusions),
    )
    return result
