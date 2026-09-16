"""Exact bounded kernel for cellular sheaf construction from cover maps.

The kernel admits the resource envelope, checks the declared field and the
stalk/cover boundary data, verifies every length-two diamond of the face
poset exactly, and then materializes every derived comparable-face
restriction along one canonical lexicographic saturated chain.  Diamond
commutation on a simplicial face poset licenses that canonical chain: any two
saturated chains differ by a sequence of length-two diamond moves, so equal
diamonds make every composite path independent.  Validators never replay the
commutativity or composition mathematics.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex, Simplex
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_DIAMONDS,
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_PRIME,
    MAX_SHEAF_RESTRICTION_CELLS,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    MAX_SHEAF_TOTAL_STALK_RANK,
    CoverRestrictionMatrix,
    DiamondCounterexample,
    FiniteCellularSheaf,
    FromCoverMapsResult,
    SheafField,
    SheafObstruction,
    SheafObstructionCode,
    SheafOutcome,
    SheafRestriction,
    SheafStalk,
)

Scalar = Fraction | int
Matrix = tuple[tuple[Scalar, ...], ...]
CoverKey = tuple[Simplex, Simplex]


def _domain(
    code: str, message: str, location: tuple[str | int, ...]
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location, code=f"topology.cellular_sheaf.{code}", message=message
    )


def _resource(
    code: str, message: str, location: tuple[str | int, ...]
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=f"topology.cellular_sheaf.{code}", message=message
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


class _ExactField:
    """One declared exact scalar field with canonical text round trips."""

    def __init__(self, field: SheafField, prime: int | None) -> None:
        self.field = field
        self.prime = prime

    def parse(self, text: str) -> Scalar:
        stripped = text.strip()
        if self.field is SheafField.RATIONAL:
            if not stripped or any(char not in "-+/0123456789" for char in stripped):
                raise ValueError(f"{text!r} is not an exact rational scalar")
            return Fraction(stripped)
        if not stripped or any(char not in "-+0123456789" for char in stripped):
            raise ValueError(f"{text!r} is not an exact integer scalar")
        assert self.prime is not None
        return int(stripped) % self.prime

    def text(self, value: Scalar) -> str:
        if self.field is SheafField.RATIONAL:
            assert isinstance(value, Fraction)
            if value.denominator == 1:
                return format_canonical_integer(value.numerator)
            return (
                f"{format_canonical_integer(value.numerator)}/"
                f"{format_canonical_integer(value.denominator)}"
            )
        return str(int(value))

    def zero(self) -> Scalar:
        return Fraction(0) if self.field is SheafField.RATIONAL else 0

    def one(self) -> Scalar:
        return Fraction(1) if self.field is SheafField.RATIONAL else 1

    def identity(self, size: int) -> Matrix:
        return tuple(
            tuple(self.one() if r == c else self.zero() for c in range(size))
            for r in range(size)
        )

    def matmul(self, left: Matrix, right: Matrix) -> Matrix:
        rows = len(left)
        inner = len(right)
        columns = len(right[0]) if inner else 0
        product: list[tuple[Scalar, ...]] = []
        for i in range(rows):
            row: list[Scalar] = []
            for j in range(columns):
                total: Scalar = self.zero()
                for k in range(inner):
                    total = total + left[i][k] * right[k][j]
                if self.field is SheafField.PRIME_FIELD:
                    assert self.prime is not None
                    total = int(total) % self.prime
                row.append(total)
            product.append(tuple(row))
        return tuple(product)

    def equals(self, left: Matrix, right: Matrix) -> bool:
        if len(left) != len(right):
            return False
        for row_left, row_right in zip(left, right, strict=True):
            if len(row_left) != len(row_right) or any(
                a != b for a, b in zip(row_left, row_right, strict=True)
            ):
                return False
        return True

    def render(self, matrix: Matrix) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(self.text(value) for value in row) for row in matrix)


def _cells(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


def _cover_relations(cells: tuple[Simplex, ...]) -> list[CoverKey]:
    """Return every codimension-one cover pair in canonical sorted order."""

    known = set(cells)
    covers: list[CoverKey] = []
    for coface in cells:
        for position in range(len(coface)):
            face = coface[:position] + coface[position + 1 :]
            if face in known:
                covers.append((face, coface))
    covers.sort()
    return covers


def _canonical_chain(source: Simplex, target: Simplex) -> list[Simplex]:
    """Return the lexicographic saturated chain from ``source`` to ``target``."""

    chain = [source]
    current = source
    for vertex in sorted(set(target) - set(source)):
        current = tuple(sorted((*current, vertex)))
        chain.append(current)
    return chain


def _admit_field(coefficient_field: SheafField, prime: int | None) -> _ExactField:
    if coefficient_field is SheafField.PRIME_FIELD:
        if prime is None:
            raise _domain(
                "prime_required",
                "GF(p) sheaves must declare their prime modulus",
                ("prime",),
            )
        if not _is_prime(prime) or not 2 <= prime <= MAX_SHEAF_PRIME:
            raise _domain(
                "prime_not_admitted",
                f"the declared modulus {prime} is not a prime within the "
                f"bounded sheaf envelope",
                ("prime",),
            )
    elif prime is not None:
        raise _domain(
            "prime_forbidden",
            "QQ sheaves must not declare a prime modulus",
            ("prime",),
        )
    return _ExactField(coefficient_field, prime)


def _admit_resources(
    stalks: tuple[SheafStalk, ...],
    cover_maps: tuple[CoverRestrictionMatrix, ...],
    cell_count: int,
) -> None:
    if cell_count > MAX_SHEAF_SIMPLICES:
        raise _resource(
            "admission.simplices",
            f"the complex has {cell_count} nonempty simplices, above the "
            f"{MAX_SHEAF_SIMPLICES}-simplex sheaf envelope",
            ("complex",),
        )
    if len(stalks) > MAX_SHEAF_SIMPLICES:
        raise _resource(
            "admission.stalks",
            f"the request declares {len(stalks)} stalks, above the "
            f"{MAX_SHEAF_SIMPLICES}-stalk envelope",
            ("stalks",),
        )
    if len(cover_maps) > MAX_SHEAF_COVER_MAPS:
        raise _resource(
            "admission.cover_maps",
            f"the request supplies {len(cover_maps)} cover maps, above "
            f"the {MAX_SHEAF_COVER_MAPS}-map envelope",
            ("cover_maps",),
        )
    total_rank = 0
    for index, stalk in enumerate(stalks):
        if len(stalk.basis) > MAX_SHEAF_STALK_RANK:
            raise _resource(
                "admission.stalk_rank",
                f"stalk {index} has rank {len(stalk.basis)}, above the "
                f"{MAX_SHEAF_STALK_RANK}-dimension stalk envelope",
                ("stalks", index),
            )
        total_rank += len(stalk.basis)
    if total_rank > MAX_SHEAF_TOTAL_STALK_RANK:
        raise _resource(
            "admission.total_stalk_rank",
            f"the aggregate stalk rank {total_rank} exceeds the "
            f"{MAX_SHEAF_TOTAL_STALK_RANK}-coordinate envelope",
            ("stalks",),
        )
    for index, cover_map in enumerate(cover_maps):
        for row in cover_map.entries:
            for entry in row:
                if len(entry) > MAX_SHEAF_ENTRY_DIGITS:
                    raise _resource(
                        "admission.entry_digits",
                        f"cover map {index} carries a scalar above the "
                        f"{MAX_SHEAF_ENTRY_DIGITS}-digit envelope",
                        ("cover_maps", index),
                    )


def _admit_stalks(
    stalks: tuple[SheafStalk, ...], cells: tuple[Simplex, ...]
) -> tuple[dict[Simplex, tuple[str, ...]], dict[Simplex, SheafStalk]]:
    known = set(cells)
    basis_for: dict[Simplex, tuple[str, ...]] = {}
    stalk_for: dict[Simplex, SheafStalk] = {}
    for index, stalk in enumerate(stalks):
        if stalk.simplex in basis_for:
            raise _domain(
                "duplicate_stalk",
                f"simplex {list(stalk.simplex)} declares more than one stalk",
                ("stalks", index),
            )
        if stalk.simplex not in known:
            raise _domain(
                "stalk_not_in_complex",
                f"stalk {index} is bound to {list(stalk.simplex)}, which is not "
                "a nonempty face of the source complex",
                ("stalks", index),
            )
        basis_for[stalk.simplex] = stalk.basis
        stalk_for[stalk.simplex] = stalk
    for cell in cells:
        if cell not in basis_for:
            raise _domain(
                "stalk_domain_incomplete",
                "every nonempty simplex needs exactly one stalk; the first "
                f"missing stalk is {list(cell)}",
                ("stalks",),
            )
    return basis_for, stalk_for


def _admit_cover_maps(
    cover_maps: tuple[CoverRestrictionMatrix, ...],
    covers: list[CoverKey],
    field: _ExactField,
) -> dict[CoverKey, Matrix]:
    cover_set = set(covers)
    supplied: dict[CoverKey, Matrix] = {}
    for index, cover_map in enumerate(cover_maps):
        key: CoverKey = (cover_map.source, cover_map.target)
        if key not in cover_set:
            raise _domain(
                "cover_map_not_a_cover",
                f"cover map {index} ({list(cover_map.source)}, "
                f"{list(cover_map.target)}) is not a codimension-one cover "
                "relation of the face poset",
                ("cover_maps", index),
            )
        if key in supplied:
            raise _domain(
                "duplicate_cover_map",
                f"cover map {index} repeats the relation "
                f"({list(cover_map.source)}, {list(cover_map.target)})",
                ("cover_maps", index),
            )
        try:
            supplied[key] = tuple(
                tuple(field.parse(entry) for entry in row) for row in cover_map.entries
            )
        except (ValueError, ZeroDivisionError) as exc:
            raise _domain(
                "entry_not_exact",
                f"cover map {index} carries a scalar outside the declared exact "
                f"field: {exc}",
                ("cover_maps", index, "entries"),
            ) from exc
    return supplied


def _admit_diagram_size(
    cells: tuple[Simplex, ...],
    basis_for: dict[Simplex, tuple[str, ...]],
    comparable: list[CoverKey],
    derived_pairs: list[CoverKey],
) -> None:
    dimension_of = {cell: len(cell) - 1 for cell in cells}
    diamonds = 0
    for source, target in comparable:
        if dimension_of[target] - dimension_of[source] != 2:
            continue
        middles = sum(
            1
            for middle in cells
            if dimension_of[middle] == dimension_of[source] + 1
            and set(source) < set(middle) < set(target)
        )
        diamonds += middles * (middles - 1) // 2
    if diamonds > MAX_SHEAF_DIAMONDS:
        raise _resource(
            "admission.diamonds",
            f"the face poset has {diamonds} length-two diamonds, above the "
            f"{MAX_SHEAF_DIAMONDS}-diamond envelope",
            ("complex",),
        )
    if len(derived_pairs) > MAX_SHEAF_DERIVED_RESTRICTIONS:
        raise _resource(
            "admission.derived_restrictions",
            f"the diagram needs {len(derived_pairs)} derived restrictions, above "
            f"the {MAX_SHEAF_DERIVED_RESTRICTIONS}-map envelope",
            ("complex",),
        )
    restriction_cells = sum(
        len(basis_for[target]) * len(basis_for[source]) for source, target in comparable
    )
    if restriction_cells > MAX_SHEAF_RESTRICTION_CELLS:
        raise _resource(
            "admission.restriction_cells",
            f"the complete restriction diagram has {restriction_cells} matrix "
            f"cells, above the {MAX_SHEAF_RESTRICTION_CELLS}-cell envelope",
            ("complex",),
        )


def _resolve_covers(
    covers: list[CoverKey],
    supplied: dict[CoverKey, Matrix],
    basis_for: dict[Simplex, tuple[str, ...]],
) -> tuple[dict[CoverKey, Matrix], SheafObstruction | None]:
    """Bind every cover relation or return the first cover obstruction."""

    resolved: dict[CoverKey, Matrix] = {}
    for source, target in covers:
        entries = supplied.get((source, target))
        if entries is None:
            return resolved, SheafObstruction(
                code=SheafObstructionCode.MISSING_COVER_MAP,
                message=(
                    "the cover diagram is incomplete: no restriction map was "
                    f"supplied for ({list(source)}, {list(target)})"
                ),
                source=source,
                target=target,
            )
        rows = len(basis_for[target])
        columns = len(basis_for[source])
        if len(entries) != rows or any(len(row) != columns for row in entries):
            return resolved, SheafObstruction(
                code=SheafObstructionCode.COVER_MAP_WRONG_AXIS,
                message=(
                    f"the map for ({list(source)}, {list(target)}) has shape "
                    f"{len(entries)}x"
                    f"{max((len(row) for row in entries), default=0)} but its "
                    f"stalk axes require {rows}x{columns}"
                ),
                source=source,
                target=target,
            )
        resolved[(source, target)] = entries
    return resolved, None


def _verify_diamonds(
    cells: tuple[Simplex, ...],
    comparable: list[CoverKey],
    resolved: dict[CoverKey, Matrix],
    field: _ExactField,
) -> tuple[SheafObstruction | None, int]:
    """Replay every length-two diamond or return the first counterexample."""

    dimension_of = {cell: len(cell) - 1 for cell in cells}
    diamond_count = 0
    for source, target in comparable:
        if dimension_of[target] - dimension_of[source] != 2:
            continue
        middles = sorted(
            middle
            for middle in cells
            if dimension_of[middle] == dimension_of[source] + 1
            and set(source) < set(middle) < set(target)
        )
        for i, first in enumerate(middles):
            for second in middles[i + 1 :]:
                diamond_count += 1
                left = field.matmul(
                    resolved[(first, target)], resolved[(source, first)]
                )
                right = field.matmul(
                    resolved[(second, target)], resolved[(source, second)]
                )
                if not field.equals(left, right):
                    return (
                        SheafObstruction(
                            code=SheafObstructionCode.NONCOMMUTING_DIAMOND,
                            message=(
                                "two saturated cover paths compose to different "
                                f"restriction maps from {list(source)} to "
                                f"{list(target)}"
                            ),
                            diamond=DiamondCounterexample(
                                source=source,
                                target=target,
                                first_path=(source, first, target),
                                second_path=(source, second, target),
                                first_matrix=field.render(left),
                                second_matrix=field.render(right),
                            ),
                        ),
                        diamond_count,
                    )
    return None, diamond_count


def _build_restrictions(
    covers: list[CoverKey],
    derived_pairs: list[CoverKey],
    resolved: dict[CoverKey, Matrix],
    basis_for: dict[Simplex, tuple[str, ...]],
    field: _ExactField,
) -> tuple[tuple[SheafRestriction, ...], tuple[SheafRestriction, ...]]:
    def compose(chain: list[Simplex]) -> Matrix:
        composite = field.identity(len(basis_for[chain[0]]))
        for earlier, later in pairwise(chain):
            composite = field.matmul(resolved[(earlier, later)], composite)
        return composite

    cover_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=basis_for[target],
            column_basis=basis_for[source],
            entries=field.render(resolved[(source, target)]),
            cover_path=(source, target),
        )
        for source, target in covers
    )
    derived_restrictions = tuple(
        SheafRestriction(
            source=source,
            target=target,
            row_basis=basis_for[target],
            column_basis=basis_for[source],
            entries=field.render(compose(_canonical_chain(source, target))),
            cover_path=tuple(_canonical_chain(source, target)),
        )
        for source, target in derived_pairs
    )
    return cover_restrictions, derived_restrictions


def _negative(obstruction: SheafObstruction) -> FromCoverMapsResult:
    return FromCoverMapsResult._from_kernel(
        outcome=SheafOutcome.NOT_A_SHEAF,
        sheaf=None,
        obstruction=obstruction,
    )


def from_cover_maps(
    complex_: FiniteSimplicialComplex,
    coefficient_field: SheafField,
    prime: int | None,
    stalks: tuple[SheafStalk, ...],
    cover_maps: tuple[CoverRestrictionMatrix, ...],
) -> FromCoverMapsResult:
    """Construct the canonical sheaf or return the first diagram obstruction."""

    field = _admit_field(coefficient_field, prime)
    try:
        require_canonical_complex_admission(complex_)
    except ValueError as exc:
        raise _domain("complex_not_canonical", str(exc), ("complex",)) from exc

    cells = _cells(complex_)
    _admit_resources(stalks, cover_maps, len(cells))
    basis_for, stalk_for = _admit_stalks(stalks, cells)
    covers = _cover_relations(cells)
    supplied = _admit_cover_maps(cover_maps, covers, field)

    dimension_of = {cell: len(cell) - 1 for cell in cells}
    comparable = sorted(
        (source, target)
        for source in cells
        for target in cells
        if source != target and set(source) < set(target)
    )
    derived_pairs = [
        pair
        for pair in comparable
        if dimension_of[pair[1]] - dimension_of[pair[0]] >= 2
    ]
    _admit_diagram_size(cells, basis_for, comparable, derived_pairs)

    resolved, obstruction = _resolve_covers(covers, supplied, basis_for)
    if obstruction is not None:
        return _negative(obstruction)
    obstruction, diamond_count = _verify_diamonds(cells, comparable, resolved, field)
    if obstruction is not None:
        return _negative(obstruction)

    cover_restrictions, derived_restrictions = _build_restrictions(
        covers, derived_pairs, resolved, basis_for, field
    )
    sheaf = FiniteCellularSheaf._from_kernel(
        complex=complex_,
        coefficient_field=coefficient_field,
        prime=prime,
        stalks=tuple(stalk_for[cell] for cell in cells),
        cover_restrictions=cover_restrictions,
        derived_restrictions=derived_restrictions,
        diamonds=diamond_count,
        comparable_pairs=len(comparable),
    )
    return FromCoverMapsResult._from_kernel(
        outcome=SheafOutcome.CELLULAR_SHEAF,
        sheaf=sheaf,
        obstruction=None,
    )


__all__ = ["from_cover_maps"]
