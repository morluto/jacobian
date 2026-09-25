"""Induced maps on exact cellular-sheaf cohomology."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.cellular_sheaves._kernel import (
    _cochain_rref,
    _ExactField,
    sheaf_cohomology,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COHOMOLOGY_CELLS,
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_TOTAL_STALK_RANK,
    SheafCohomologyGroup,
    SheafCohomologyResult,
    SheafField,
    SheafScalar,
    _require_field_scalars,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafCochainMapResult,
    SheafMorphismResult,
    cochain_map,
)

MAX_SHEAF_COHOMOLOGY_MAP_WORK = MAX_SHEAF_MORPHISM_WORK
Scalar = Fraction | int


class SheafCohomologyMapRequest(StrictModel):
    """A source-bound natural sheaf morphism whose induced map is requested."""

    morphism: SheafMorphismResult


class SheafCohomologyMapResult(StrictModel):
    """Degreewise maps on the retained ordered cohomology bases.

    ``components[k]`` has target Betti number rows and source Betti number
    columns.  The morphism binds both sheaf parents and their exact cochain
    axes; each group retains its degree, cochain dimension, Betti number, and
    ordered cocycle representatives that define the matrix coordinates.
    """

    morphism: SheafMorphismResult
    source_groups: tuple[SheafCohomologyGroup, ...] = Field(min_length=1)
    target_groups: tuple[SheafCohomologyGroup, ...] = Field(min_length=1)
    components: tuple[tuple[tuple[SheafScalar, ...], ...], ...]

    @model_validator(mode="after")
    def require_structural_axes(self) -> Self:
        morphism = self.morphism
        if (
            not morphism.natural
            or morphism.obstruction is not None
            or morphism.source.complex != morphism.target.complex
            or morphism.source.coefficient_field != morphism.target.coefficient_field
            or morphism.source.prime != morphism.target.prime
        ):
            raise ValueError(
                "cohomology maps must retain one natural morphism over one complex and field"
            )
        source_groups = self.source_groups
        target_groups = self.target_groups
        dimension = morphism.source.complex.dimension
        if len(source_groups) != dimension + 1 or len(target_groups) != dimension + 1:
            raise ValueError("cohomology groups must cover every source degree")
        expected_faces = morphism.source.complex.faces_by_dimension
        for degree, (source_group, target_group, face_axis) in enumerate(
            zip(source_groups, target_groups, expected_faces, strict=True)
        ):
            source_stalks = {item.simplex: item for item in morphism.source.stalks}
            target_stalks = {item.simplex: item for item in morphism.target.stalks}
            source_dimension = sum(
                len(source_stalks[face].basis) for face in face_axis.faces
            )
            target_dimension = sum(
                len(target_stalks[face].basis) for face in face_axis.faces
            )
            if (
                source_group.degree != degree
                or target_group.degree != degree
                or source_group.cochain_dimension != source_dimension
                or target_group.cochain_dimension != target_dimension
            ):
                raise ValueError("cohomology groups do not match morphism cochain axes")
            for group in (source_group, target_group):
                if (
                    group.cocycle_rank > group.cochain_dimension
                    or group.coboundary_rank > group.cocycle_rank
                    or group.betti_number != group.cocycle_rank - group.coboundary_rank
                    or (degree == 0 and group.coboundary_rank != 0)
                    or len(group.cocycle_representatives) != group.betti_number
                    or any(
                        len(vector) != group.cochain_dimension
                        for vector in group.cocycle_representatives
                    )
                ):
                    raise ValueError("cohomology representative axes are malformed")
        if len(self.components) != len(source_groups) or len(target_groups) != len(
            source_groups
        ):
            raise ValueError("induced maps must cover every cohomology degree")
        for degree, matrix in enumerate(self.components):
            if len(matrix) != target_groups[degree].betti_number or any(
                len(row) != source_groups[degree].betti_number for row in matrix
            ):
                raise ValueError("induced map matrix does not match cohomology axes")
        _require_field_scalars(
            tuple(group.cocycle_representatives for group in source_groups)
            + tuple(group.cocycle_representatives for group in target_groups)
            + self.components,
            morphism.source.coefficient_field,
            morphism.source.prime,
            label="induced cohomology map",
        )
        scalar_count = sum(
            len(vector)
            for group in (*source_groups, *target_groups)
            for vector in group.cocycle_representatives
        ) + sum(len(row) for matrix in self.components for row in matrix)
        if (
            len(morphism.model_dump_json())
            + sheaf_scalar_json_bound(scalar_count)
            + 256 * (len(source_groups) + len(target_groups))
            > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
        ):
            raise ValueError("cohomology map coordinates exceed their output bound")
        if any(
            sheaf_scalar_digits(value) > 64
            for group in (*source_groups, *target_groups)
            for vector in group.cocycle_representatives
            for value in vector
        ) or any(
            sheaf_scalar_digits(value) > 64
            for matrix in self.components
            for row in matrix
            for value in row
        ):
            raise ValueError("cohomology map scalars exceed their digit bound")
        return self


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.cohomology_map.{code}",
        message=message,
    )


def _parse_vector(field: _ExactField, vector: tuple[SheafScalar, ...]) -> list[Scalar]:
    return [field.parse(value) for value in vector]


def _mat_vec(
    field: _ExactField,
    matrix: tuple[tuple[SheafScalar, ...], ...],
    vector: list[Scalar],
) -> list[Scalar]:
    result = []
    for row in matrix:
        total = field.zero()
        for scalar, coordinate in zip(row, vector, strict=True):
            total += field.parse(scalar) * coordinate
        if field.field is SheafField.PRIME_FIELD:
            total %= field.prime_modulus()
        result.append(total)
    return result


def _scalar_digits(value: Scalar) -> int:
    if isinstance(value, Fraction):
        return max(len(str(abs(value.numerator))), len(str(value.denominator)))
    return len(str(abs(value)))


def _rational_growth_components(value: SheafScalar | Scalar) -> tuple[int, int]:
    """Return numerator digits and a log bound for a rational denominator."""
    if isinstance(value, CanonicalRational):
        numerator, denominator = value.num, value.den
    elif isinstance(value, Fraction):
        numerator, denominator = value.numerator, value.denominator
    else:
        numerator, denominator = int(value), 1
    denominator_exponent = 0 if denominator == 1 else len(str(denominator - 1))
    return len(str(abs(numerator))), denominator_exponent


def _max_image_sum_terms(
    field: _ExactField,
    matrix: tuple[tuple[SheafScalar, ...], ...],
    representatives: tuple[tuple[SheafScalar, ...], ...],
) -> int:
    """Bound nonzero products contributing to any image coordinate."""
    maximum = 0
    parsed_matrix = tuple(tuple(field.parse(value) for value in row) for row in matrix)
    parsed_representatives = tuple(
        tuple(field.parse(value) for value in vector) for vector in representatives
    )
    for row in parsed_matrix:
        for vector in parsed_representatives:
            terms = sum(
                left != 0 and right != 0
                for left, right in zip(row, vector, strict=True)
            )
            maximum = max(maximum, terms)
    return maximum


def _admit_quotient_reduction(
    induced_cochains: SheafCochainMapResult,
    source: SheafCohomologyResult,
    target: SheafCohomologyResult,
    field: _ExactField,
) -> tuple[tuple[tuple[Scalar, ...], ...], ...]:
    """Admit quotient work, scalar growth, matrix cells, and serialized output."""
    quotient_work = 0
    output_cells = 0
    for degree, (source_group, target_group) in enumerate(
        zip(source.groups, target.groups, strict=True)
    ):
        target_dimension = target_group.cochain_dimension
        previous_dimension = target.cochain_dimensions[degree - 1] if degree else 0
        image_work = (
            source_group.betti_number
            if target_dimension == 0
            else target_dimension
            * source_group.cochain_dimension
            * source_group.betti_number
        )
        image_preflight_work = (
            image_work
            if field.field is SheafField.RATIONAL and target_group.cocycle_rank
            else 0
        )
        quotient_work += (
            target_dimension**2
            * (
                previous_dimension
                + target_group.coboundary_rank
                + target_group.betti_number
                + source_group.betti_number
            )
            + image_work
            + image_preflight_work
        )
        output_cells += target_group.betti_number * source_group.betti_number
    if quotient_work > MAX_SHEAF_COHOMOLOGY_MAP_WORK:
        raise _resource(
            "work_bound",
            f"cohomology-map quotient reduction needs {quotient_work} scalar steps, "
            f"above the {MAX_SHEAF_COHOMOLOGY_MAP_WORK}-step bound",
        )
    if output_cells > MAX_SHEAF_COHOMOLOGY_CELLS:
        raise _resource(
            "matrix_cells_bound",
            "cohomology-map matrices exceed their cell bound",
        )
    representative_values = sum(
        len(vector)
        for group in (*source.groups, *target.groups)
        for vector in group.cocycle_representatives
    )
    # Cohomology representatives come from exact row reduction of admitted
    # 64-digit inputs. Their coordinates are not constrained to 64 digits;
    # conservatively admit the determinant-scale growth possible in a
    # 512-coordinate cochain complex before constructing the result model.
    representative_digit_bound = 64 * (MAX_SHEAF_TOTAL_STALK_RANK + 1)
    if field.field is SheafField.RATIONAL:
        for group in (*source.groups, *target.groups):
            for vector in group.cocycle_representatives:
                if any(
                    sheaf_scalar_digits(value) > representative_digit_bound
                    for value in vector
                ):
                    raise _resource(
                        "representative_growth_bound",
                        "cohomology representative exceeds the conservative exact-growth bound",
                    )
    output_chars = (
        len(induced_cochains.morphism.model_dump_json())
        + sheaf_scalar_json_bound(representative_values + output_cells)
        + 256 * (len(source.groups) + len(target.groups))
    )
    if output_chars > MAX_SHEAF_MORPHISM_OUTPUT_CHARS:
        raise _resource(
            "output_bound",
            "cohomology-map matrices exceed their serialized output bound",
        )

    images_by_degree: list[tuple[tuple[Scalar, ...], ...]] = []
    for degree, (source_group, target_group) in enumerate(
        zip(source.groups, target.groups, strict=True)
    ):
        quotient_rank = target_group.cocycle_rank
        if target_group.cochain_dimension == 0:
            images_by_degree.append(
                tuple(() for _ in source_group.cocycle_representatives)
            )
            continue
        source_vectors = tuple(
            _parse_vector(field, vector)
            for vector in source_group.cocycle_representatives
        )
        if source_group.betti_number == 0:
            images_by_degree.append(())
            continue
        source_matrix = induced_cochains.components[degree]
        if field.field is SheafField.RATIONAL and quotient_rank:
            max_terms = _max_image_sum_terms(
                field, source_matrix, source_group.cocycle_representatives
            )
            max_map_digits = max(
                (sheaf_scalar_digits(value) for row in source_matrix for value in row),
                default=1,
            )
            max_source_digits = max(
                (
                    _scalar_digits(value)
                    for vector in source_vectors
                    for value in vector
                ),
                default=1,
            )
            image_digits_bound = max_terms * (max_map_digits + max_source_digits) + (
                len(str(max_terms - 1)) if max_terms > 1 else 0
            )
            if image_digits_bound > 64:
                raise _resource(
                    "scalar_growth_bound",
                    "the pre-admitted cochain-image digit bound "
                    f"{image_digits_bound} exceeds the 64-digit limit",
                )
        images = tuple(
            tuple(_mat_vec(field, source_matrix, vector)) for vector in source_vectors
        )
        images_by_degree.append(images)
        if field.field is not SheafField.RATIONAL or quotient_rank == 0:
            continue
        rational_values: list[SheafScalar | Scalar] = [
            value for vector in images for value in vector
        ]
        rational_values.extend(
            value for vector in target_group.cocycle_representatives for value in vector
        )
        if degree:
            rational_values.extend(
                value for row in target.coboundary_matrices[degree - 1] for value in row
            )
        numerator_digits, denominator_exponent = zip(
            *(_rational_growth_components(value) for value in rational_values),
            strict=True,
        )
        max_numerator_digits = max(numerator_digits, default=1)
        max_denominator_exponent = max(denominator_exponent, default=0)
        # Clear denominators row-by-row in a q-by-q pivot minor. Each integer
        # row entry uses one numerator and at most q-1 row denominators; the
        # determinant and Cramer's ratio then give this digit bound.
        determinant_digits = (
            quotient_rank * max_numerator_digits
            + (2 * quotient_rank**2 - quotient_rank) * max_denominator_exponent
            + len(str(factorial(quotient_rank)))
            + 2
        )
        if determinant_digits > 64:
            raise _resource(
                "scalar_growth_bound",
                "the conservative exact quotient-coordinate digit bound "
                f"{determinant_digits} exceeds the 64-digit result limit",
            )
    return tuple(images_by_degree)


def cohomology_map(value: SheafMorphismResult) -> SheafCohomologyMapResult:
    """Compute the map induced on cohomology by a natural sheaf morphism."""
    # Re-establish naturality and derive all cochain and cohomology data from
    # the source diagrams. Serialized ``natural`` and cocycle claims are not
    # accepted as mathematical evidence by this consumer.
    induced_cochains = cochain_map(value)
    source = sheaf_cohomology(induced_cochains.morphism.source)
    target = sheaf_cohomology(induced_cochains.morphism.target)
    sheaf = induced_cochains.morphism.source
    field = _ExactField(sheaf.coefficient_field, sheaf.prime)
    images_by_degree = _admit_quotient_reduction(
        induced_cochains, source, target, field
    )

    matrices: list[tuple[tuple[SheafScalar, ...], ...]] = []
    for degree, (_source_group, target_group) in enumerate(
        zip(source.groups, target.groups, strict=True)
    ):
        n = target_group.cochain_dimension
        # Columns span B^k followed by the canonical returned H^k
        # representatives. By the cohomology computation these columns are
        # independent and span Z^k.
        columns: list[list[Scalar]] = []
        if degree:
            previous = target.coboundary_matrices[degree - 1]
            parsed_previous = [
                [field.parse(value) for value in row] for row in previous
            ]
            # Select a basis of the image, since the differential's raw
            # columns are generally dependent (for example, graph incidence
            # columns sum to zero on each connected component).
            _reduced_boundary, boundary_pivots = _cochain_rref(field, parsed_previous)
            for column in boundary_pivots:
                columns.append([row[column] for row in parsed_previous])
        columns.extend(
            _parse_vector(field, vector)
            for vector in target_group.cocycle_representatives
        )
        quotient_dimension = len(columns)
        images = images_by_degree[degree]
        # Row operations on [B | H | image columns] express each image in
        # the direct-sum basis B^k + chosen representatives of H^k.
        augmented_columns: list[list[Scalar]] = [
            *columns,
            *(list(image) for image in images),
        ]
        rows = [
            [augmented_columns[column][row] for column in range(len(augmented_columns))]
            for row in range(n)
        ]
        reduced, pivots = _cochain_rref(field, rows)
        if any(pivot >= quotient_dimension for pivot in pivots):
            raise RuntimeError("a natural morphism did not send cocycles to cocycles")
        if len(pivots) != quotient_dimension:
            raise RuntimeError(
                "target boundary and cohomology representatives are dependent"
            )
        # Pivot row i corresponds to basis column pivots[i]. Since the basis
        # columns are independent and listed first, pivots are 0..q-1.
        if pivots != tuple(range(quotient_dimension)):
            raise RuntimeError("target quotient basis has an unexpected pivot profile")
        target_betti = target_group.betti_number
        rows_of_map = []
        for target_index in range(target_betti):
            basis_index = len(columns) - target_betti + target_index
            rows_of_map.append(
                tuple(
                    field.typed(reduced[basis_index][quotient_dimension + source_index])
                    for source_index in range(len(images))
                )
            )
        matrices.append(tuple(rows_of_map))

    return SheafCohomologyMapResult(
        morphism=induced_cochains.morphism,
        source_groups=source.groups,
        target_groups=target.groups,
        components=tuple(matrices),
    )


__all__ = [
    "SheafCohomologyMapRequest",
    "SheafCohomologyMapResult",
    "cohomology_map",
]
