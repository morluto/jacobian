"""Canonical values for homogeneous monomial systems on algebraic tori."""

from __future__ import annotations

from itertools import pairwise
from math import ceil, log10, prod
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_INPUT_DIGITS,
    MAX_CERTIFIED_SNF_INPUT_DIMENSION,
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)

TorusAxisLabel = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$",
        strict=True,
    ),
]
MAX_TORUS_COMPONENT_DIGITS = (
    MAX_CERTIFIED_SNF_INPUT_DIMENSION * MAX_CERTIFIED_SNF_INPUT_DIGITS
    + ceil(
        MAX_CERTIFIED_SNF_INPUT_DIMENSION * log10(MAX_CERTIFIED_SNF_INPUT_DIMENSION) / 2
    )
    + 1
)
PositiveTorusInteger = Annotated[
    CanonicalInteger,
    StringConstraints(
        max_length=MAX_TORUS_COMPONENT_DIGITS,
        pattern=r"^[1-9][0-9]*$",
        strict=True,
    ),
]


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _raw_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _integer_digits(value: str) -> int:
    return len(value.lstrip("-"))


class HomogeneousMonomialSystem(StrictModel):
    """An axis-bound system ``product_j x_j^A_ij = 1`` on ``(CC*)^n``."""

    exponent_matrix: CertifiedIntegerMatrix
    equation_axis: tuple[TorusAxisLabel, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    coordinate_axis: tuple[TorusAxisLabel, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    coordinate_domain: Literal["NONZERO_COMPLEX"] = "NONZERO_COMPLEX"
    equation_convention: Literal["ROWS_ARE_X_TO_INTEGER_EXPONENT_EQUALS_ONE"] = (
        "ROWS_ARE_X_TO_INTEGER_EXPONENT_EQUALS_ONE"
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_smith_envelope(cls, data: Any) -> Any:
        """Reject oversized nested matrices before their general value parser."""

        if not isinstance(data, dict):
            return data
        matrix = data.get("exponent_matrix")
        rows = _raw_field(matrix, "row_count")
        columns = _raw_field(matrix, "column_count")
        entries = _raw_field(matrix, "entries")
        for size in (rows, columns):
            if type(size) is int and not (
                0 <= size <= MAX_CERTIFIED_SNF_INPUT_DIMENSION
            ):
                raise _validation_error(
                    "algebraic_torus.monomial_system_dimension_bound",
                    "monomial systems admit at most 16 equations and coordinates",
                )
        if isinstance(entries, (list, tuple)):
            if type(rows) is int and len(entries) != rows:
                raise _validation_error(
                    "algebraic_torus.monomial_system_matrix_shape",
                    "exponent entries must match the declared matrix rows",
                )
            if type(columns) is int and any(
                isinstance(row, (list, tuple)) and len(row) != columns
                for row in entries
            ):
                raise _validation_error(
                    "algebraic_torus.monomial_system_matrix_shape",
                    "exponent entries must match the declared matrix columns",
                )
            if any(
                isinstance(value, str)
                and _integer_digits(value) > MAX_CERTIFIED_SNF_INPUT_DIGITS
                for row in entries
                if isinstance(row, (list, tuple))
                for value in row
            ):
                raise _validation_error(
                    "algebraic_torus.monomial_system_exponent_bound",
                    "monomial exponents may contain at most 32 decimal digits",
                )
        return data

    @model_validator(mode="after")
    def require_axes_and_envelope(self) -> Self:
        matrix = self.exponent_matrix
        if (
            matrix.row_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
            or matrix.column_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
        ):
            raise _validation_error(
                "algebraic_torus.monomial_system_dimension_bound",
                "monomial systems admit at most 16 equations and coordinates",
            )
        if len(self.equation_axis) != matrix.row_count:
            raise _validation_error(
                "algebraic_torus.monomial_system_equation_axis",
                "equation axis must match the exponent-matrix rows",
            )
        if len(self.coordinate_axis) != matrix.column_count:
            raise _validation_error(
                "algebraic_torus.monomial_system_coordinate_axis",
                "coordinate axis must match the exponent-matrix columns",
            )
        if len(set(self.equation_axis)) != len(self.equation_axis):
            raise _validation_error(
                "algebraic_torus.monomial_system_equation_axis",
                "equation-axis labels must be unique",
            )
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise _validation_error(
                "algebraic_torus.monomial_system_coordinate_axis",
                "coordinate-axis labels must be unique",
            )
        if any(
            _integer_digits(value) > MAX_CERTIFIED_SNF_INPUT_DIGITS
            for row in matrix.entries
            for value in row
        ):
            raise _validation_error(
                "algebraic_torus.monomial_system_exponent_bound",
                "monomial exponents may contain at most 32 decimal digits",
            )
        return self


class TorsionCharacterGroup(StrictModel):
    """Compact component indices ``product_i Z/d_i`` in Smith order."""

    invariant_factors: tuple[PositiveTorusInteger, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    label_convention: Literal["CARTESIAN_PRODUCT_OF_CANONICAL_RESIDUES"] = (
        "CARTESIAN_PRODUCT_OF_CANONICAL_RESIDUES"
    )
    root_convention: Literal["ZETA_D_EQUALS_EXP_2_PI_I_OVER_D"] = (
        "ZETA_D_EQUALS_EXP_2_PI_I_OVER_D"
    )

    @model_validator(mode="after")
    def require_canonical_invariant_factors(self) -> Self:
        factors = tuple(
            parse_canonical_integer(value) for value in self.invariant_factors
        )
        if any(value <= 1 for value in factors) or any(
            right % left for left, right in pairwise(factors)
        ):
            raise _validation_error(
                "algebraic_torus.torsion_invariant_factors",
                "torsion invariant factors must exceed one and form a divisibility chain",
            )
        return self


class AlgebraicTorusSolutionSubgroup(StrictModel):
    """The complete compact subgroup solving one homogeneous monomial system.

    For ``D = U A V``, Smith parameters ``z`` map to source coordinates by
    ``x_i = product_j z_j ** V[i,j]``. Nontrivial Smith coordinates index the
    connected components and the final columns are free torus parameters.
    """

    source: HomogeneousMonomialSystem
    smith_certificate: SmithNormalFormCertificate
    torsion_character_group: TorsionCharacterGroup
    connected_component_count: PositiveTorusInteger
    torsion_parameter_axis: tuple[TorusAxisLabel, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    smith_free_parameter_axis: tuple[TorusAxisLabel, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    reduced_free_parameter_axis: tuple[TorusAxisLabel, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_INPUT_DIMENSION
    )
    torsion_exponent_map: CertifiedIntegerMatrix
    smith_free_exponent_map: CertifiedIntegerMatrix
    reduced_free_exponent_map: CertifiedIntegerMatrix
    smith_free_parameters_from_reduced: CertifiedIntegerMatrix
    free_rank: StrictInt = Field(ge=0, le=MAX_CERTIFIED_SNF_INPUT_DIMENSION)
    parameterization_convention: Literal[
        "X_I_EQUALS_PRODUCT_OF_PARAMETERS_TO_EXPONENT_MAP_IJ"
    ] = "X_I_EQUALS_PRODUCT_OF_PARAMETERS_TO_EXPONENT_MAP_IJ"

    @model_validator(mode="before")
    @classmethod
    def require_raw_result_shapes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        source = data.get("source")
        coordinate_axis = _raw_field(source, "coordinate_axis")
        free_rank = data.get("free_rank")
        torsion_axis = data.get("torsion_parameter_axis")
        if not isinstance(coordinate_axis, (list, tuple)) or type(free_rank) is not int:
            return data
        coordinate_count = len(coordinate_axis)
        torsion_count = (
            len(torsion_axis) if isinstance(torsion_axis, (list, tuple)) else None
        )
        expected_shapes = (
            ("torsion_exponent_map", coordinate_count, torsion_count),
            ("smith_free_exponent_map", coordinate_count, free_rank),
            ("reduced_free_exponent_map", coordinate_count, free_rank),
            ("smith_free_parameters_from_reduced", free_rank, free_rank),
        )
        for field_name, rows, columns in expected_shapes:
            matrix = data.get(field_name)
            raw_rows = _raw_field(matrix, "row_count")
            raw_columns = _raw_field(matrix, "column_count")
            if type(raw_rows) is int and raw_rows != rows:
                raise _validation_error(
                    "algebraic_torus.solution_map_shape",
                    "solution exponent maps must match their declared axes",
                )
            if (
                columns is not None
                and type(raw_columns) is int
                and raw_columns != columns
            ):
                raise _validation_error(
                    "algebraic_torus.solution_map_shape",
                    "solution exponent maps must match their declared axes",
                )
        return data

    @model_validator(mode="after")
    def require_source_bound_shapes(self) -> Self:
        coordinate_count = len(self.source.coordinate_axis)
        rank = self.smith_certificate.rank
        factors = self.smith_certificate.invariant_factors
        nontrivial_factors = tuple(factor for factor in factors if factor != "1")
        torsion_count = len(nontrivial_factors)
        if self.smith_certificate.source != self.source.exponent_matrix:
            raise _validation_error(
                "algebraic_torus.solution_source_binding",
                "Smith certificate source must equal the exponent matrix",
            )
        if self.free_rank != coordinate_count - rank:
            raise _validation_error(
                "algebraic_torus.solution_rank_shape",
                "free rank must be coordinate count minus Smith rank",
            )
        if self.torsion_character_group.invariant_factors != nontrivial_factors:
            raise _validation_error(
                "algebraic_torus.solution_torsion_shape",
                "torsion characters must follow the nontrivial Smith factors",
            )
        expected_component_count = prod(
            (
                parse_canonical_integer(value)
                for value in self.torsion_character_group.invariant_factors
            ),
            start=1,
        )
        if (
            parse_canonical_integer(self.connected_component_count)
            != expected_component_count
        ):
            raise _validation_error(
                "algebraic_torus.solution_component_count",
                "connected component count must equal the product of torsion invariant factors",
            )
        if (
            len(self.torsion_parameter_axis) != torsion_count
            or len(set(self.torsion_parameter_axis)) != torsion_count
        ):
            raise _validation_error(
                "algebraic_torus.solution_torsion_axis",
                "torsion parameter axis must match the nontrivial Smith factors",
            )
        if (
            len(self.smith_free_parameter_axis) != self.free_rank
            or len(set(self.smith_free_parameter_axis)) != self.free_rank
            or len(self.reduced_free_parameter_axis) != self.free_rank
            or len(set(self.reduced_free_parameter_axis)) != self.free_rank
        ):
            raise _validation_error(
                "algebraic_torus.solution_free_axis",
                "free parameter axis must match the free rank",
            )
        expected_shapes = (
            (self.torsion_exponent_map, coordinate_count, torsion_count),
            (self.smith_free_exponent_map, coordinate_count, self.free_rank),
            (self.reduced_free_exponent_map, coordinate_count, self.free_rank),
            (
                self.smith_free_parameters_from_reduced,
                self.free_rank,
                self.free_rank,
            ),
        )
        if any(
            (matrix.row_count, matrix.column_count) != (rows, columns)
            for matrix, rows, columns in expected_shapes
        ):
            raise _validation_error(
                "algebraic_torus.solution_map_shape",
                "solution exponent maps must match their declared axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: HomogeneousMonomialSystem,
        smith_certificate: SmithNormalFormCertificate,
        torsion_character_group: TorsionCharacterGroup,
        connected_component_count: str,
        torsion_parameter_axis: tuple[str, ...],
        smith_free_parameter_axis: tuple[str, ...],
        reduced_free_parameter_axis: tuple[str, ...],
        torsion_exponent_map: CertifiedIntegerMatrix,
        smith_free_exponent_map: CertifiedIntegerMatrix,
        reduced_free_exponent_map: CertifiedIntegerMatrix,
        smith_free_parameters_from_reduced: CertifiedIntegerMatrix,
        free_rank: int,
    ) -> Self:
        return cls.model_construct(
            source=source,
            smith_certificate=smith_certificate,
            torsion_character_group=torsion_character_group,
            connected_component_count=connected_component_count,
            torsion_parameter_axis=torsion_parameter_axis,
            smith_free_parameter_axis=smith_free_parameter_axis,
            reduced_free_parameter_axis=reduced_free_parameter_axis,
            torsion_exponent_map=torsion_exponent_map,
            smith_free_exponent_map=smith_free_exponent_map,
            reduced_free_exponent_map=reduced_free_exponent_map,
            smith_free_parameters_from_reduced=smith_free_parameters_from_reduced,
            free_rank=free_rank,
        )


__all__ = [
    "AlgebraicTorusSolutionSubgroup",
    "HomogeneousMonomialSystem",
    "TorsionCharacterGroup",
]
