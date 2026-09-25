"""Canonical exact values for supported modular forms and q-expansions."""

from __future__ import annotations

from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.kernel import (
    NamedLevelOneModularForm,
    metadata,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

MAX_MODULAR_FORM_LEVEL = 100_000
MAX_MODULAR_FORM_WEIGHT = 1_000_000
MAX_MODULAR_CHARACTER_INCLUSION_LEVEL = 2_048

# Operation owners keep the reusable q-prefix carrier broader than any one
# transform.  Transform admission below uses this source envelope before it
# indexes coefficients or allocates a result.
MAX_Q_TRANSFORM_SOURCE_ORDER = 25_280
MAX_Q_TRANSFORM_OUTPUT_PRECISION = 4_096
MAX_Q_TRANSFORM_COEFFICIENT_DIGITS = 4_096
MAX_GAMMA0_OPERATION_LEVEL = 10_000
MAX_LEVEL_ONE_BASIS_WEIGHT = 120
MAX_LEVEL_ONE_BASIS_PRECISION = 128
MAX_GAMMA0_THREE_BASIS_PRECISION = 63
MAX_LEVEL_ONE_BASIS_COORDINATES = 32
MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS = 128
MAX_MODULAR_FORM_COEFFICIENT_FIELD_ORDER = 128
MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE = 32


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"modular_forms.{reason}", message)


def _require_rational_space(space: ModularFormSpace, value_kind: str) -> None:
    """Reject cyclotomic parents on carriers whose coefficients are rational."""

    if space.coefficient_domain != "QQ":
        raise _validation_error(
            "rational_value_parent",
            f"{value_kind} stores rational coefficients and requires a QQ space",
        )


class LevelOneModularQExpansion(StrictModel):
    """One normalized named form in QQ[[q]] through a declared precision.

    ``q_expansion`` contains every coefficient from q^0 through q^(P-1).
    Coefficients beyond that finite prefix are intentionally not represented.
    """

    form: NamedLevelOneModularForm
    congruence_subgroup: Literal["SL2Z"] = "SL2Z"
    level: Literal[1] = 1
    weight: Literal[4, 6, 12]
    space_kind: Literal["HOLOMORPHIC", "CUSP"]
    coefficient_domain: Literal["QQ"] = "QQ"
    normalization: str = Field(min_length=1, max_length=96)
    q_expansion: TruncatedSeries

    @model_validator(mode="after")
    def require_structural_named_form(self) -> Self:
        weight, space_kind, normalization = metadata(self.form)
        if (
            self.weight != weight
            or self.space_kind != space_kind
            or self.normalization != normalization
        ):
            raise _validation_error(
                "metadata_mismatch",
                "level-one modular metadata does not match the named form",
            )
        if self.q_expansion.variable != "q":
            raise _validation_error(
                "variable_mismatch",
                "a modular q-expansion must use the canonical variable q",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: NamedLevelOneModularForm,
        weight: Literal[4, 6, 12],
        space_kind: Literal["HOLOMORPHIC", "CUSP"],
        normalization: str,
        q_expansion: TruncatedSeries,
    ) -> Self:
        """Construct a value after the owner kernel established its coefficients."""

        return cls.model_construct(
            form=form,
            weight=weight,
            space_kind=space_kind,
            normalization=normalization,
            q_expansion=q_expansion,
        )


class ModularQExpansion(StrictModel):
    """A finite q-prefix bound to one concrete exact-character space."""

    space: ModularFormSpace | None = None
    weight: StrictInt = Field(ge=0)
    q_expansion: TruncatedSeries
    basis_id: str = Field(default="canonical", min_length=1, max_length=96)

    @model_validator(mode="after")
    def require_q_parent(self) -> Self:
        if self.q_expansion.variable != "q":
            raise _validation_error("q_variable", "modular q-expansions use q")
        if self.q_expansion.truncation_order > MAX_Q_TRANSFORM_SOURCE_ORDER:
            raise _validation_error(
                "q_prefix_bound",
                "modular q-expansion exceeds the bounded source-prefix envelope",
            )
        if self.space is not None and self.space.weight != self.weight:
            raise _validation_error(
                "weight_parent", "space and q-expansion weight differ"
            )
        if self.space is not None:
            _require_rational_space(self.space, "modular q-expansion")
        return self


class ModularFormSpace(StrictModel):
    """One exact holomorphic or cuspidal modular-form space.

    A space binds its exact Dirichlet character. Operations may support only
    a subset of these structurally representable parents.
    """

    group: Literal["GAMMA0"] = Field(
        default="GAMMA0", description="Congruence subgroup family."
    )
    level: StrictInt = Field(
        ge=1,
        le=MAX_MODULAR_FORM_LEVEL,
        description="Level N of Gamma0(N).",
    )
    weight: StrictInt = Field(
        ge=0,
        le=MAX_MODULAR_FORM_WEIGHT,
        description="Integer modular weight k.",
    )
    kind: Literal["M", "S"] = Field(
        description="Full holomorphic space M_k or cuspidal subspace S_k."
    )
    character: Literal["TRIVIAL"] | DirichletCharacter = Field(
        default="TRIVIAL", description="Trivial character or exact Dirichlet character."
    )
    coefficient_domain: Literal["QQ"] | RationalCyclotomicField = Field(
        default="QQ",
        description=(
            "Exact coefficient field: QQ or the canonical Q[zeta_n] power-basis "
            "presentation. General basis and operator paths use QQ; an explicit "
            "scalar-extension slice supports trivial-character coordinates over "
            "Q(zeta_6)."
        ),
    )

    @model_validator(mode="after")
    def require_character_coefficient_compatibility(self) -> Self:
        """Check that the declared field contains the exact character values.

        Character modulus equality is intentional for this representation
        slice: callers must explicitly inflate a character to the Gamma0 level.
        For cyclotomic coefficient fields, requiring the character value order
        to divide the field's root order gives a canonical explicit embedding.
        """

        if self.character == "TRIVIAL":
            character_order = 1
        else:
            character = self.character
            if not isinstance(character, DirichletCharacter):
                raise _validation_error(
                    "character_type",
                    "character must be TRIVIAL or a Dirichlet character",
                )
            if type(character.group.modulus) is not int:
                raise _validation_error(
                    "character_modulus", "character modulus must be a strict integer"
                )
            if character.group.modulus != self.level:
                raise _validation_error(
                    "character_level_mismatch",
                    "character modulus must equal the Gamma0 level; inflate it explicitly first",
                )
            if (
                type(character.coordinates) is not tuple
                or type(character.group.generator_orders) is not tuple
                or len(character.coordinates) != len(character.group.generator_orders)
                or any(
                    type(coordinate) is not int
                    or type(order) is not int
                    or order <= 0
                    or not 0 <= coordinate < order
                    for coordinate, order in zip(
                        character.coordinates,
                        character.group.generator_orders,
                        strict=False,
                    )
                )
            ):
                raise _validation_error(
                    "character_coordinates", "character coordinates must be canonical"
                )
            # The group decomposition is caller-supplied mathematical data.
            # Validate it before relying on the claimed generator orders.
            try:
                from jacobian.math.number_theory.characters.operations import (
                    require_complete_character_group,
                )

                require_complete_character_group(character.group)
            except (ValueError, TypeError) as error:
                raise _validation_error(
                    "character_group", "character unit-group presentation is invalid"
                ) from error
            from math import gcd, lcm

            character_order = 1
            for coordinate, order in zip(
                character.coordinates, character.group.generator_orders, strict=True
            ):
                character_order = lcm(character_order, order // gcd(coordinate, order))

        if self.coefficient_domain == "QQ":
            if character_order > 2:
                raise _validation_error(
                    "character_coefficient_field",
                    "QQ coefficients cannot contain non-rational character values",
                )
            return self

        field = self.coefficient_domain
        if (
            not isinstance(field, RationalCyclotomicField)
            or type(field.order) is not int
            or field.order <= 2
            or field.order > MAX_MODULAR_FORM_COEFFICIENT_FIELD_ORDER
            or field.degree > MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE
        ):
            raise _validation_error(
                "coefficient_field_bound",
                "cyclotomic coefficient fields require canonical order in [3, "
                f"{MAX_MODULAR_FORM_COEFFICIENT_FIELD_ORDER}] and degree at most "
                f"{MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE}",
            )
        if character_order > 2 and field.order % character_order:
            raise _validation_error(
                "character_coefficient_field",
                "cyclotomic coefficient field must contain the character value field",
            )
        return self


class ModularFormSpaceInclusion(StrictModel):
    """The natural inclusion from one rational trivial-character Gamma0 space.

    For ``source.level | target.level``, the subgroup inclusion is
    ``Gamma0(target.level) <= Gamma0(source.level)``. This carrier only
    represents the same-weight, same-coefficient-parent map; it does not
    encode arbitrary character or field maps.
    """

    map_kind: Literal["natural_gamma0_level_inclusion"] = (
        "natural_gamma0_level_inclusion"
    )
    source_space: ModularFormSpace
    target_space: ModularFormSpace

    @model_validator(mode="after")
    def require_supported_inclusion(self) -> Self:
        issue = natural_gamma0_inclusion_issue(
            self.map_kind, self.source_space, self.target_space
        )
        if issue is not None:
            reason, message = issue
            raise _validation_error(reason, message)
        return self


class ModularCharacterSpaceInclusion(StrictModel):
    """Same-weight inclusion along Gamma0 level and Dirichlet-character inflation.

    Source and target use identical coefficient parents. The character values
    themselves are bound to their respective levels by ``ModularFormSpace``;
    the inflation relation is established by the constructing operation and
    must be rechecked by any consumer that relies on this authored map.
    """

    map_kind: Literal["gamma0_character_inflation"] = "gamma0_character_inflation"
    source_space: ModularFormSpace
    target_space: ModularFormSpace

    @model_validator(mode="after")
    def require_structural_inclusion(self) -> Self:
        source = self.source_space
        target = self.target_space
        if (
            source.group != "GAMMA0"
            or target.group != "GAMMA0"
            or not isinstance(source.character, DirichletCharacter)
            or not isinstance(target.character, DirichletCharacter)
            or type(source.level) is not int
            or type(target.level) is not int
            or source.level > MAX_MODULAR_CHARACTER_INCLUSION_LEVEL
            or target.level > MAX_MODULAR_CHARACTER_INCLUSION_LEVEL
            or target.level % source.level
            or source.weight != target.weight
            or source.kind != target.kind
            or source.coefficient_domain != target.coefficient_domain
        ):
            raise _validation_error(
                "character_inclusion_shape",
                "character inclusion requires nested Gamma0 levels, matching weight, space kind, and coefficient parent",
            )
        return self


def natural_gamma0_inclusion_issue(
    map_kind: str,
    source: ModularFormSpace,
    target: ModularFormSpace,
) -> tuple[str, str] | None:
    """Return why a claimed natural rational Gamma0 inclusion is invalid."""

    if map_kind != "natural_gamma0_level_inclusion":
        return "inclusion_kind_tag", "inclusion map kind is not canonical"
    if type(source) is not ModularFormSpace or type(target) is not ModularFormSpace:
        return "inclusion_space_type", "inclusion parents must be exact modular spaces"
    if (
        source.group != "GAMMA0"
        or target.group != "GAMMA0"
        or source.character != "TRIVIAL"
        or target.character != "TRIVIAL"
        or source.coefficient_domain != "QQ"
        or target.coefficient_domain != "QQ"
    ):
        return (
            "inclusion_parent",
            "natural Gamma0 inclusion requires trivial-character QQ spaces",
        )
    if (
        type(source.level) is not int
        or type(target.level) is not int
        or not 1 <= source.level <= MAX_MODULAR_FORM_LEVEL
        or not 1 <= target.level <= MAX_MODULAR_FORM_LEVEL
        or type(source.weight) is not int
        or type(target.weight) is not int
        or not 0 <= source.weight <= MAX_MODULAR_FORM_WEIGHT
        or not 0 <= target.weight <= MAX_MODULAR_FORM_WEIGHT
        or source.kind not in ("M", "S")
        or target.kind not in ("M", "S")
    ):
        return "inclusion_space_invalid", "inclusion spaces must be canonical"
    if source.weight != target.weight:
        return "inclusion_weight", "source and target weights must agree"
    if target.level % source.level:
        return "inclusion_level", "source Gamma0 level must divide the target level"
    if source.kind == "M" and target.kind == "S":
        return "inclusion_kind", "the full space does not embed into the cusp space"
    return None


class ModularFormBasisElement(StrictModel):
    """One named vector of a deterministic exact modular-form basis."""

    label: str = Field(min_length=1, max_length=96)
    expansion: ModularQExpansion


class ModularFormBasis(StrictModel):
    """A complete q-prefix presentation of one deterministic exact basis."""

    space: ModularFormSpace
    basis_id: Literal[
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
    ]
    precision: StrictInt = Field(ge=1, le=MAX_LEVEL_ONE_BASIS_PRECISION)
    elements: tuple[ModularFormBasisElement, ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )

    @model_validator(mode="after")
    def require_rational_space(self) -> Self:
        _require_rational_space(self.space, "modular-form basis")
        return self


class ModularFormCoordinates(StrictModel):
    """One exact form in an admitted basis over its declared coefficient field."""

    space: ModularFormSpace
    basis_id: Literal[
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
        "gamma0-13-even-order6-character-sturm-v1",
    ]
    coordinates: tuple[CanonicalRational | RationalCyclotomicElement, ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )

    @model_validator(mode="after")
    def require_coefficient_parent(self) -> Self:
        field = self.space.coefficient_domain
        if field == "QQ":
            if any(
                not isinstance(value, CanonicalRational) for value in self.coordinates
            ):
                raise _validation_error(
                    "coordinate_scalar_parent",
                    "QQ modular-form coordinates must be rational scalars",
                )
            return self
        if any(
            not isinstance(value, RationalCyclotomicElement) or value.field != field
            for value in self.coordinates
        ):
            raise _validation_error(
                "coordinate_scalar_parent",
                "cyclotomic coordinates must belong to the declared space coefficient field",
            )
        return self


class ModularFormFieldQExpansion(StrictModel):
    """A finite exact q-prefix in a modular-form coefficient parent."""

    space: ModularFormSpace
    coefficients: tuple[RationalCyclotomicElement, ...] = Field(
        min_length=1, max_length=MAX_LEVEL_ONE_BASIS_PRECISION
    )

    @model_validator(mode="after")
    def require_coefficient_parent(self) -> Self:
        field = self.space.coefficient_domain
        if type(field) is not RationalCyclotomicField:
            raise _validation_error(
                "field_q_expansion_parent",
                "field q-expansions require a cyclotomic coefficient space",
            )
        if any(value.field != field for value in self.coefficients):
            raise _validation_error(
                "field_q_coefficient_field",
                "every q coefficient must belong to the space coefficient field",
            )
        return self


class ModularFormChangeOfBasisFrame(StrictModel):
    """A labeled rational basis expressed in one canonical modular basis.

    ``entries[row][column]`` is the canonical coordinate of the framed basis
    vector named by ``labels[column]`` along the canonical basis vector named
    by ``source_labels[row]``. Invertibility is checked by coordinate
    conversion, where it is needed.
    """

    space: ModularFormSpace
    source_basis_id: Literal[
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
    ]
    source_labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )

    @model_validator(mode="after")
    def require_labeled_square_matrix(self) -> Self:
        _require_rational_space(self.space, "rational basis frame")
        size = len(self.source_labels)
        if (
            len(self.labels) != size
            or len(set(self.source_labels)) != size
            or len(set(self.labels)) != size
            or len(self.entries) != size
            or any(len(row) != size for row in self.entries)
            or any(
                not label or len(label) > 96
                for label in (*self.source_labels, *self.labels)
            )
        ):
            raise _validation_error(
                "basis_frame_shape",
                "change-of-basis frame requires unique ordered labels and a nonempty square matrix",
            )
        return self


class ModularFormFramedCoordinates(StrictModel):
    """Exact coordinates of a form in a source-bound change-of-basis frame."""

    frame: ModularFormChangeOfBasisFrame
    coordinates: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )


class ModularFormHeckeMatrix(StrictModel):
    """Exact T_n matrix in one labeled modular-form basis.

    ``entries[row][column]`` is the coefficient of ``row_labels[row]`` in
    T_n applied to the basis vector ``column_labels[column]``.
    """

    space: ModularFormSpace
    basis_id: Literal[
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
    ]
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)
    row_labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    column_labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )

    @model_validator(mode="after")
    def require_square_labeled_matrix(self) -> Self:
        _require_rational_space(self.space, "rational Hecke matrix")
        size = len(self.row_labels)
        if (
            self.column_labels != self.row_labels
            or len(self.entries) != size
            or any(len(row) != size for row in self.entries)
        ):
            raise _validation_error(
                "hecke_matrix_shape",
                "Hecke matrix must be square with identical ordered row and column basis labels",
            )
        return self


class ModularFormFramedHeckeMatrix(StrictModel):
    """Exact Hecke matrix conjugated into one source-bound rational frame.

    ``entries[row][column]`` is the coefficient of the row-labeled framed
    basis vector in ``T_n`` applied to the column-labeled framed basis vector.
    """

    frame: ModularFormChangeOfBasisFrame
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)
    row_labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    column_labels: tuple[str, ...] = Field(max_length=MAX_LEVEL_ONE_BASIS_COORDINATES)
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_LEVEL_ONE_BASIS_COORDINATES
    )

    @model_validator(mode="after")
    def require_frame_bound_matrix(self) -> Self:
        labels = self.frame.labels
        size = len(labels)
        if (
            self.row_labels != labels
            or self.column_labels != labels
            or len(self.entries) != size
            or any(len(row) != size for row in self.entries)
        ):
            raise _validation_error(
                "framed_hecke_matrix_shape",
                "Hecke matrix axes must match the exact frame labels",
            )
        return self


def _is_prime_level(value: object) -> bool:
    if type(value) is not int or value < 2:
        return False
    return all(value % divisor for divisor in range(2, isqrt(value) + 1))


class ModularFormOperatorImage(StrictModel):
    """Exact U_p or V_p image of a coordinate-defined level-one form."""

    source_form: ModularFormCoordinates
    operator: Literal["U", "V"]
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)
    codomain: ModularFormSpace

    @model_validator(mode="after")
    def require_operator_domain_and_codomain(self) -> Self:
        _require_rational_space(self.source_form.space, "rational operator source")
        _require_rational_space(self.codomain, "rational operator image")
        if self.source_form.space.level != 1:
            raise _validation_error(
                "operator_source_level",
                "coordinate U_p and V_p images currently require level one",
            )
        if not _is_prime_level(self.prime):
            raise _validation_error("operator_prime", "operator index must be prime")
        if (
            self.codomain.level != self.prime
            or self.codomain.weight != self.source_form.space.weight
            or self.codomain.kind != self.source_form.space.kind
        ):
            raise _validation_error(
                "operator_codomain",
                "operator codomain must be the matching Gamma0(prime) space",
            )
        return self


class ModularFormOperatorImagePrefix(StrictModel):
    """Finite exact q-prefix retaining its exact operator-image parent."""

    image: ModularFormOperatorImage
    q_expansion: TruncatedSeries

    @model_validator(mode="after")
    def require_q_prefix(self) -> Self:
        if self.q_expansion.variable != "q":
            raise _validation_error(
                "operator_prefix_variable", "operator prefixes use q"
            )
        if (
            not 1
            <= self.q_expansion.truncation_order
            <= MAX_Q_TRANSFORM_OUTPUT_PRECISION
        ):
            raise _validation_error(
                "operator_prefix_precision",
                "operator prefix exceeds its precision envelope",
            )
        return self


__all__ = [
    "MAX_GAMMA0_THREE_BASIS_PRECISION",
    "MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS",
    "MAX_LEVEL_ONE_BASIS_COORDINATES",
    "MAX_LEVEL_ONE_BASIS_PRECISION",
    "MAX_LEVEL_ONE_BASIS_WEIGHT",
    "MAX_MODULAR_CHARACTER_INCLUSION_LEVEL",
    "MAX_MODULAR_FORM_LEVEL",
    "MAX_MODULAR_FORM_WEIGHT",
    "LevelOneModularQExpansion",
    "ModularCharacterSpaceInclusion",
    "ModularFormBasis",
    "ModularFormBasisElement",
    "ModularFormCoordinates",
    "ModularFormFramedHeckeMatrix",
    "ModularFormOperatorImage",
    "ModularFormOperatorImagePrefix",
    "ModularFormSpace",
    "ModularFormSpaceInclusion",
    "ModularQExpansion",
]
