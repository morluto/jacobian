"""Wire contracts for exact bounded modular-form spaces and transforms."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.kernel import NamedLevelOneModularForm
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_LEVEL_ONE_BASIS_COORDINATES,
    MAX_LEVEL_ONE_BASIS_PRECISION,
    MAX_MODULAR_FORM_WEIGHT,
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
    ModularFormChangeOfBasisFrame,
    ModularFormCoordinates,
    ModularFormFramedCoordinates,
    ModularFormOperatorImage,
    ModularFormSpace,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"modular_forms.{reason}", message)


class LevelOneNamedQExpansionRequest(StrictModel):
    """Construct one reviewed normalized level-one modular-form q-prefix."""

    form: NamedLevelOneModularForm = Field(
        description="Closed normalized family: E4, E6, or Ramanujan DELTA."
    )
    truncation_order: StrictInt = Field(
        ge=1,
        description=(
            "Return coefficients q^0 through q^(P-1); exact work and retained "
            "coefficient cardinality bound P before finite-series arithmetic."
        ),
    )


class Gamma0DimensionSpaceInput(StrictModel):
    """Operation request projection with its exact scalar admission limits."""

    group: Literal["GAMMA0"] = "GAMMA0"
    level: StrictInt = Field(ge=1, le=MAX_GAMMA0_OPERATION_LEVEL)
    weight: StrictInt = Field(ge=0, le=MAX_MODULAR_FORM_WEIGHT)
    kind: Literal["M", "S"]
    character: Literal["TRIVIAL"] | DirichletCharacter = "TRIVIAL"
    coefficient_domain: Literal["QQ"] = "QQ"

    def as_value(self) -> ModularFormSpace:
        """Create the canonical reusable space value from validated fields."""

        return ModularFormSpace.model_construct(
            group=self.group,
            level=self.level,
            weight=self.weight,
            kind=self.kind,
            character=self.character,
            coefficient_domain=self.coefficient_domain,
        )


class SpaceDimensionRequest(StrictModel):
    """Compute the exact dimension of one supported modular-form space."""

    space: Gamma0DimensionSpaceInput = Field(
        description=(
            "Holomorphic M_k or cuspidal S_k on Gamma0(N) over QQ with "
            "trivial character, or the exact chi_-4 character at level 4 "
            "and weight 1 or 3; this operation admits levels 1 through "
            "10,000 for trivial character."
        ),
    )


class ModularFormBasisRequest(StrictModel):
    """Construct a deterministic finite q-prefix basis of an exact space."""

    space: ModularFormSpace
    precision: StrictInt = Field(ge=1, le=MAX_LEVEL_ONE_BASIS_PRECISION)


class ModularFormCoordinatesQExpansionRequest(StrictModel):
    """Construct the finite q-prefix of one exact coordinate vector."""

    form: ModularFormCoordinates
    precision: StrictInt = Field(ge=1, le=MAX_LEVEL_ONE_BASIS_PRECISION)


class ModularFormCoordinatesProductRequest(StrictModel):
    """Multiply two exact forms in supported trivial-character QQ spaces."""

    left: ModularFormCoordinates
    right: ModularFormCoordinates


class ModularFormCoordinatesFieldExtensionRequest(StrictModel):
    """Extend one rational coordinate value to an explicit cyclotomic parent."""

    form: ModularFormCoordinates
    coefficient_field: RationalCyclotomicField


class ModularFormCoordinatesTransportRequest(StrictModel):
    """Transport exact coordinates along a nested trivial-character Gamma0 inclusion."""

    form: ModularFormCoordinates
    target_space: ModularFormSpace


class ModularFormCoordinatesHeckeRequest(StrictModel):
    """Apply a bounded level-one Hecke operator to exact coordinates."""

    form: ModularFormCoordinates
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)


class ModularFormCoordinatesAtkinLehnerRequest(StrictModel):
    """Apply one exact Atkin-Lehner involution to represented coordinates."""

    form: ModularFormCoordinates
    divisor: StrictInt = Field(ge=1, le=MAX_GAMMA0_OPERATION_LEVEL)


class ModularFormAtkinLehnerTargetRequest(StrictModel):
    """Identify a full Fricke target parent, without applying the slash action."""

    space: ModularFormSpace


class ModularFormHeckeMatrixRequest(StrictModel):
    """Compute the exact matrix of T_n in a canonical represented basis."""

    space: ModularFormSpace
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)


class ModularFormFramedHeckeMatrixRequest(StrictModel):
    """Compute a Hecke matrix directly in one admitted rational basis frame."""

    frame: ModularFormChangeOfBasisFrame
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)


class ModularFormCoordinatesU2Request(StrictModel):
    """Apply bounded U_2 to a represented Gamma0(2) or Gamma0(4) space."""

    form: ModularFormCoordinates


class ModularFormCoordinatesUPrimeRequest(StrictModel):
    """Apply U_p in a represented Gamma0 space when p divides its level."""

    form: ModularFormCoordinates
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)


class ModularFormCoordinatesV2Request(StrictModel):
    """Apply V_2 from SL2Z to Gamma0(2), or Gamma0(2) to Gamma0(4)."""

    form: ModularFormCoordinates


class ModularFormCoordinatesV3Request(StrictModel):
    """Apply V_3 from level one to the exact Gamma0(3) basis."""

    form: ModularFormCoordinates


class ModularFormCoordinatesVDegeneracyRequest(StrictModel):
    """Apply V_d from Gamma0(M) to Gamma0(Md) in an admitted exact basis."""

    form: ModularFormCoordinates
    d: StrictInt = Field(ge=1, le=MAX_GAMMA0_OPERATION_LEVEL)


class ModularFormBasisFrameRequest(StrictModel):
    """Declare a rational basis in coordinates of a supported canonical basis."""

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
                "change-of-basis frame requires unique ordered labels and a square matrix, including the empty frame",
            )
        return self

    def as_frame(self) -> ModularFormChangeOfBasisFrame:
        return ModularFormChangeOfBasisFrame(
            space=self.space,
            source_basis_id=self.source_basis_id,
            source_labels=self.source_labels,
            labels=self.labels,
            entries=self.entries,
        )


class ModularFormCanonicalToFramedRequest(StrictModel):
    frame: ModularFormChangeOfBasisFrame
    form: ModularFormCoordinates


class ModularFormFramedToCanonicalRequest(StrictModel):
    form: ModularFormFramedCoordinates


class ModularFormOperatorImageRequest(StrictModel):
    """Bind U_p or V_p to a level-one coordinate-defined form."""

    source_form: ModularFormCoordinates
    operator: Literal["U", "V"]
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)


class ModularFormOperatorImagePrefixRequest(StrictModel):
    """Evaluate a bounded q-prefix of an exact U_p or V_p image."""

    image: ModularFormOperatorImage
    precision: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_OUTPUT_PRECISION)


class SpaceDimensionResult(StrictModel):
    """Exact dimension of a source space with its Gamma0 geometric data.

    For the declared space, ``dimension`` is the exact complex dimension;
    ``eisenstein_dimension`` and ``cusp_dimension`` are the explicit
    projections of the same space family (the Eisenstein subspace of the
    ambient ``M_k`` and the cuspidal subspace). For ``kind == "M"`` the
    dimension is their sum; for ``kind == "S"`` it equals the cusp
    projection. No q-expansion coefficients are carried.
    """

    space: ModularFormSpace
    dimension: StrictInt = Field(ge=0)
    index: StrictInt = Field(ge=1, description="[SL2(Z): Gamma0(level)].")
    genus: StrictInt = Field(
        ge=0, description="Genus of the compact modular curve X0(level)."
    )
    cusp_count: StrictInt = Field(ge=1, description="Number of Gamma0(level) cusps.")
    elliptic_points_order_2: StrictInt = Field(ge=0)
    elliptic_points_order_3: StrictInt = Field(ge=0)
    eisenstein_dimension: StrictInt = Field(ge=0)
    cusp_dimension: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_dimension_decomposition(self) -> Self:
        if self.space.coefficient_domain != "QQ":
            raise _validation_error(
                "rational_dimension_result_parent",
                "this dimension result carrier currently represents QQ spaces only",
            )
        if self.space.kind == "M":
            expected = self.eisenstein_dimension + self.cusp_dimension
        else:
            expected = self.cusp_dimension
        if self.dimension != expected:
            raise _validation_error(
                "dimension_decomposition",
                "dimension must decompose into the reported Eisenstein/cusp data",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        space: ModularFormSpace,
        *,
        dimension: int,
        eisenstein_dimension: int,
        cusp_dimension: int,
        index: int,
        genus: int,
        cusp_count: int,
        elliptic_points_order_2: int,
        elliptic_points_order_3: int,
    ) -> Self:
        """Build one result after the admitted kernel established its count."""

        return cls.model_construct(
            space=space,
            dimension=dimension,
            eisenstein_dimension=eisenstein_dimension,
            cusp_dimension=cusp_dimension,
            index=index,
            genus=genus,
            cusp_count=cusp_count,
            elliptic_points_order_2=elliptic_points_order_2,
            elliptic_points_order_3=elliptic_points_order_3,
        )


__all__ = [
    "Gamma0DimensionSpaceInput",
    "LevelOneNamedQExpansionRequest",
    "ModularFormBasisRequest",
    "ModularFormCoordinatesHeckeRequest",
    "ModularFormCoordinatesQExpansionRequest",
    "ModularFormCoordinatesU2Request",
    "ModularFormCoordinatesUPrimeRequest",
    "ModularFormCoordinatesV2Request",
    "ModularFormCoordinatesV3Request",
    "ModularFormFramedHeckeMatrixRequest",
    "ModularFormHeckeMatrixRequest",
    "ModularFormOperatorImagePrefixRequest",
    "ModularFormOperatorImageRequest",
    "SpaceDimensionRequest",
    "SpaceDimensionResult",
]
