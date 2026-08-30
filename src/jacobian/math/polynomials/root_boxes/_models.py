"""Typed contracts for exact boxed real-root certification."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json, sha256_digest
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.maps.values import RationalPolynomialMap

MAX_ROOT_BOX_DIMENSION = 5
MAX_ROOT_BOX_COMPONENT_TERMS = 64
MAX_ROOT_BOX_AGGREGATE_TERMS = 256
MAX_ROOT_BOX_TOTAL_DEGREE = 8
MAX_ROOT_BOX_ENDPOINT_DIGITS = 128
MAX_ROOT_BOX_POINT_VALUE_DIGITS = 512
MAX_ROOT_BOX_ENCLOSURE_DIGITS = 2_048
MAX_ROOT_BOX_INTERMEDIATE_DIGITS = 65_536
MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS = 32_768
MAX_ROOT_BOX_SOURCE_BYTES = 512 * 1_024
MAX_ROOT_BOX_RESULT_BYTES = CanonicalLimits().max_output_bytes

ROOT_BOX_ADMISSION_SUMMARY = (
    f"Bounds: square systems through dimension {MAX_ROOT_BOX_DIMENSION}; "
    f"{MAX_ROOT_BOX_COMPONENT_TERMS} terms per component and "
    f"{MAX_ROOT_BOX_AGGREGATE_TERMS} aggregate terms; total degree "
    f"{MAX_ROOT_BOX_TOTAL_DEGREE}; {MAX_ROOT_BOX_ENDPOINT_DIGITS}-digit box "
    f"endpoint components; {MAX_ROOT_BOX_POINT_VALUE_DIGITS}-digit admitted "
    f"point values; {MAX_ROOT_BOX_ENCLOSURE_DIGITS:,}-digit scalar interval "
    f"enclosures; {MAX_ROOT_BOX_INTERMEDIATE_DIGITS:,}-digit interval-matrix "
    f"intermediates; {MAX_ROOT_BOX_SOURCE_BYTES:,}-byte retained source; and "
    f"{MAX_ROOT_BOX_RESULT_BYTES:,}-byte canonical result."
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.root_box_invariant", message)


def _source_payload(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> dict[str, object]:
    return {
        "polynomial_map": polynomial_map.model_dump(mode="json"),
        "box": box.model_dump(mode="json"),
    }


def root_box_source_digest(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> Sha256Digest:
    """Bind one ordered polynomial system to one exact rational box."""

    return sha256_digest(encode_strict_json(_source_payload(polynomial_map, box)))


class PolynomialSystemRootBoxRequest(StrictModel):
    """Certify one bounded square polynomial system on one rational box."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Certify a square ordered QQ polynomial system on a complete "
                "closed rational box. The map input axis and box axis must agree "
                "exactly. The deterministic method uses the exact inverse of the "
                "Jacobian at the box midpoint; failure to prove exclusion or "
                "strict Krawczyk inclusion returns UNKNOWN. "
                f"{ROOT_BOX_ADMISSION_SUMMARY}"
            )
        }
    )

    polynomial_map: RationalPolynomialMap = Field(
        description=(
            "Ordered square QQ polynomial system. Its output-polynomial order is "
            "the equation axis, and every component uses input_variables exactly."
        )
    )
    box: RationalBox = Field(
        description=(
            "Complete closed QQ box in exactly polynomial_map.input_variables order. "
            "Point and boundary-root boxes are accepted but cannot satisfy strict "
            "interior inclusion."
        )
    )


type RootBoxIntervalRow = Annotated[
    tuple[ClosedRationalInterval, ...],
    Field(min_length=1, max_length=MAX_ROOT_BOX_DIMENSION),
]


class RootBoxJacobianEnclosure(StrictModel):
    """A nonempty square row-major interval enclosure of a Jacobian matrix."""

    entries: tuple[RootBoxIntervalRow, ...] = Field(
        min_length=1,
        max_length=MAX_ROOT_BOX_DIMENSION,
        description=(
            "Rows follow the source system's output order; columns follow its "
            "ordered input-variable axis."
        ),
    )

    @model_validator(mode="after")
    def require_square_shape(self) -> Self:
        order = len(self.entries)
        if any(len(row) != order for row in self.entries):
            raise _validation_error("Jacobian enclosure must be square")
        return self


class RootBoxMidpointData(StrictModel):
    """Exact midpoint value and derivative data for one certification attempt."""

    center: VariablePoint
    value_at_center: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ROOT_BOX_DIMENSION
    )
    jacobian_at_center: RationalMatrix
    jacobian_enclosure: RootBoxJacobianEnclosure

    @model_validator(mode="after")
    def require_one_square_system(self) -> Self:
        order = len(self.center.values)
        if len(self.value_at_center) != order:
            raise _validation_error(
                "midpoint value count must match the coordinate dimension"
            )
        if len(self.jacobian_at_center.entries) != order or any(
            len(row) != order for row in self.jacobian_at_center.entries
        ):
            raise _validation_error(
                "midpoint Jacobian shape must match the coordinate dimension"
            )
        if len(self.jacobian_enclosure.entries) != order:
            raise _validation_error(
                "Jacobian enclosure shape must match the coordinate dimension"
            )
        return self


class RootBoxKrawczykEvidence(RootBoxMidpointData):
    """Exact data defining one midpoint-preconditioned Krawczyk image."""

    preconditioner: RationalMatrix = Field(
        description="Exact two-sided inverse of jacobian_at_center over QQ."
    )
    krawczyk_image: RationalBox

    @model_validator(mode="after")
    def require_krawczyk_axes_and_shape(self) -> Self:
        order = len(self.center.values)
        if len(self.preconditioner.entries) != order or any(
            len(row) != order for row in self.preconditioner.entries
        ):
            raise _validation_error(
                "preconditioner shape must match the coordinate dimension"
            )
        if self.krawczyk_image.variables != self.center.variables:
            raise _validation_error(
                "Krawczyk image and center must use the same ordered axis"
            )
        return self


class RootBoxComponentExclusion(StrictModel):
    """One component range excludes zero on the complete source box."""

    method: Literal["COMPONENT_RANGE_EXCLUSION"] = "COMPONENT_RANGE_EXCLUSION"
    component_index: int = Field(ge=0, le=MAX_ROOT_BOX_DIMENSION - 1, strict=True)
    enclosure: ClosedRationalInterval


class RootBoxKrawczykDisjointness(StrictModel):
    """The exact Krawczyk image is disjoint from the complete source box."""

    method: Literal["KRAWCZYK_DISJOINTNESS"] = "KRAWCZYK_DISJOINTNESS"
    evidence: RootBoxKrawczykEvidence


type RootBoxNoRootEvidence = Annotated[
    RootBoxComponentExclusion | RootBoxKrawczykDisjointness,
    Field(discriminator="method"),
]


class RootBoxCertifiedUniqueNonsingular(StrictModel):
    """Strict Krawczyk inclusion proves one unique nonsingular real zero."""

    status: Literal["CERTIFIED_UNIQUE_NONSINGULAR"] = "CERTIFIED_UNIQUE_NONSINGULAR"
    evidence: RootBoxKrawczykEvidence


class RootBoxNoRoot(StrictModel):
    """Exact complete-box evidence proves that the system has no zero."""

    status: Literal["NO_ROOT"] = "NO_ROOT"
    evidence: RootBoxNoRootEvidence


class RootBoxSingularMidpointAttempt(StrictModel):
    """The deterministic midpoint Jacobian has no exact inverse."""

    kind: Literal["MIDPOINT_JACOBIAN_SINGULAR"] = "MIDPOINT_JACOBIAN_SINGULAR"
    data: RootBoxMidpointData


class RootBoxInconclusiveKrawczykAttempt(StrictModel):
    """The Krawczyk image proves neither strict inclusion nor disjointness."""

    kind: Literal["KRAWCZYK_INCONCLUSIVE"] = "KRAWCZYK_INCONCLUSIVE"
    evidence: RootBoxKrawczykEvidence


type RootBoxUnknownAttempt = Annotated[
    RootBoxSingularMidpointAttempt | RootBoxInconclusiveKrawczykAttempt,
    Field(discriminator="kind"),
]


class RootBoxUnknown(StrictModel):
    """One exact bounded attempt did not establish a root conclusion."""

    status: Literal["UNKNOWN"] = "UNKNOWN"
    attempt: RootBoxUnknownAttempt


type RootBoxConclusion = Annotated[
    RootBoxCertifiedUniqueNonsingular | RootBoxNoRoot | RootBoxUnknown,
    Field(discriminator="status"),
]


def root_box_record_digest(
    source_digest: Sha256Digest,
    conclusion: RootBoxConclusion,
) -> Sha256Digest:
    """Bind the exact returned evidence to its already-bound source."""

    return sha256_digest(
        encode_strict_json(
            {
                "source_digest": source_digest,
                "conclusion": conclusion.model_dump(mode="json"),
            }
        )
    )


def _midpoint_data(conclusion: RootBoxConclusion) -> RootBoxMidpointData | None:
    if isinstance(conclusion, RootBoxCertifiedUniqueNonsingular):
        return conclusion.evidence
    if isinstance(conclusion, RootBoxNoRoot):
        if isinstance(conclusion.evidence, RootBoxKrawczykDisjointness):
            return conclusion.evidence.evidence
        return None
    if isinstance(conclusion.attempt, RootBoxSingularMidpointAttempt):
        return conclusion.attempt.data
    return conclusion.attempt.evidence


class PolynomialSystemRootBoxResult(StrictModel):
    """One source-bound exact conclusion or explicit non-conclusion."""

    polynomial_map: RationalPolynomialMap
    box: RationalBox
    source_digest: Sha256Digest
    conclusion: RootBoxConclusion
    record_digest: Sha256Digest = Field(
        description=(
            "Canonical integrity binding for the retained source and evidence. "
            "It is not an independently supplied proof-verification API."
        )
    )

    @model_validator(mode="after")
    def require_source_and_evidence_binding(self) -> Self:
        if self.source_digest != root_box_source_digest(self.polynomial_map, self.box):
            raise _validation_error(
                "source digest does not bind the polynomial system and box"
            )
        if self.record_digest != root_box_record_digest(
            self.source_digest, self.conclusion
        ):
            raise _validation_error(
                "record digest does not bind the returned exact evidence"
            )
        order = len(self.polynomial_map.input_variables)
        if order > MAX_ROOT_BOX_DIMENSION:
            raise _validation_error(
                "retained polynomial system exceeds the root-box dimension bound"
            )
        if len(self.polynomial_map.output_polynomials) != order:
            raise _validation_error("retained polynomial system must be square")
        if (
            self.box.domain != "QQ"
            or self.box.variables != self.polynomial_map.input_variables
        ):
            raise _validation_error(
                "retained box must use the polynomial system's complete ordered axis"
            )
        if (
            isinstance(self.conclusion, RootBoxNoRoot)
            and isinstance(self.conclusion.evidence, RootBoxComponentExclusion)
            and self.conclusion.evidence.component_index >= order
        ):
            raise _validation_error(
                "excluded component index must belong to the source system"
            )
        data = _midpoint_data(self.conclusion)
        if data is not None and data.center.variables != self.box.variables:
            raise _validation_error(
                "midpoint evidence must use the retained source axis"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial_map: RationalPolynomialMap,
        box: RationalBox,
        conclusion: RootBoxConclusion,
    ) -> Self:
        """Build a result after the admitted kernel established its outcome."""

        source_digest = root_box_source_digest(polynomial_map, box)
        return cls(
            polynomial_map=polynomial_map,
            box=box,
            source_digest=source_digest,
            conclusion=conclusion,
            record_digest=root_box_record_digest(source_digest, conclusion),
        )


__all__ = [
    "MAX_ROOT_BOX_AGGREGATE_TERMS",
    "MAX_ROOT_BOX_COMPONENT_TERMS",
    "MAX_ROOT_BOX_DIMENSION",
    "MAX_ROOT_BOX_ENCLOSURE_DIGITS",
    "MAX_ROOT_BOX_ENDPOINT_DIGITS",
    "MAX_ROOT_BOX_INTERMEDIATE_DIGITS",
    "MAX_ROOT_BOX_POINT_VALUE_DIGITS",
    "MAX_ROOT_BOX_RESULT_BYTES",
    "MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS",
    "MAX_ROOT_BOX_SOURCE_BYTES",
    "MAX_ROOT_BOX_TOTAL_DEGREE",
    "ROOT_BOX_ADMISSION_SUMMARY",
    "PolynomialSystemRootBoxRequest",
    "PolynomialSystemRootBoxResult",
    "RootBoxCertifiedUniqueNonsingular",
    "RootBoxComponentExclusion",
    "RootBoxConclusion",
    "RootBoxInconclusiveKrawczykAttempt",
    "RootBoxJacobianEnclosure",
    "RootBoxKrawczykDisjointness",
    "RootBoxKrawczykEvidence",
    "RootBoxMidpointData",
    "RootBoxNoRoot",
    "RootBoxSingularMidpointAttempt",
    "RootBoxUnknown",
    "root_box_record_digest",
    "root_box_source_digest",
]
