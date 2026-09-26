"""Exact bounded classification of generic-j finite-field curve twists."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields.values import FiniteFieldElement
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    MAX_FINITE_FIELD_ISOMORPHISM_ORDER,
    MAX_FINITE_FIELD_ISOMORPHISM_WORK,
    MAX_FINITE_FIELD_TWIST_ORDER,
    MAX_FINITE_FIELD_TWIST_WORK,
    FiniteFieldIsomorphismResult,
    FiniteFieldQuadraticTwistRelation,
    FiniteFieldShortWeierstrassCurve,
    _coordinates,
    _curve_admit,
    _multiply,
    _power,
    finite_field_discriminant,
    finite_field_isomorphism,
    finite_field_quadratic_twist_relation,
)

MAX_FINITE_FIELD_TWIST_CLASS_WORK = 12_000_000


class FiniteFieldTwistClassResult(StrictModel):
    """Classify two generic-j curves over one exact finite-field presentation.

    ``DIFFERENT_J`` is an exact invariant obstruction. For equal generic j,
    the two twist classes are distinguished by the complete model-isomorphism
    search; a quadratic-twist result carries the canonical twist and its map
    to the target.
    """

    source: FiniteFieldShortWeierstrassCurve
    target: FiniteFieldShortWeierstrassCurve
    relation: Literal["DIFFERENT_J", "ISOMORPHIC", "QUADRATIC_TWIST"]
    source_j_invariant: FiniteFieldElement
    target_j_invariant: FiniteFieldElement
    isomorphism: FiniteFieldIsomorphismResult | None = None
    twist: FiniteFieldQuadraticTwistRelation | None = None
    twist_to_target: FiniteFieldIsomorphismResult | None = None

    @model_validator(mode="after")
    def require_consistent_witness_shape(self) -> Self:
        if self.source.field != self.target.field:
            raise ValueError(
                "twist-class curves must share one exact field presentation"
            )
        if (
            self.source.field.characteristic**self.source.field.degree
            > MAX_FINITE_FIELD_TWIST_ORDER
        ):
            raise ValueError(
                "twist-class result exceeds its admitted finite-field order"
            )
        if (
            self.source_j_invariant.presentation != self.source.field
            or self.target_j_invariant.presentation != self.source.field
        ):
            raise ValueError("twist-class invariants must use the common curve field")
        if self.source_j_invariant == self.target_j_invariant and (
            not _nonzero(self.source.coefficient_a)
            or not _nonzero(self.source.coefficient_b)
        ):
            raise ValueError(
                "equal-j twist-class results require generic j outside {0, 1728}"
            )
        if self.relation == "DIFFERENT_J":
            valid = (
                self.source_j_invariant != self.target_j_invariant
                and self.isomorphism is None
                and self.twist is None
                and self.twist_to_target is None
            )
        elif self.relation == "ISOMORPHIC":
            valid = (
                self.source_j_invariant == self.target_j_invariant
                and self.isomorphism is not None
                and self.isomorphism.source == self.source
                and self.isomorphism.target == self.target
                and self.isomorphism.isomorphic
                and _valid_isomorphism_witness(self.isomorphism)
                and self.twist is None
                and self.twist_to_target is None
            )
        else:
            valid = (
                self.source_j_invariant == self.target_j_invariant
                and self.isomorphism is None
                and self.twist is not None
                and self.twist.source_curve == self.source
                and self.twist_to_target is not None
                and self.twist_to_target.source == self.twist.twisted_curve
                and self.twist_to_target.target == self.target
                and self.twist_to_target.isomorphic
                and _valid_twist_witness(self.twist)
                and _valid_isomorphism_witness(self.twist_to_target)
            )
        if not valid:
            raise ValueError(
                "twist-class result has inconsistent invariants or witnesses"
            )
        return self


def _nonzero(element) -> bool:
    return any(_coordinates(element))


def _valid_isomorphism_witness(result: FiniteFieldIsomorphismResult) -> bool:
    if not result.isomorphic or result.scaling is None:
        return False
    field = result.source.field
    if field.characteristic**field.degree > MAX_FINITE_FIELD_ISOMORPHISM_ORDER:
        return False
    u = _coordinates(result.scaling)
    if not any(u):
        return False
    u2 = _multiply(field, u, u)
    u4 = _multiply(field, u2, u2)
    u6 = _multiply(field, u4, u2)
    return _multiply(
        field, u4, _coordinates(result.source.coefficient_a)
    ) == _coordinates(result.target.coefficient_a) and _multiply(
        field, u6, _coordinates(result.source.coefficient_b)
    ) == _coordinates(result.target.coefficient_b)


def _valid_twist_witness(relation: FiniteFieldQuadraticTwistRelation) -> bool:
    field = relation.source_curve.field
    q = field.characteristic**field.degree
    if q > MAX_FINITE_FIELD_TWIST_ORDER or relation.twisted_curve.field != field:
        return False
    d = _coordinates(relation.parameter)
    if not any(d):
        return False
    minus_one = (field.characteristic - 1,) + (0,) * (field.degree - 1)
    if _power(field, d, (q - 1) // 2) != minus_one:
        return False
    d2 = _multiply(field, d, d)
    d3 = _multiply(field, d2, d)
    return _multiply(
        field, d2, _coordinates(relation.source_curve.coefficient_a)
    ) == _coordinates(relation.twisted_curve.coefficient_a) and _multiply(
        field, d3, _coordinates(relation.source_curve.coefficient_b)
    ) == _coordinates(relation.twisted_curve.coefficient_b)


def _admit_pair(
    source: FiniteFieldShortWeierstrassCurve,
    target: FiniteFieldShortWeierstrassCurve,
) -> tuple[
    FiniteFieldShortWeierstrassCurve,
    FiniteFieldShortWeierstrassCurve,
    int,
    FiniteFieldElement,
    FiniteFieldElement,
]:
    if not isinstance(source, FiniteFieldShortWeierstrassCurve) or not isinstance(
        target, FiniteFieldShortWeierstrassCurve
    ):
        raise OperationDomainValidationError(
            location=("source", "target"),
            code="elliptic_curve.finite_field.twist_class_curve_type",
            message="twist-class inputs must be finite-field short-Weierstrass curves",
        )
    source, target = _curve_admit(source), _curve_admit(target)
    if source.field != target.field:
        raise OperationDomainValidationError(
            location=("target", "field"),
            code="elliptic_curve.finite_field.twist_class_field_mismatch",
            message="twist-class decision requires the same exact field presentation",
        )
    require_field(source.field)
    field = source.field
    q = field.characteristic**field.degree
    if q > min(MAX_FINITE_FIELD_ISOMORPHISM_ORDER, MAX_FINITE_FIELD_TWIST_ORDER):
        raise OperationResourceAdmissionError(
            location=("source", "field"),
            code="elliptic_curve.finite_field.twist_class_order_bound",
            message="complete twist-class decision requires field order at most 4096",
        )
    source_data = finite_field_discriminant(
        source.field, source.coefficient_a, source.coefficient_b
    )
    target_data = finite_field_discriminant(
        target.field, target.coefficient_a, target.coefficient_b
    )
    source_j, target_j = source_data.j_invariant, target_data.j_invariant
    if source_j is None or target_j is None:
        raise OperationDomainValidationError(
            location=("source", "target"),
            code="elliptic_curve.finite_field.twist_class_singular",
            message="twist-class inputs must be nonsingular curves",
        )
    if source_j != target_j:
        return source, target, q, source_j, target_j

    iso_work = q * field.degree**2 * (4 + 4 * q.bit_length())
    twist_work = q * field.degree**2 * (2 * q.bit_length() + 8)
    total = 2 * iso_work + twist_work
    if (
        iso_work > MAX_FINITE_FIELD_ISOMORPHISM_WORK
        or twist_work > MAX_FINITE_FIELD_TWIST_WORK
        or total > MAX_FINITE_FIELD_TWIST_CLASS_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("source", "target", "field"),
            code="elliptic_curve.finite_field.twist_class_work_bound",
            message="complete twist-class decision exceeds its exact-work envelope",
        )
    return source, target, q, source_j, target_j


def finite_field_twist_class(
    source: FiniteFieldShortWeierstrassCurve,
    target: FiniteFieldShortWeierstrassCurve,
) -> FiniteFieldTwistClassResult:
    """Decide the twist relation for equal generic j, or reject unequal j.

    For ``p > 3``, curves with ``j`` outside ``{0,1728}`` have exactly two
    isomorphism classes over a finite field with that geometric isomorphism
    type: the class of the source and its nontrivial quadratic twist. The
    exhaustive scaling and twist searches are bounded by the field order.
    """
    source, target, _, source_j, target_j = _admit_pair(source, target)
    if source_j != target_j:
        return FiniteFieldTwistClassResult(
            source=source,
            target=target,
            relation="DIFFERENT_J",
            source_j_invariant=source_j,
            target_j_invariant=target_j,
        )
    # For p > 3, j=0 iff A=0 and j=1728 iff B=0. These automorphism-rich
    # classes need their own explicit branch semantics and are intentionally
    # outside this operation's generic-j postcondition.
    if not _nonzero(source.coefficient_a) or not _nonzero(source.coefficient_b):
        raise OperationDomainValidationError(
            location=("source", "target"),
            code="elliptic_curve.finite_field.twist_class_exceptional_j",
            message="twist-class decision currently requires j outside {0, 1728}",
        )
    direct = finite_field_isomorphism(source, target)
    if direct.isomorphic:
        return FiniteFieldTwistClassResult(
            source=source,
            target=target,
            relation="ISOMORPHIC",
            source_j_invariant=source_j,
            target_j_invariant=target_j,
            isomorphism=direct,
        )
    twist = finite_field_quadratic_twist_relation(source)
    twisted_map = finite_field_isomorphism(twist.twisted_curve, target)
    if not twisted_map.isomorphic:
        raise RuntimeError(
            "generic-j equal-invariant curve was neither source nor quadratic twist"
        )
    return FiniteFieldTwistClassResult(
        source=source,
        target=target,
        relation="QUADRATIC_TWIST",
        source_j_invariant=source_j,
        target_j_invariant=target_j,
        twist=twist,
        twist_to_target=twisted_map,
    )


__all__ = ["FiniteFieldTwistClassResult", "finite_field_twist_class"]
