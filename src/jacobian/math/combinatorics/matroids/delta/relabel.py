"""Exact ground-label transport for finite delta-matroids."""

from __future__ import annotations

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)


def _require_target_ground(
    source_ground: tuple[str, ...], target_ground: object
) -> None:
    if type(target_ground) is not tuple or any(
        type(label) is not str for label in target_ground
    ):
        raise PydanticCustomError(
            "delta_matroid.relabel_target_type",
            "target ground labels must be a tuple of strings",
        )
    if len(target_ground) != len(source_ground):
        raise PydanticCustomError(
            "delta_matroid.relabel_axis",
            "target ground labels must cover the source ground exactly",
        )
    if len(set(target_ground)) != len(target_ground):
        raise PydanticCustomError(
            "delta_matroid.relabel_target_duplicate",
            "target ground labels must be unique",
        )
    try:
        label_bytes = sum(len(label.encode("utf-8")) for label in target_ground)
    except UnicodeEncodeError:
        raise PydanticCustomError(
            "delta_matroid.relabel_target_encoding",
            "target ground labels must be UTF-8-representable",
        ) from None
    if label_bytes > MAX_DELTA_LABEL_BYTES:
        raise PydanticCustomError(
            "delta_matroid.relabel_target_bytes",
            f"target ground labels exceed the {MAX_DELTA_LABEL_BYTES}-byte envelope",
        )


class DeltaMatroidRelabelRequest(StrictModel):
    """Map source ground labels position-wise to a new labelled ground set."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Rename every ground element through the total bijection "
                "source.ground[i] -> target_ground[i]. Feasible rows retain "
                "their exact ground positions. Source memberships, source and "
                "target UTF-8 label bytes, and symmetric-exchange candidate "
                "work are admitted under the declared limits."
            ),
            "admission_limits": {
                "max_source_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_source_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_pass": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
                "max_target_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete source delta-matroid, admitted at no more than "
            f"{MAX_DELTA_MEMBERSHIPS} feasible-row memberships, "
            f"{MAX_DELTA_LABEL_BYTES} source-label UTF-8 bytes, and "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} symmetric-exchange candidate "
            "checks per admission or replay pass."
        )
    )
    target_ground: tuple[str, ...] = Field(
        description=(
            "Unique target labels in source-ground order; entry i is the image "
            f"of source ground label i. Aggregate UTF-8 length is bounded by "
            f"{MAX_DELTA_LABEL_BYTES} bytes."
        )
    )

    @model_validator(mode="after")
    def require_bijective_bounded_target(self) -> DeltaMatroidRelabelRequest:
        _require_target_ground(self.delta_matroid.ground, self.target_ground)
        return self


def relabel(
    delta_matroid: FiniteDeltaMatroid, target_ground: tuple[str, ...]
) -> FiniteDeltaMatroid:
    """Return the same feasible family under a total ground-label bijection."""

    if type(delta_matroid) is not FiniteDeltaMatroid:
        raise TypeError("delta_matroid must be a canonical FiniteDeltaMatroid")
    _require_target_ground(delta_matroid.ground, target_ground)
    try:
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source is not a structurally valid finite delta-matroid"
        ) from exc
    require_delta_matroid_admission(system)
    if first_symmetric_exchange_obstruction(system) is not None:
        raise ValueError("source feasible family fails symmetric exchange")

    # A ground bijection does not change any indexed feasible subset or its
    # symmetric differences, so the complete axiom replay transfers exactly.
    return FiniteDeltaMatroid._from_kernel(
        FiniteFeasibleSetSystem(ground=target_ground, feasible=system.feasible)
    )


__all__ = ["DeltaMatroidRelabelRequest", "relabel"]
