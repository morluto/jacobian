"""Independent checker declaration for projective arrangements."""

from collections.abc import Callable

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import ProviderObservation
from jacobian.contracts.projective_geometry import ProjectiveLineArrangementRequest
from jacobian.providers import flint_runtime


def _projective_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return flint_runtime.projective_arrangement_checker_provider_runtime(
        checker_ids=checker_ids
    )


def _at_most_lines(maximum: int) -> Callable[[object], bool]:
    def supports(payload: object) -> bool:
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("lines"), list)
            and len(payload["lines"]) <= maximum
        )

    return supports


PROJECTIVE_GEOMETRY_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "geometry.projective_line_arrangement.flats.materialize",
        ProjectiveLineArrangementRequest,
        "check_projective_line_arrangement_flats",
        "geometry.projective-line-arrangement.flats.exhaustive-replay",
        entrypoint_module="jacobian_checkers.projective_arrangements",
        observation_loader=_projective_runtime,
        supports_input=_at_most_lines(64),
        replay_method="exact projective pair-incidence exhaustive replay",
        reason=(
            "operator-authorized standard-library checker independently normalizes "
            "the lines, rebuilds every pair intersection and recovers all incidences"
        ),
        verification_operation_id=("geometry.projective_line_arrangement.flats.verify"),
        verification_title="Verify projective line-arrangement flats",
        verification_description=(
            "Independently rebuild every exact projective intersection, full "
            "incidence group, multiplicity and line-pair accounting identity from "
            "one stored materialization."
        ),
        verification_tags=(
            "verification",
            "exact",
            "geometry",
            "projective",
            "line-arrangement",
            "incidence",
        ),
    ),
)

__all__ = ["PROJECTIVE_GEOMETRY_AUTHORIZED_CHECKERS"]
