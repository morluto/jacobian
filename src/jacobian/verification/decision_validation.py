"""Admission checks for checker decisions before record publication."""

from __future__ import annotations

from collections.abc import Collection

from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.results import Coverage


class VerificationDecisionValidator:
    """Validate the claim a checker decision is allowed to certify."""

    def rejection_detail(
        self,
        decision: CheckerDecision,
        *,
        permits_bounded_coverage: bool,
    ) -> str | None:
        """Return a caller-safe rejection reason for a non-admissible decision."""

        if not decision.accepted:
            return decision.detail
        if not permits_bounded_coverage and decision.coverage is Coverage.BOUNDED:
            return (
                "Inline exact verification cannot bind a bounded scope; the checker "
                "must report exhaustive or not-applicable coverage."
            )
        return None

    @staticmethod
    def require_request_bound_endpoints(
        decision: CheckerDecision,
        request_artifact_uris: Collection[str],
    ) -> None:
        """Reject checker claims about artifacts absent from its request."""

        request_uris = set(request_artifact_uris)
        decision_endpoints = {
            *decision.relationship_source_artifact_uris,
            *decision.relationship_target_artifact_uris,
        }
        if not decision_endpoints.issubset(request_uris):
            raise ValueError(
                "The checker certified a relationship endpoint outside its "
                "verification request."
            )
        if (
            decision.obligation_uri is not None
            and decision.obligation_uri not in request_uris
        ):
            raise ValueError(
                "The checker certified an obligation outside its verification request."
            )
