"""Independent checker declaration for certified Smith normal forms."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.certified_snf import CertifiedSmithNormalFormRequest
from jacobian.contracts.operations import ProviderObservation
from jacobian.providers import flint_runtime


def _certified_snf_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return flint_runtime.certified_snf_checker_provider_runtime(checker_ids=checker_ids)


CERTIFIED_SNF_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "matrix.normal_form.smith.certified.compute",
        CertifiedSmithNormalFormRequest,
        "check_certified_smith_normal_form",
        "matrix.smith-normal-form.transformation-certificate-v1",
        entrypoint_module="jacobian_checkers.certified_snf",
        observation_loader=_certified_snf_runtime,
        replay_method="independent Smith transformation-certificate replay",
        reason=(
            "operator-authorized independent checker validates D=UAV, both "
            "unimodular determinants, and the complete canonical divisibility chain"
        ),
        verification_operation_id="matrix.normal_form.smith.certified.verify",
        verification_title="Verify a transformation-certified Smith normal form",
        verification_description=(
            "Independently verify the full Smith diagonal and both unimodular "
            "basis transformations against the exact submitted integer matrix input."
        ),
        verification_tags=(
            "verification",
            "exact",
            "matrix",
            "integer",
            "smith-normal-form",
            "certificate",
        ),
    ),
)

__all__ = ["CERTIFIED_SNF_AUTHORIZED_CHECKERS"]
