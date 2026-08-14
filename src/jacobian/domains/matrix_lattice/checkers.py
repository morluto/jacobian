"""Independent checker declarations owned by the matrix domain."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.matrix_lattice import HermiteNormalFormRequest
from jacobian.contracts.matrix_operations import (
    IntegerMatrixRequest,
    MatrixDeterminantRequest,
    MatrixRankRequest,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.contracts.operations import (
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import source_provider_runtime
from jacobian.providers import flint_runtime


def _flint_exact_replay_runtime(
    *, checker_ids: tuple[str, ...] = (), refresh: bool = False
) -> ProviderObservation:
    return flint_runtime.exact_domain_checker_provider_runtime(
        checker_ids=checker_ids,
        refresh=refresh,
    )


def _hnf_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return source_provider_runtime(
        "jacobian.matrix-hnf-checker",
        version="1",
        entrypoint="jacobian_checkers.matrix_normal_forms:check_hermite_normal_form",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("standard-library-integer-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


MATRIX_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "matrix.normal_form.hermite.materialize",
        HermiteNormalFormRequest,
        "check_hermite_normal_form",
        "matrix.normal_form.hermite",
        entrypoint_module="jacobian_checkers.matrix_normal_forms",
        replay_method="independent row-HNF and unimodular-transform replay",
        reason=(
            "standard-library checker independently validates H=UA, unimodularity, "
            "and the full retained row-HNF certificate"
        ),
        verification_operation_id="matrix.normal_form.hermite.verify",
        verification_title="Verify a transformation-certified row Hermite normal form",
        verification_description=(
            "Independently verify the retained H and U certificate against its "
            "stored integer matrix input."
        ),
        verification_tags=("verification", "exact", "matrix", "hermite-normal-form"),
        observation_loader=_hnf_runtime,
    ),
    AuthorizedChecker(
        "matrix.determinant.compute",
        MatrixDeterminantRequest,
        "check_matrix_determinant",
        "matrix.determinant.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.rank.compute",
        MatrixRankRequest,
        "check_matrix_rank",
        "matrix.rank.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.multiply.compute",
        RationalMatrixProductRequest,
        "check_matrix_product",
        "matrix.product.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.normal_form.rref.compute",
        RationalMatrixRequest,
        "check_matrix_rref",
        "matrix.rref.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.nullspace.compute",
        RationalMatrixRequest,
        "check_matrix_nullspace",
        "matrix.nullspace.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.characteristic_polynomial.compute",
        SquareRationalMatrixRequest,
        "check_matrix_characteristic_polynomial",
        "matrix.characteristic-polynomial.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
    AuthorizedChecker(
        "matrix.normal_form.smith.compute",
        IntegerMatrixRequest,
        "check_matrix_smith_normal_form",
        "matrix.smith-normal-form.flint-replay",
        observation_loader=_flint_exact_replay_runtime,
        optional=True,
    ),
)

__all__ = ["MATRIX_AUTHORIZED_CHECKERS"]
