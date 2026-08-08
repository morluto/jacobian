"""Independent checker declarations owned by the matrix domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.matrix_lattice import HermiteNormalFormRequest
from jacobian.contracts.matrix_operations import (
    IntegerMatrixRequest,
    MatrixDeterminantRequest,
    MatrixRankRequest,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    SquareRationalMatrixRequest,
)

MATRIX_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
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
        verification_capability_id="matrix.normal_form.hermite.verify",
        verification_title="Verify a transformation-certified row Hermite normal form",
        verification_description=(
            "Independently verify the retained H and U certificate against its "
            "stored integer matrix input."
        ),
        verification_tags=("verification", "exact", "matrix", "hermite-normal-form"),
    ),
    ExactReplayCheckerDeclaration(
        "matrix.determinant.compute",
        MatrixDeterminantRequest,
        "check_matrix_determinant",
        "matrix.determinant.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.rank.compute",
        MatrixRankRequest,
        "check_matrix_rank",
        "matrix.rank.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.multiply.compute",
        RationalMatrixProductRequest,
        "check_matrix_product",
        "matrix.product.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.normal_form.rref.compute",
        RationalMatrixRequest,
        "check_matrix_rref",
        "matrix.rref.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.nullspace.compute",
        RationalMatrixRequest,
        "check_matrix_nullspace",
        "matrix.nullspace.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.characteristic_polynomial.compute",
        SquareRationalMatrixRequest,
        "check_matrix_characteristic_polynomial",
        "matrix.characteristic-polynomial.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "matrix.normal_form.smith.compute",
        IntegerMatrixRequest,
        "check_matrix_smith_normal_form",
        "matrix.smith-normal-form.flint-replay",
    ),
)

__all__ = ["MATRIX_EXACT_REPLAY_CHECKERS"]
