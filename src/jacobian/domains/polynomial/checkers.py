"""Independent checker declarations owned by the polynomial domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.jacobian_syzygy import GradedJacobianSyzygyRequest
from jacobian.contracts.polynomial_operations import (
    PolynomialDiscriminantRequest,
    PolynomialFactorRequest,
    PolynomialGcdRequest,
    PolynomialResultantRequest,
    PolynomialSquareFreeRequest,
)

POLYNOMIAL_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "polynomial.jacobian_syzygy.minimum_degree.compute",
        GradedJacobianSyzygyRequest,
        "check_graded_jacobian_syzygy",
        "polynomial.jacobian-syzygy.graded-fraction-replay",
        entrypoint_module="jacobian_checkers.jacobian_syzygy",
        replay_method="standard-library exact rational graded-map replay",
        reason=(
            "operator-authorized exact rational checker independently reconstructs "
            "the homogeneous coefficient maps without importing the SymPy producer"
        ),
        verification_capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
        verification_title="Verify a first graded Jacobian syzygy degree",
        verification_description=(
            "Independently reconstruct every bounded homogeneous coefficient map, "
            "rank ledger, nonzero minor, and first kernel from the exact producer "
            "input and the complete, unmodified producer output.result object."
        ),
        verification_tags=(
            "verification",
            "exact",
            "polynomial",
            "jacobian",
            "syzygy",
        ),
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.gcd",
        PolynomialGcdRequest,
        "check_polynomial_gcd",
        "polynomial.gcd.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.resultant",
        PolynomialResultantRequest,
        "check_polynomial_resultant",
        "polynomial.resultant.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.discriminant",
        PolynomialDiscriminantRequest,
        "check_polynomial_discriminant",
        "polynomial.discriminant.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.square_free_decomposition",
        PolynomialSquareFreeRequest,
        "check_polynomial_square_free",
        "polynomial.square-free.flint-replay",
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.factor.compute",
        PolynomialFactorRequest,
        "check_polynomial_factorization",
        "polynomial.factorization.flint-replay",
    ),
)

__all__ = ["POLYNOMIAL_EXACT_REPLAY_CHECKERS"]
