"""Independent checker declarations owned by the number-theory domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.number_theory import (
    FactorizationRequest,
    FiniteAbelianGroupFactorizationRequest,
    ModularPolynomialResidueImageRequest,
    PowerfulNumberRequest,
)

_EXACT_DOMAIN_ENTRYPOINT = "jacobian_checkers.exact_domain_operations"
_FINITE_GROUP_ENTRYPOINT = "jacobian_checkers.finite_abelian_groups"

NUMBER_THEORY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "finite_abelian_group.exact_factorization.compute",
        FiniteAbelianGroupFactorizationRequest,
        "check_finite_abelian_group_exact_factorization",
        "finite-abelian-group.exact-factorization.stdlib-replay",
        entrypoint_module=_FINITE_GROUP_ENTRYPOINT,
        replay_method="Python standard-library exhaustive group replay",
        reason=(
            "operator-authorized checker independently normalizes both factors "
            "and replays every sum in the complete finite group"
        ),
        verification_capability_id="finite_abelian_group.exact_factorization.verify",
        verification_title="Verify a finite abelian group exact factorization",
        verification_description=(
            "Independently verify the complete representation histogram, exact "
            "factorization decision, and first failure witnesses."
        ),
        verification_tags=(
            "verification",
            "exact",
            "number-theory",
            "finite-abelian-group",
            "factorization",
            "coset-transversal",
        ),
    ),
    ExactReplayCheckerDeclaration(
        "integer.compute.prime_factorization",
        FactorizationRequest,
        "check_integer_prime_factorization",
        "integer.prime-factorization.flint-replay",
        entrypoint_module=_EXACT_DOMAIN_ENTRYPOINT,
        replay_method="Python-FLINT prime-factorization replay",
        reason=(
            "operator-authorized Python-FLINT checker independent of the "
            "isolated SymPy factorization producer"
        ),
        verification_capability_id="integer.prime_factorization.verify",
        verification_title="Verify an integer prime factorization",
        verification_description=(
            "Independently verify the complete canonical prime-power "
            "factorization of one stored nonzero integer result."
        ),
        verification_tags=(
            "verification",
            "exact",
            "integer",
            "number-theory",
            "prime-factorization",
        ),
    ),
    ExactReplayCheckerDeclaration(
        "integer.decide.powerful",
        PowerfulNumberRequest,
        "check_integer_powerful_number",
        "integer.powerful.flint-replay",
        entrypoint_module=_EXACT_DOMAIN_ENTRYPOINT,
        replay_method="Python-FLINT powerful-number replay",
        reason=(
            "operator-authorized Python-FLINT checker independent of the "
            "isolated SymPy powerful-number producer"
        ),
        verification_capability_id="integer.powerful.verify",
        verification_title="Verify a powerful-number decision",
        verification_description=(
            "Independently verify one stored powerful-number decision, its "
            "complete canonical factor witness, and every violating prime."
        ),
        verification_tags=(
            "verification",
            "exact",
            "integer",
            "number-theory",
            "powerful-number",
        ),
    ),
    ExactReplayCheckerDeclaration(
        "modular.polynomial_residue_image.compute",
        ModularPolynomialResidueImageRequest,
        "check_modular_polynomial_residue_image",
        "modular.polynomial-residue-image.flint-replay",
        entrypoint_module=_EXACT_DOMAIN_ENTRYPOINT,
        replay_method="Python-FLINT exhaustive modular-polynomial replay",
        reason=(
            "operator-authorized Python-FLINT checker independently reconstructs "
            "the declared Cartesian product and replays every modular-polynomial "
            "evaluation without importing the stdlib producer"
        ),
        verification_capability_id="modular.polynomial_residue_image.verify",
        verification_title="Verify a modular polynomial residue image",
        verification_description=(
            "Independently verify one complete bounded modular-polynomial residue "
            "image, including every assignment, multiplicity, and first witness."
        ),
        verification_tags=(
            "verification",
            "exact",
            "number-theory",
            "modular",
            "polynomial",
            "residue",
            "enumeration",
        ),
    ),
)


__all__ = ["NUMBER_THEORY_EXACT_REPLAY_CHECKERS"]
