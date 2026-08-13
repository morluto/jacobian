"""Typed formal modular-polynomial identity operation."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.domains._examples import example
from jacobian.domains.number_theory._support import number_theory_operation
from jacobian.math.modular_polynomials import (
    ModularPolynomialIdentityRequest,
    ModularPolynomialIdentityValue,
    modular_polynomial_identity,
)
from jacobian.provider_runtime import source_provider_runtime


def _modular_identity_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.modular-polynomial-identity-checker",
        version="1",
        entrypoint="jacobian_checkers.modular_polynomial_identity:check_modular_polynomial_identity",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-integer-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


MODULAR_IDENTITY_CAPABILITIES = (
    number_theory_operation(
        "modular.polynomial_identity.compute",
        "Compare modular polynomial coefficients",
        (
            "Canonicalize two sparse integer polynomials and compare their formal "
            "coefficients modulo m. This is polynomial-ring identity, not equality "
            "of induced functions on residue assignments."
        ),
        ModularPolynomialIdentityRequest,
        ModularPolynomialIdentityValue,
        modular_polynomial_identity,
        "number-theory",
        "modular",
        "polynomial",
        "identity",
        "coefficientwise",
        invocation_examples=(
            example(
                "coefficientwise_identity_mod_4",
                "Compare two formal polynomial coefficients modulo 4.",
                {
                    "modulus": 4,
                    "variables": ["z"],
                    "left": [{"coefficient": "9", "exponents": [6]}],
                    "right": [{"coefficient": "-7", "exponents": [6]}],
                },
            ),
        ),
    ),
)

MODULAR_IDENTITY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "modular.polynomial_identity.compute",
        ModularPolynomialIdentityRequest,
        "check_modular_polynomial_identity",
        "modular.polynomial-identity.stdlib-replay",
        entrypoint_module="jacobian_checkers.modular_polynomial_identity",
        provider_runtime_factory=_modular_identity_runtime,
        replay_method="standard-library coefficientwise modular replay",
        reason=(
            "operator-authorized standard-library checker independently "
            "canonicalizes every formal coefficient modulo m"
        ),
        verification_capability_id="modular.polynomial_identity.verify",
        verification_title="Verify a modular polynomial identity",
        verification_description=(
            "Independently verify formal coefficientwise identity over Z/mZ; "
            "this does not compare induced polynomial functions."
        ),
        verification_tags=(
            "verification",
            "exact",
            "number-theory",
            "modular",
            "polynomial",
            "identity",
            "coefficientwise",
        ),
    ),
)

__all__ = ["MODULAR_IDENTITY_CAPABILITIES", "MODULAR_IDENTITY_CHECKERS"]
