"""Independent checker declarations owned by the number-theory domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderRuntime,
)
from jacobian.contracts.number_theory import (
    FactorizationRequest,
    IntegerPairRequest,
    ModularPolynomialResidueImageRequest,
    PowerfulNumberRequest,
)
from jacobian.provider_runtime import source_provider_runtime
from jacobian.providers import flint_runtime

_EXACT_DOMAIN_ENTRYPOINT = "jacobian_checkers.exact_domain_operations"


def _flint_exact_replay_runtime(
    *, checker_ids: tuple[str, ...] = (), refresh: bool = False
) -> CapabilityProviderRuntime:
    return flint_runtime.exact_domain_checker_provider_runtime(
        checker_ids=checker_ids,
        refresh=refresh,
    )


def _integer_lcm_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return source_provider_runtime(
        "jacobian.integer-lcm-checker",
        version="1",
        entrypoint="jacobian_checkers.integer_lcm:check_integer_lcm",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("standard-library-integer-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


NUMBER_THEORY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "integer.compute.lcm",
        IntegerPairRequest,
        "check_integer_lcm",
        "integer.lcm.euclidean-replay",
        entrypoint_module="jacobian_checkers.integer_lcm",
        provider_runtime_factory=_integer_lcm_runtime,
        replay_method="standard-library Euclidean recurrence replay",
        reason=(
            "operator-authorized standard-library checker independently evaluates "
            "the bounded least common multiple by a Euclidean recurrence without "
            "calling math.lcm or importing producer code"
        ),
    ),
    ExactReplayCheckerDeclaration(
        "integer.compute.prime_factorization",
        FactorizationRequest,
        "check_integer_prime_factorization",
        "integer.prime-factorization.flint-replay",
        entrypoint_module=_EXACT_DOMAIN_ENTRYPOINT,
        provider_runtime_factory=_flint_exact_replay_runtime,
        replay_method="Python-FLINT prime-factorization replay",
        reason=(
            "operator-authorized Python-FLINT checker independent of the "
            "isolated SymPy factorization producer"
        ),
        verification_capability_id="integer.prime_factorization.verify",
        verification_title="Verify an integer prime factorization",
        verification_description=(
            "Independently verify the complete canonical prime-power "
            "factorization submitted with its exact nonzero integer input."
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
        provider_runtime_factory=_flint_exact_replay_runtime,
        replay_method="Python-FLINT powerful-number replay",
        reason=(
            "operator-authorized Python-FLINT checker independent of the "
            "isolated SymPy powerful-number producer"
        ),
        verification_capability_id="integer.powerful.verify",
        verification_title="Verify a powerful-number decision",
        verification_description=(
            "Independently verify one submitted powerful-number decision against "
            "its exact integer input, complete canonical factor witness, and every "
            "violating prime."
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
        provider_runtime_factory=_flint_exact_replay_runtime,
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
