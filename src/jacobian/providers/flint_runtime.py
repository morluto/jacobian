"""Provider-owned runtime declarations for FLINT-backed mathematics."""

import importlib

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.provider_runtime import (
    PYTHON_FLINT_HNF_FLINT_VERSION,
    PYTHON_FLINT_LLL_CONFIGURATION,
    PYTHON_FLINT_VERSION,
    ProviderRuntimeError,
    _jacobian_identity,
    _unavailable_runtime,
    composite_provider_runtime,
    python_distribution_provider_runtime,
    source_provider_runtime,
)


def python_flint_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the exact optional Python-FLINT compatibility profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpq", "fmpq_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-rational",
            "dense-matrix",
            "reduced-row-echelon-form",
        ),
        configuration={
            "domain": "QQ",
            "operation": "fmpq_mat.rref",
            "maximum_rows": 32,
            "maximum_columns": 32,
            "free_variable_policy": "ZERO",
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} compatibility profile."
            ),
        )
    return runtime


def python_flint_exact_checker_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT API used by exact-domain replay."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=(
            "fmpq",
            "fmpq_mat",
            "fmpq_poly",
            "fmpz",
            "fmpz_mat",
            "fmpz_poly",
        ),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=("exact-domain-independent-replay",),
        configuration={
            "import_name": "flint",
            "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} exact-checker profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "The pinned Python-FLINT exact-checker runtime cannot be imported."
                ),
            )
        if getattr(flint, "__FLINT_VERSION__", None) != (
            PYTHON_FLINT_HNF_FLINT_VERSION
        ):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned exact-checker profile."
                ),
            )
    return runtime


def python_flint_analysis_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Arb API used by validated real analysis."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("arb", "ctx", "fmpq"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=("arb-ball-arithmetic",),
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} real-analysis profile."
            ),
        )
    return runtime


def python_flint_probability_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned exact-rational API used by probability producers."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpq",),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-rational-moments",
            "finite-event-probability",
            "finite-conditioning",
            "finite-pushforward",
            "finite-convolution",
            "gaussian-polynomial-complex-rational-moments",
            "small-graph-exact-reliability",
        ),
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} exact-probability profile."
            ),
        )
    return runtime


def exact_domain_checker_source_provider_runtime() -> CapabilityProviderRuntime:
    """Identify the bundled source used by exact-domain replay checkers."""

    try:
        version, _, _ = _jacobian_identity()
    except ProviderRuntimeError:
        return _unavailable_runtime(
            provider="jacobian.exact-domain-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The exact-domain checker source could not be identified.",
        )
    return source_provider_runtime(
        "jacobian.exact-domain-checker-source",
        version=version,
        entrypoint=("jacobian_checkers.exact_domain_operations:check_polynomial_gcd"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("clean-process-replay",),
    )


def exact_domain_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Bind independent checker source and its pinned FLINT replay backend."""

    return composite_provider_runtime(
        "jacobian.exact-domain-checkers",
        components=(
            exact_domain_checker_source_provider_runtime(),
            python_flint_exact_checker_provider_runtime(refresh=refresh),
        ),
        features=("clean-process-replay", "python-flint"),
        checker_ids=checker_ids,
    )


def graph_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-graph checker source without FLINT."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.graph-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The finite-graph checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.graph-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.graph_exact_operations:"
                "check_graph_induced_tree_maximum"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.graph-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-subset-exhaustive-replay",
            "hamiltonian-path-exhaustive-replay",
            "tutte-berge-barrier-replay",
            "declared-graph-symmetry-orbit-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def probability_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-probability checker source without FLINT."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.probability-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic=(
                "The finite-probability checker source could not be identified."
            ),
        )
    else:
        source = source_provider_runtime(
            "jacobian.probability-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.exact_probability_operations:check_finite_raw_moment"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.probability-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-rational-probability-replay",
            "gaussian-polynomial-coefficient-contraction",
            "small-graph-reliability-exhaustive-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def combinatorics_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind exact combinatorics checker source without producer dependencies."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        recurrence_source = _unavailable_runtime(
            provider="jacobian.combinatorics-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The recurrence checker source could not be identified.",
        )
        additive_source = _unavailable_runtime(
            provider="jacobian.additive-combinatorics-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The additive-combinatorics checker source could not be identified.",
        )
    else:
        recurrence_source = source_provider_runtime(
            "jacobian.combinatorics-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.recurrence_series:check_linear_recurrence_evaluation"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
        additive_source = source_provider_runtime(
            "jacobian.additive-combinatorics-checker-source",
            version=version,
            entrypoint=("jacobian_checkers.additive_combinatorics:check_integer_sidon"),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=(
                "clean-process-replay",
                "fixed-order-extension-exhaustion",
                "standard-library-only",
            ),
        )
    return composite_provider_runtime(
        "jacobian.combinatorics-exact-checkers",
        components=(recurrence_source, additive_source),
        features=(
            "clean-process-replay",
            "additive-difference-set-replay",
            "fixed-order-extension-exhaustion",
            "linear-recurrence-replay",
            "rational-series-residual-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def topology_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-simplicial-topology checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.topology-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The simplicial-topology checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.topology-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.simplicial_topology:"
                "check_simplicial_complex_materialization"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.topology-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-face-closure-replay",
            "oriented-boundary-replay",
            "prime-field-quotient-replay",
            "integral-smith-certificate-replay",
            "free-and-torsion-generator-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def certified_snf_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent transformation-certified Smith checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.certified-snf-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The certified-Smith checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.certified-snf-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.certified_snf:check_certified_smith_normal_form"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.certified-snf-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "full-transformation-relation-replay",
            "bareiss-unimodularity-replay",
            "smith-divisibility-chain-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def poset_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-poset checker source without NetworkX."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.poset-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The finite-poset checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.poset-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.finite_posets:check_finite_poset_materialization"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.poset-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-poset-closure-replay",
            "dilworth-dual-certificate-replay",
            "complete-ideal-dp-replay",
            "mobius-convolution-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def graded_syzygy_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the standard-library graded-syzygy checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.graded-syzygy-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The graded-syzygy checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.graded-syzygy-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.jacobian_syzygy:check_graded_jacobian_syzygy"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=(
                "clean-process-replay",
                "exact-rational",
                "standard-library-only",
            ),
        )
    return composite_provider_runtime(
        "jacobian.graded-syzygy-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "graded-coefficient-map-reconstruction",
            "exact-rational-rank-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def projective_arrangement_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the standard-library projective-arrangement checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.projective-arrangement-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic=(
                "The projective-arrangement checker source could not be identified."
            ),
        )
    else:
        source = source_provider_runtime(
            "jacobian.projective-arrangement-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.projective_arrangements:"
                "check_projective_line_arrangement_flats"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=(
                "clean-process-replay",
                "exact-rational",
                "standard-library-only",
            ),
        )
    return composite_provider_runtime(
        "jacobian.projective-arrangement-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "projective-pair-incidence-exhaustive-replay",
            "exact-rational",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def python_flint_hnf_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT integer row-HNF profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpz", "fmpz_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-integer",
            "dense-matrix",
            "row-hermite-normal-form",
            "left-transformation",
        ),
        configuration={
            "domain": "ZZ",
            "operation": "fmpz_mat.hnf(transform=True)",
            "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
            "maximum_rows": 32,
            "maximum_columns": 32,
            "normal_form_convention": "FLINT_ROW_HNF",
            "relation": "H=U*A",
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} HNF compatibility profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic="The pinned Python-FLINT HNF runtime cannot be imported.",
            )
        if getattr(flint, "__FLINT_VERSION__", None) != PYTHON_FLINT_HNF_FLINT_VERSION:
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned "
                    f"{PYTHON_FLINT_HNF_FLINT_VERSION} HNF profile."
                ),
            )
    return runtime


def python_flint_lll_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT exact-gram LLL profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpz", "fmpz_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-integer",
            "dense-matrix",
            "lll-reduction",
            "left-transformation",
        ),
        configuration=PYTHON_FLINT_LLL_CONFIGURATION,
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} LLL profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic="The pinned Python-FLINT LLL runtime cannot be imported.",
            )
        if getattr(flint, "__FLINT_VERSION__", None) != (
            PYTHON_FLINT_HNF_FLINT_VERSION
        ):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned LLL profile."
                ),
            )
    return runtime
