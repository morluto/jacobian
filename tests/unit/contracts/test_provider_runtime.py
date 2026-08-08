from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import jacobian.provider_runtime as provider_runtime
import jacobian.providers.flint_runtime as flint_runtime
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.provider_measurements import (
    ProviderMeasurementSample,
    ProviderMeasurementStatus,
)
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    ProviderRuntimeErrorCode,
    composite_provider_runtime,
    python_distribution_provider_runtime,
    require_provider_runtime_ready,
    require_provider_runtime_unchanged,
)
from jacobian.providers.flint_runtime import (
    exact_domain_checker_provider_runtime,
    python_flint_exact_checker_provider_runtime,
)
from jacobian.providers.lean_runtime import (
    LeanRuntimeIdentityError,
    lean_frontend_provider_runtime,
    lean_provider_runtime,
    require_lean_semantic_runtime_identity,
)


def _runtime(**updates: object) -> CapabilityProviderRuntime:
    values: dict[str, object] = {
        "provider": "tests.fixture",
        "availability": CapabilityProviderAvailability.AVAILABLE,
        "version": "1.2.3",
        "digest": "sha256:" + "a" * 64,
        "digest_kind": CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        "platform": "linux-x86_64",
        "install_tier": CapabilityInstallTier.T1,
        "license_id": "MIT",
        "license_files": ("fixture.dist-info/licenses/LICENSE",),
        "features": ("exact-arithmetic",),
        "checker_ids": ("checker://sha256/" + "b" * 64,),
    }
    values.update(updates)
    return CapabilityProviderRuntime(**values)


def _lean_runtime_layout(
    tmp_path: Path,
    *,
    with_mathlib_project: bool,
) -> tuple[Path, Path | None]:
    toolchain = tmp_path / "toolchain"
    executable = toolchain / "bin" / "lean"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"pinned-lean")
    (toolchain / "bin" / "lake").write_bytes(b"pinned-lake")
    init_module = toolchain / "lib" / "lean" / "Init.olean"
    init_module.parent.mkdir(parents=True)
    init_module.write_bytes(b"pinned-init")
    if not with_mathlib_project:
        return executable, None
    project = tmp_path / "project"
    for path, content in {
        "lake-manifest.json": b'{"packages":[]}',
        "lakefile.toml": b'name = "jacobianLeanRuntime"',
        "lean-toolchain": b"leanprover/lean4:v4.31.0",
        "JacobianLeanRuntime.lean": b"import Mathlib",
        "JacobianLeanProofState.lean": b"import Mathlib",
        ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean": b"mathlib",
        ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Algebra.olean": b"algebra",
        ".lake/build/lib/lean/JacobianLeanRuntime.olean": b"runtime-module",
        ".lake/build/lib/lean/JacobianLeanProofState.olean": b"state-module",
        ".lake/build/bin/jacobian_lean_proof_state": b"proof-state-helper",
    }.items():
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return executable, project


def test_available_provider_requires_exact_version_and_digest() -> None:
    with pytest.raises(
        ValidationError,
        match="available provider runtime requires version, digest, and digest kind",
    ):
        _runtime(version=None, digest=None, digest_kind=None)


def test_provider_metadata_rejects_duplicate_features_and_checker_ids() -> None:
    with pytest.raises(ValidationError, match="provider features must be unique"):
        _runtime(features=("exact-arithmetic", "exact-arithmetic"))
    checker_id = "checker://sha256/" + "b" * 64
    with pytest.raises(ValidationError, match="provider checker IDs must be unique"):
        _runtime(checker_ids=(checker_id, checker_id))


def test_unavailable_provider_requires_a_public_diagnostic() -> None:
    with pytest.raises(
        ValidationError,
        match="unavailable provider runtime requires a diagnostic",
    ):
        _runtime(
            availability=CapabilityProviderAvailability.UNAVAILABLE,
            version=None,
            digest=None,
            digest_kind=None,
        )


def test_descriptor_provider_must_match_runtime_identity() -> None:
    with pytest.raises(ValidationError, match="descriptor provider must match"):
        CapabilityDescriptor(
            capability_id="fixture.increment",
            version="1",
            title="Increment",
            description="Increment one integer.",
            provider="tests.other",
            provider_runtime=_runtime(),
            modes=(CapabilityMode.EXPLORE,),
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )


def test_composite_provider_binds_all_component_identities() -> None:
    first = _runtime(provider="tests.first", version="1")
    second = _runtime(
        provider="tests.second",
        version="2",
        digest="sha256:" + "c" * 64,
    )

    runtime = composite_provider_runtime(
        "tests.composite",
        components=(first, second),
        features=("two-backends",),
    )

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.digest_kind is CapabilityProviderDigestKind.COMPOSITE
    assert runtime.digest is not None
    assert tuple(
        component["provider"] for component in runtime.configuration["components"]
    ) == ("tests.first", "tests.second")
    changed = composite_provider_runtime(
        "tests.composite",
        components=(first, second.model_copy(update={"version": "3"})),
        features=("two-backends",),
    )
    assert changed.digest != runtime.digest


def test_composite_provider_fails_closed_when_one_component_is_unavailable() -> None:
    unavailable = _runtime(
        provider="tests.missing",
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        version=None,
        digest=None,
        digest_kind=None,
        diagnostic="Missing fixture runtime.",
    )

    runtime = composite_provider_runtime(
        "tests.composite",
        components=(_runtime(provider="tests.present"), unavailable),
    )

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.digest is None
    assert runtime.diagnostic is not None
    assert "tests.missing" in runtime.diagnostic


def test_exact_checker_composite_runtime_is_remeasured_recursively() -> None:
    runtime = exact_domain_checker_provider_runtime(refresh=True)
    require_provider_runtime_unchanged(runtime)

    components = list(runtime.configuration["components"])
    flint = dict(components[1])
    flint["digest"] = "sha256:" + "0" * 64
    components[1] = flint
    changed = runtime.model_copy(
        update={
            "configuration": {
                **runtime.configuration,
                "components": components,
            }
        }
    )

    with pytest.raises(ProviderRuntimeError, match="identity changed"):
        require_provider_runtime_unchanged(changed)


def test_exact_checker_runtime_defers_rational_polynomial_api_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_flint = SimpleNamespace(
        __FLINT_VERSION__=flint_runtime.PYTHON_FLINT_HNF_FLINT_VERSION,
        fmpq=object(),
        fmpq_mat=object(),
        fmpz=object(),
        fmpz_mat=object(),
        fmpz_poly=object(),
    )
    monkeypatch.setattr(
        flint_runtime.importlib,
        "import_module",
        lambda _name: incomplete_flint,
    )

    runtime = python_flint_exact_checker_provider_runtime(refresh=True)

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.distribution_required_attributes == (
        "fmpq",
        "fmpq_mat",
        "fmpq_poly",
        "fmpz",
        "fmpz_mat",
        "fmpz_poly",
    )


def test_exact_checker_runtime_rejects_different_linked_flint_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = python_flint_exact_checker_provider_runtime()
    assert available.availability is CapabilityProviderAvailability.AVAILABLE
    monkeypatch.setattr(
        flint_runtime,
        "python_distribution_provider_runtime",
        lambda *_args, **_kwargs: available,
    )
    monkeypatch.setattr(
        flint_runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__FLINT_VERSION__="3.5.0"),
    )

    runtime = python_flint_exact_checker_provider_runtime(refresh=True)

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.digest is None
    assert runtime.diagnostic is not None
    assert "linked FLINT library" in runtime.diagnostic


def test_measurement_status_cannot_hide_missing_elapsed_time() -> None:
    with pytest.raises(
        ValidationError,
        match="completed provider measurement requires elapsed seconds",
    ):
        ProviderMeasurementSample(status=ProviderMeasurementStatus.COMPLETED)

    with pytest.raises(
        ValidationError,
        match="incomplete provider measurement requires a detail",
    ):
        ProviderMeasurementSample(status=ProviderMeasurementStatus.SKIPPED)


def test_python_distribution_unchanged_check_does_not_import_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = python_distribution_provider_runtime(
        "pydantic",
        distribution_name="pydantic",
        import_name="pydantic",
        required_attributes=("BaseModel",),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        refresh=True,
    )
    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.distribution_import_name == "pydantic"
    assert runtime.distribution_required_attributes == ("BaseModel",)

    def fail_import(_name: str) -> object:
        pytest.fail("distribution identity replay must not import the provider")

    monkeypatch.setattr(provider_runtime.importlib, "import_module", fail_import)
    require_provider_runtime_unchanged(runtime)


def test_python_provider_readiness_checks_required_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = python_distribution_provider_runtime(
        "pydantic",
        distribution_name="pydantic",
        import_name="pydantic",
        required_attributes=("missing_required_attribute",),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        refresh=True,
    )
    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    monkeypatch.setattr(
        provider_runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    with pytest.raises(ProviderRuntimeError) as raised:
        require_provider_runtime_ready(runtime)

    assert raised.value.code is ProviderRuntimeErrorCode.READINESS_FAILED


def test_z3_version_mismatch_reports_smt_solver_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = _runtime(provider="jacobian.z3", version="0.0.0")
    monkeypatch.setattr(
        provider_runtime,
        "python_distribution_provider_runtime",
        lambda *_args, **_kwargs: mismatched,
    )

    runtime = provider_runtime.known_provider_runtime("jacobian.z3")

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.diagnostic == (
        "Z3 is installed but does not match the pinned "
        f"{provider_runtime.Z3_SOLVER_VERSION} SMT solver profile."
    )


def test_disappeared_executable_is_unavailable(tmp_path: Path) -> None:
    runtime = _runtime(
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        configuration={"executable": str(tmp_path / "gone")},
    )

    with pytest.raises(ProviderRuntimeError) as raised:
        require_provider_runtime_unchanged(runtime)

    assert raised.value.code is ProviderRuntimeErrorCode.UNAVAILABLE


def test_lean_frontend_runtime_binds_the_pinned_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, _project = _lean_runtime_layout(tmp_path, with_mathlib_project=False)
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            executable,
            None
            if not require_mathlib
            else pytest.fail("CORE must not require Mathlib"),
        ),
    )

    runtime = lean_frontend_provider_runtime()

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE
    assert runtime.features == ("CORE", "elaboration", "lean-statement")
    assert runtime.configuration["executable"] == str(executable)
    assert runtime.configuration["profiles"]["CORE"]["import_name"] == "Init.Prelude"


def test_lean_frontend_runtime_preserves_actionable_probe_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    def fail(*, require_mathlib: bool) -> tuple[Path, Path | None]:
        assert require_mathlib is False
        raise RuntimeError(
            "TOOLCHAIN_RESOLUTION: the pinned Lean executable is unavailable"
        )

    monkeypatch.setattr(lean4, "inspect_runtime", fail)

    runtime = lean_frontend_provider_runtime()

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.diagnostic is not None
    assert "TOOLCHAIN_RESOLUTION" in runtime.diagnostic
    assert "executable is unavailable" in runtime.diagnostic


def test_lean_frontend_runtime_bounds_probe_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (_ for _ in ()).throw(OSError("x" * 2_000)),
    )

    runtime = lean_frontend_provider_runtime()

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert runtime.diagnostic is not None
    assert len(runtime.diagnostic) == 512


def test_lean_mathlib_runtime_binds_the_lake_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, project = _lean_runtime_layout(tmp_path, with_mathlib_project=True)
    assert project is not None
    lake = executable.with_name("lake")
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, project) if require_mathlib else (executable, None)
        ),
    )

    runtime = lean_provider_runtime(
        profiles={
            "mathlib": {"mathlib_commit": "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"}
        },
        checker_ids=(),
    )

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE
    assert runtime.configuration["executable"] == str(executable)
    assert runtime.configuration["lake_executable"] == str(lake)
    assert runtime.configuration["lake_digest"] == provider_runtime._sha256_file(lake)


@pytest.mark.parametrize(
    "relative_path",
    (
        "lakefile.toml",
        "JacobianLeanProofState.lean",
        ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean",
        ".lake/build/bin/jacobian_lean_proof_state",
    ),
)
def test_lean_semantic_runtime_identity_rejects_project_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    from jacobian_checkers import lean4

    executable, project = _lean_runtime_layout(tmp_path, with_mathlib_project=True)
    assert project is not None
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, project) if require_mathlib else (executable, None)
        ),
    )
    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )

    (project / relative_path).write_bytes(b"replacement")

    with pytest.raises(LeanRuntimeIdentityError, match="identity changed"):
        require_lean_semantic_runtime_identity(runtime)


def test_lean_semantic_runtime_identity_rejects_lake_launcher_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, project = _lean_runtime_layout(tmp_path, with_mathlib_project=True)
    assert project is not None
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, project) if require_mathlib else (executable, None)
        ),
    )
    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )

    executable.with_name("lake").write_bytes(b"replacement")

    with pytest.raises(LeanRuntimeIdentityError, match="identity changed"):
        require_lean_semantic_runtime_identity(runtime)


def test_lean_semantic_runtime_identity_rejects_imported_mathlib_module_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, project = _lean_runtime_layout(tmp_path, with_mathlib_project=True)
    assert project is not None
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, project) if require_mathlib else (executable, None)
        ),
    )
    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}}, checker_ids=()
    )

    (
        project / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Algebra.olean"
    ).write_bytes(b"replacement")

    with pytest.raises(LeanRuntimeIdentityError, match="identity changed"):
        require_lean_semantic_runtime_identity(runtime)


def test_lean_semantic_runtime_identity_requires_the_resolved_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, project = _lean_runtime_layout(tmp_path, with_mathlib_project=True)
    assert project is not None
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, project) if require_mathlib else (executable, None)
        ),
    )
    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )

    (project / ".lake/build/bin/jacobian_lean_proof_state").unlink()

    with pytest.raises(LeanRuntimeIdentityError, match=r"component .* is unavailable"):
        require_lean_semantic_runtime_identity(runtime)


def test_lean_mathlib_runtime_is_unavailable_without_a_lake_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, _project = _lean_runtime_layout(tmp_path, with_mathlib_project=False)
    executable.with_name("lake").unlink()
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            (executable, tmp_path) if require_mathlib else (executable, None)
        ),
    )

    runtime = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE


def test_lean_core_runtime_does_not_bind_a_lake_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable, _project = _lean_runtime_layout(tmp_path, with_mathlib_project=False)
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (
            pytest.fail("CORE must not require Mathlib")
            if require_mathlib
            else (executable, None)
        ),
    )

    runtime = lean_provider_runtime(
        profiles={"core": {"mathlib_commit": None}},
        checker_ids=(),
    )

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert "lake_executable" not in runtime.configuration
    assert "lake_digest" not in runtime.configuration
