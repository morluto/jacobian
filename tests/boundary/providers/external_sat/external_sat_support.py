from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tests.support.artifacts import sha256_file
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.providers.external_solver_runtime import (
    cadical_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.runtime import CheckerAuthorityMode
from jacobian.sat_smt.cadical import install_cadical_capabilities
from jacobian.sat_smt.sat_capabilities import (
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)
from jacobian.sat_smt.smt_capabilities import install_smt_unsat_proof_checker


def fake_drat_trim(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "drat-trim"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
            "import sys\n"
            "if '-h' in sys.argv:\n"
            "    print('usage: drat-trim [INPUT] [<PROOF>] [<option> ...]')\n"
            "    raise SystemExit(0)\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    manifest = executable.with_name(executable.name + ".jacobian-runtime.json")
    manifest.write_text(
        (
            "{\n"
            '  "runtime_manifest_version": "1",\n'
            '  "provider": "drat-trim",\n'
            '  "release_tag": "v05.22.2023",\n'
            '  "source_repository": '
            '"https://github.com/marijnheule/drat-trim",\n'
            '  "source_commit": '
            '"2e5e29cb0019d5cfd547d4208dca1b3ec290349f",\n'
            f'  "executable_sha256": "{sha256_file(executable)}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return executable


def fake_carcara(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "carcara"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('carcara 1.1.0 [git master 394edbb]')\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:] == ['check', '--help']:\n"
            "    print('--strict-parsing --parse-hole-args '\n"
            "          '--allow-int-real-subtyping --expand-let-bindings')\n"
            "    raise SystemExit(0)\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    manifest = executable.with_name(executable.name + ".jacobian-runtime.json")
    manifest.write_text(
        (
            "{\n"
            '  "runtime_manifest_version": "1",\n'
            '  "provider": "carcara",\n'
            '  "version": "1.1.0",\n'
            '  "source_repository": "https://github.com/ufmg-smite/carcara",\n'
            '  "source_commit": '
            '"394edbb15ba95c47893f1d821fddde7e016af178",\n'
            '  "compatible_cvc5_version": "1.3.4",\n'
            f'  "executable_sha256": "{sha256_file(executable)}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return executable


@contextmanager
def open_sat_proof_verifier_services(
    root: Path,
    runtime: CapabilityProviderRuntime,
    *,
    checker_authority: CheckerAuthorityMode,
) -> Iterator[DomainTestServices]:
    """Open only the production SAT proof-verifier graph."""

    with open_domain_services(
        root,
        checker_authority=checker_authority,
    ) as services:
        with atomic_installation(services.core):
            verifier, _installation = install_sat_unsat_proof_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.sat,
                services.application.verification,
                services.core.checkers,
                runtime,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if verifier is not None:
                services.installation.register_capability(verifier)
        yield services


@contextmanager
def open_smt_proof_verifier_services(
    root: Path,
    runtime: CapabilityProviderRuntime,
    *,
    checker_authority: CheckerAuthorityMode,
) -> Iterator[DomainTestServices]:
    """Open only the production SMT proof-verifier graph."""

    with open_domain_services(
        root,
        checker_authority=checker_authority,
    ) as services:
        with atomic_installation(services.core):
            verifier, _installation = install_smt_unsat_proof_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.smt,
                services.application.verification,
                services.core.checkers,
                runtime,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if verifier is not None:
                services.installation.register_capability(verifier)
        yield services


@contextmanager
def open_cadical_services(root: Path) -> Iterator[DomainTestServices]:
    """Open the production CaDiCaL producer graph without unrelated bundles."""

    with _open_external_sat_services(root) as services:
        yield services


@contextmanager
def open_verified_external_sat_services(
    root: Path,
) -> Iterator[DomainTestServices]:
    """Open CaDiCaL producers with both production SAT evidence checkers."""

    with _open_external_sat_services(
        root,
        install_assignment_checker=True,
        install_proof_checker=True,
    ) as services:
        yield services


@contextmanager
def open_verified_unsat_services(root: Path) -> Iterator[DomainTestServices]:
    """Open the production CaDiCaL-to-DRAT verification graph."""

    with _open_external_sat_services(
        root,
        install_proof_checker=True,
    ) as services:
        yield services


@contextmanager
def _open_external_sat_services(
    root: Path,
    *,
    install_assignment_checker: bool = False,
    install_proof_checker: bool = False,
) -> Iterator[DomainTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if install_assignment_checker or install_proof_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            cadical = cadical_provider_runtime()
            for adapter in install_cadical_capabilities(services.core.sat, cadical):
                services.installation.register_capability(adapter)

            if install_assignment_checker:
                assignment, _installation = install_sat_assignment_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.sat,
                    services.application.verification,
                    services.core.checkers,
                    authorize_checker=services.installation.authorizes_bundled_checkers,
                )
                if assignment is None:
                    raise RuntimeError("the SAT assignment checker was not installed")
                services.installation.register_capability(assignment)

            if install_proof_checker:
                proof, _proof_installation = install_sat_unsat_proof_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.sat,
                    services.application.verification,
                    services.core.checkers,
                    drat_trim_provider_runtime(),
                    authorize_checker=services.installation.authorizes_bundled_checkers,
                )
                if proof is None:
                    raise RuntimeError("the SAT proof checker was not installed")
                services.installation.register_capability(proof)
        yield services
