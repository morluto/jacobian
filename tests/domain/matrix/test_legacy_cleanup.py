"""Prove the pilot domain bundle provides determinant and rank without legacy adapters.

After the authoritative ``RationalMatrix``/``IntegerMatrix`` refactor, the
artifact-backed legacy producers (``jacobian.matrices.capabilities``,
``jacobian.matrices.determinant``, ``jacobian.matrices.rank``) were removed
because they duplicated the inline ``ComputedOperation`` capabilities owned by
``jacobian.domains.matrix_lattice``.  These tests verify that the pilot bundle
alone is sufficient: ``matrix.determinant.compute``, ``matrix.rank.compute``,
and their ExactReplay verifiers all work without any legacy installation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.rationals import rational_payload as _q
from tests.support.services import open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import OperationInstaller


def _matrix(rows: list[list[int]]) -> dict[str, object]:
    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[_q(value) for value in row] for row in rows],
    }


@pytest.fixture
def pilot_matrix_runtime(tmp_path: Path):
    """Open a runtime with only the pilot matrix bundle and its ExactReplay checkers."""
    with open_domain_services(
        tmp_path,
        build_matrix_bundle(),
        checker_authority=__import__(
            "jacobian.runtime",
            fromlist=["CheckerAuthorityMode"],
        ).CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        bundle = build_matrix_bundle()
        installed = OperationInstaller(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
        ).install(bundle)
        adapters, _installation = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.installation.verification,
            services.core.checkers,
            bundles={"matrix": (bundle, installed)},
            authorize=True,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
        yield services


def test_pilot_provides_matrix_determinant_compute_without_legacy(
    pilot_matrix_runtime,
) -> None:
    """matrix.determinant.compute is an inline ComputedOperation from the pilot bundle."""
    result = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": _matrix([[1, 2], [3, 4]])},
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()
    assert result.output["result"]["determinant"] == _q(-2)
    assert result.output["result"]["method"] == "FRACTION_FREE_BAREISS"


def test_pilot_provides_matrix_rank_compute_without_legacy(
    pilot_matrix_runtime,
) -> None:
    """matrix.rank.compute is an inline ComputedOperation from the pilot bundle."""
    result = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute",
            input={"matrix": _matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])},
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()
    assert result.output["result"]["rank"] == 2
    assert result.output["result"]["pivot_columns"] == [0, 1]


def test_pilot_provides_matrix_determinant_verify_without_legacy(
    pilot_matrix_runtime,
) -> None:
    """matrix.determinant.verify comes from the pilot's ExactReplay checker."""
    computed = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": _matrix([[1, 2], [3, 4]])},
        )
    )
    verified = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": {"matrix": _matrix([[1, 2], [3, 4]])},
                "candidate": computed.output["result"],
            },
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_pilot_provides_matrix_rank_verify_without_legacy(
    pilot_matrix_runtime,
) -> None:
    """matrix.rank.verify comes from the pilot's ExactReplay checker."""
    computed = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute",
            input={"matrix": _matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])},
        )
    )
    verified = pilot_matrix_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": {"matrix": _matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])},
                "candidate": computed.output["result"],
            },
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_legacy_matrix_capabilities_module_is_removed() -> None:
    """The dead legacy producer module must not exist."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.matrices.capabilities")


def test_legacy_matrix_determinant_module_is_removed() -> None:
    """The dead legacy determinant checker module must not exist."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.matrices.determinant")


def test_legacy_matrix_rank_module_is_removed() -> None:
    """The dead legacy rank checker module must not exist."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.matrices.rank")
