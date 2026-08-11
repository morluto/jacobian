"""Architecture checks for composable mathematical-contract boundaries."""

from __future__ import annotations

from pathlib import Path

from tools.check_architecture import check_architecture


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _codes(root: Path) -> set[str]:
    return {violation.code for violation in check_architecture(root).violations}


def test_contracts_cannot_import_runtime_or_domains(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/value.py",
        "from jacobian.runtime.model import JacobianRuntime\n"
        "from jacobian.domains.matrix_lattice import kernels\n",
    )

    assert "contract-dependency-leaf" in _codes(tmp_path)


def test_contracts_can_import_canonical_and_contract_modules(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/value.py",
        "from jacobian.canonical import canonicalize_json\n"
        "from jacobian.contracts.results import ContractModel\n",
    )

    assert "contract-dependency-leaf" not in _codes(tmp_path)


def test_contracts_cannot_bypass_leaf_policy_with_relative_imports(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/value.py",
        "from ..domains import matrix_lattice\n",
    )

    assert "contract-dependency-leaf" in _codes(tmp_path)


def test_native_math_cannot_load_runtime_or_capability_layers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices.py",
        "from jacobian.runtime.model import JacobianRuntime\n"
        "from jacobian.adapters.mcp import tooling\n"
        "from jacobian.capability_service import CapabilityService\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_migrated_matrix_math_cannot_import_legacy_domain_kernels(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices/operations.py",
        "from jacobian.domains.matrix_lattice import kernels\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_native_math_cannot_bypass_isolation_with_relative_runtime_import(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices.py",
        "from ..runtime import JacobianRuntime\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_native_math_cannot_import_domain_operations(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices.py",
        "from jacobian.domains.matrix_lattice import operations\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_checker_cannot_import_producer_conversions_or_kernels(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/matrix.py",
        "from jacobian.domains.matrix_lattice import conversions, kernels\n",
    )

    assert "checker-producer-isolation" in _codes(tmp_path)


def test_checker_cannot_import_a_symbol_below_a_producer_kernel(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/matrix.py",
        "from jacobian.domains.matrix_lattice.kernels.linear import solve\n",
    )

    assert "checker-producer-isolation" in _codes(tmp_path)


def test_concrete_contract_generics_are_not_erased_to_contract_model(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/operation.py",
        "from collections.abc import Callable\n"
        "from jacobian.contracts.results import ContractModel\n"
        "operation: Callable[[ContractModel], ContractModel]\n",
    )

    assert "erased-contract-operation" in _codes(tmp_path)


def test_qualified_callable_cannot_erase_operation_contract_types(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/operation.py",
        "import typing\n"
        "from jacobian.contracts.results import ContractModel\n"
        "operation: typing.Callable[[ContractModel], ContractModel]\n",
    )

    assert "erased-contract-operation" in _codes(tmp_path)


def test_superseded_matrix_contract_variants_are_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/matrix_operation.py",
        "from jacobian.contracts.matrix_operations import RationalOutputMatrix\n"
        "result: RationalOutputMatrix\n",
    )

    assert "output-only-contract" in _codes(tmp_path)


def test_qualified_superseded_matrix_contract_variants_are_rejected(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/matrix_operation.py",
        "import jacobian.contracts.matrix_operations as matrix_operations\n"
        "result: matrix_operations.RationalOutputMatrix\n",
    )

    assert "output-only-contract" in _codes(tmp_path)
