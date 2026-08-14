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
        "from jacobian.domains.matrices import kernels\n",
    )

    assert "contract-dependency-leaf" in _codes(tmp_path)


def test_product_code_cannot_construct_internal_operation_requests(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/polynomials/composition.py",
        "from jacobian.contracts.operations import OperationRequest\n"
        "request = OperationRequest(operation_id='x', input={})\n",
    )

    assert "internal-operation-request" in _codes(tmp_path)


def test_product_code_cannot_construct_public_result_envelopes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/polynomials/adapter.py",
        "from jacobian.contracts.operations import OperationResult\n"
        "result = OperationResult(operation_id='x', operation_version='1', "
        "execution={}, output={})\n",
    )

    assert "operation-result-projection" in _codes(tmp_path)


def test_result_envelope_ratchet_resolves_import_aliases(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/polynomials/aliased_adapter.py",
        "from jacobian.contracts.operations import OperationResult as PublicResult\n"
        "result = PublicResult(operation_id='x', operation_version='1', "
        "execution={}, output={})\n",
    )
    _write(
        tmp_path,
        "src/jacobian/polynomials/qualified_adapter.py",
        "import jacobian.contracts.operations as operations\n"
        "result = operations.OperationResult(operation_id='x', "
        "operation_version='1', execution={}, output={})\n",
    )
    _write(
        tmp_path,
        "src/jacobian/polynomials/parent_alias_adapter.py",
        "import jacobian.contracts as contracts\n"
        "result = contracts.operations.OperationResult(operation_id='x', "
        "operation_version='1', execution={}, output={})\n",
    )
    _write(
        tmp_path,
        "src/jacobian/polynomials/root_alias_adapter.py",
        "import jacobian as j\n"
        "result = j.contracts.operations.OperationResult(operation_id='x', "
        "operation_version='1', execution={}, output={})\n",
    )

    violations = [
        item
        for item in check_architecture(tmp_path).violations
        if item.code == "operation-result-projection"
    ]
    assert {item.path for item in violations} == {
        "src/jacobian/polynomials/aliased_adapter.py",
        "src/jacobian/polynomials/parent_alias_adapter.py",
        "src/jacobian/polynomials/qualified_adapter.py",
        "src/jacobian/polynomials/root_alias_adapter.py",
    }


def test_result_envelope_ratchet_resolves_relative_imports(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/example/adapter.py",
        "from ...contracts.operations import OperationResult\n"
        "result = OperationResult(operation_id='x', operation_version='1', "
        "execution={}, output={})\n",
    )
    _write(
        tmp_path,
        "src/jacobian/domains/example/qualified_adapter.py",
        "from ...contracts import operations as caps\n"
        "result = caps.OperationResult(operation_id='x', "
        "operation_version='1', execution={}, output={})\n",
    )

    violations = [
        item
        for item in check_architecture(tmp_path).violations
        if item.code == "operation-result-projection"
    ]
    assert {item.path for item in violations} == {
        "src/jacobian/domains/example/adapter.py",
        "src/jacobian/domains/example/qualified_adapter.py",
    }


def test_dispatcher_may_construct_public_result_envelopes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/operation_dispatcher.py",
        "from jacobian.contracts.operations import OperationResult\n"
        "result = OperationResult(operation_id='x', operation_version='1', "
        "execution={}, output={})\n",
    )

    assert "operation-result-projection" not in _codes(tmp_path)


def test_product_code_cannot_restore_marker_selected_adapter_modes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/polynomials/adapter.py",
        "class Adapter:\n    typed_input = True\n",
    )
    _write(
        tmp_path,
        "src/jacobian/graphs/adapter.py",
        "from jacobian.operation_adapters import TypedInputAdapter\n",
    )

    violations = [
        item
        for item in check_architecture(tmp_path).violations
        if item.code == "legacy-adapter-mode"
    ]
    assert {item.path for item in violations} == {
        "src/jacobian/graphs/adapter.py",
        "src/jacobian/polynomials/adapter.py",
    }


def test_contracts_can_import_canonical_and_contract_modules(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/value.py",
        "from jacobian.canonical import canonicalize_json\n"
        "from jacobian.contracts.base import ContractModel\n",
    )

    assert "contract-dependency-leaf" not in _codes(tmp_path)


def test_contracts_cannot_bypass_leaf_policy_with_relative_imports(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/value.py",
        "from ..domains import lattices\n",
    )

    assert "contract-dependency-leaf" in _codes(tmp_path)


def test_native_math_cannot_load_runtime_or_operation_layers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices.py",
        "from jacobian.runtime.model import JacobianRuntime\n"
        "from jacobian.adapters.mcp import tooling\n"
        "from jacobian.catalog.collector import CatalogOperationCollector\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_migrated_matrix_math_cannot_import_legacy_domain_kernels(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/matrices/operations.py",
        "from jacobian.domains.matrices import kernels\n",
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
        "from jacobian.domains.matrices import operations\n",
    )

    assert "native-math-boundary" in _codes(tmp_path)


def test_domains_cannot_import_private_math_backends(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/polynomial/operations.py",
        "from jacobian.math.polynomials import _sympy\n",
    )

    assert "private-math-backend" in _codes(tmp_path)


def test_domains_can_import_public_math_packages(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/polynomial/operations.py",
        "from jacobian.math import polynomials\n",
    )

    assert "private-math-backend" not in _codes(tmp_path)


def test_checker_cannot_import_producer_conversions_or_kernels(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/matrix.py",
        "from jacobian.domains.matrices import conversions, kernels\n",
    )

    assert "checker-producer-isolation" in _codes(tmp_path)


def test_checker_cannot_import_a_symbol_below_a_producer_kernel(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/matrix.py",
        "from jacobian.domains.matrices.kernels.linear import solve\n",
    )

    assert "checker-producer-isolation" in _codes(tmp_path)


def test_concrete_contract_generics_are_not_erased_to_contract_model(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/domains/operation.py",
        "from collections.abc import Callable\n"
        "from jacobian.contracts.base import ContractModel\n"
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
        "from jacobian.contracts.base import ContractModel\n"
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


def test_inline_executor_cannot_import_control_plane_layers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/inline_execution.py",
        "from jacobian.adapters.mcp.server import create_server\n",
    )

    assert "inline-executor-boundary" in _codes(tmp_path)
