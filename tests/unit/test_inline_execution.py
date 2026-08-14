from __future__ import annotations

import ast
from pathlib import Path

from jacobian.contracts.domain_operations import InlineOperationOutput
from jacobian.contracts.matrix_operations import (
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
)
from jacobian.contracts.operations import OperationRequest
from jacobian.domains.matrix_lattice.operation_declarations import (
    MATRIX_DETERMINANT_COMPUTE,
)
from jacobian.inline_execution import InlineOperationAdapter, run_inline
from jacobian.operation_binding import DeclaredOperationAdapter
from jacobian.operation_declarations import InlineOperation
from jacobian.operations import Completed


def test_matrix_determinant_is_an_inline_operation() -> None:
    assert isinstance(MATRIX_DETERMINANT_COMPUTE, InlineOperation)
    assert MATRIX_DETERMINANT_COMPUTE.operation_id == "matrix.determinant.compute"


def test_run_inline_computes_determinant_without_a_binder() -> None:
    request = MatrixDeterminantRequest.model_validate(
        {
            "matrix": {
                "entries": [
                    [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                    [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                ]
            }
        }
    )
    terminal = run_inline(MATRIX_DETERMINANT_COMPUTE, request)

    assert isinstance(terminal, Completed)
    assert terminal.value == MatrixDeterminantResult.model_validate(
        {"determinant": {"num": "-6", "den": "1"}}
    )


def test_inline_adapter_is_not_a_declared_adapter() -> None:
    adapter = InlineOperationAdapter(MATRIX_DETERMINANT_COMPUTE)
    assert not isinstance(adapter, DeclaredOperationAdapter)
    result = adapter.invoke(
        adapter.prepare(
            OperationRequest(
                operation_id="matrix.determinant.compute",
                input={
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ]
                    }
                },
            )
        )
    )
    assert result.publication is not None
    assert isinstance(result.publication.output, InlineOperationOutput)
    assert result.publication.artifact_uris == ()


def test_inline_executor_source_omits_control_plane_imports() -> None:
    source = Path("src/jacobian/inline_execution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (
        "jacobian.storage",
        "jacobian.verification",
        "jacobian.sat_smt",
        "jacobian.lean_frontend",
        "jacobian.adapters",
        "jacobian.operation_binding",
        "jacobian.operation_catalog",
        "jacobian.operation_publication",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    leaked = [
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    ]
    assert leaked == []
