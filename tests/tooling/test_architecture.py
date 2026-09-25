"""Focused tests for Jacobian's remaining custom source rules."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from tools.check_architecture import (
    ArchitecturePolicyError,
    assert_architecture,
    check_architecture,
)


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _violations(root: Path, code: str) -> list[str]:
    return [
        violation.path
        for violation in check_architecture(root).violations
        if violation.code == code
    ]


def test_production_asserts_are_rejected_but_text_is_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "# assert this is documentation\n"
        "message = 'assert this is data'\n"
        "def compute(value):\n"
        "    assert value > 0\n"
        "    return value\n",
    )

    assert _violations(tmp_path, "semantic-production-assert") == [
        "src/jacobian/math/example.py"
    ]


def test_production_assert_rule_has_no_path_allowlist(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_kernel.py",
        "def compute(value):\n    assert value\n",
    )
    _write(
        tmp_path,
        "src/jacobian/process.py",
        "def supervise(value):\n    assert value\n",
    )

    assert _violations(tmp_path, "semantic-production-assert") == [
        "src/jacobian/math/example/_kernel.py",
        "src/jacobian/process.py",
    ]


def test_generic_private_operation_shadows_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/math/example/_operations.py", "pass\n")
    _write(tmp_path, "src/jacobian/math/example/operations.py", "pass\n")
    _write(tmp_path, "src/jacobian/math/example/_singular.py", "pass\n")

    assert _violations(tmp_path, "generic-operation-shadow") == [
        "src/jacobian/math/example/_operations.py"
    ]


def test_mathematical_contracts_cannot_own_output_byte_limits(
    tmp_path: Path,
) -> None:
    codec_source = "from jacobian.canonical import CanonicalLimits\nlimit = CanonicalLimits().max_output_bytes\n"
    named_source = "MAX_PROFILE_RESULT_BYTES = 1024\n"
    helper_source = "def estimate_result_bytes():\n    return 1024\n"
    _write(tmp_path, "src/jacobian/math/example/values.py", codec_source)
    _write(tmp_path, "src/jacobian/math/example/operations.py", named_source)
    _write(tmp_path, "src/jacobian/math/example/_models.py", helper_source)
    _write(
        tmp_path, "src/jacobian/math/example/_worker.py", codec_source + named_source
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/_worker_bounds.py",
        "predicted_worker_output_bytes = 1024\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/_process_adapter.py",
        "from jacobian.process import run_bounded_process\n" + named_source,
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/_process_channel.py",
        "from jacobian.process import run_bounded_process\n"
        "worker_stdout_bytes = 1024\n",
    )

    assert _violations(tmp_path, "mathematical-transport-limit") == [
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_process_adapter.py",
        "src/jacobian/math/example/operations.py",
        "src/jacobian/math/example/values.py",
    ]


def test_direct_process_use_is_confined_to_process_owner(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/process.py", "import subprocess\n")
    _write(tmp_path, "src/jacobian/math/example.py", "import subprocess\n")

    assert _violations(tmp_path, "subprocess-confined") == [
        "src/jacobian/math/example.py"
    ]


def test_bounded_process_gateway_requires_external_tool_owner(tmp_path: Path) -> None:
    source = (
        "from jacobian.process import run_bounded_process\n"
        "run_bounded_process(['tool'])\n"
    )
    _write(
        tmp_path,
        "src/jacobian/math/_singular.py",
        source,
    )
    _write(tmp_path, "src/jacobian/backends.py", source)
    _write(tmp_path, "src/jacobian/math/logic/_sat.py", source)
    _write(tmp_path, "src/jacobian/math/example/__init__.py", source)
    _write(tmp_path, "src/jacobian/math/example/_helpers.py", source)
    _write(tmp_path, "src/jacobian/math/example/_operations.py", source)

    assert _violations(tmp_path, "bounded-process-gateway") == [
        "src/jacobian/math/example/__init__.py",
        "src/jacobian/math/example/__init__.py",
        "src/jacobian/math/example/_helpers.py",
        "src/jacobian/math/example/_helpers.py",
        "src/jacobian/math/example/_operations.py",
        "src/jacobian/math/example/_operations.py",
    ]


def test_checked_worker_gateway_requires_external_tool_owner(tmp_path: Path) -> None:
    source = (
        "from jacobian.process import run_checked_worker_process\n"
        "run_checked_worker_process(['tool'], input_bytes=b'', timeout_seconds=1, "
        "environment={}, stdout_limit=1, stderr_limit=1, decode_result=lambda value: value)\n"
    )
    _write(tmp_path, "src/jacobian/math/example/_solver_process.py", source)
    _write(tmp_path, "src/jacobian/math/example/operations.py", source)

    assert _violations(tmp_path, "bounded-process-gateway") == [
        "src/jacobian/math/example/operations.py",
        "src/jacobian/math/example/operations.py",
    ]


def test_worker_dialogue_gateway_requires_an_already_supervised_worker(
    tmp_path: Path,
) -> None:
    source = (
        "from jacobian.process import run_bounded_worker_dialogue\n"
        "run_bounded_worker_dialogue(['tool'], lambda dialogue: None)\n"
    )
    _write(tmp_path, "src/jacobian/math/example/_solver_worker.py", source)
    _write(tmp_path, "src/jacobian/math/example/_solver_process.py", source)
    _write(tmp_path, "src/jacobian/math/example/_helpers.py", source)

    assert _violations(tmp_path, "bounded-process-gateway") == [
        "src/jacobian/math/example/_helpers.py",
        "src/jacobian/math/example/_helpers.py",
        "src/jacobian/math/example/_solver_process.py",
        "src/jacobian/math/example/_solver_process.py",
    ]


def test_executable_resolution_requires_external_tool_owner(tmp_path: Path) -> None:
    source = "import shutil\nshutil.which('tool')\n"
    _write(tmp_path, "src/jacobian/math/example/_solver_backend.py", source)
    _write(tmp_path, "src/jacobian/math/example/__init__.py", source)
    _write(tmp_path, "src/jacobian/math/example/_helpers.py", source)

    assert _violations(tmp_path, "shutil-which-resolver") == [
        "src/jacobian/math/example/__init__.py",
        "src/jacobian/math/example/_helpers.py",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "import os\nenvironment = dict(os.environ)\n",
        "import os\nenvironment = os.environ.copy()\n",
        "import os\nenvironment = {**os.environ, 'SAFE': '1'}\n",
    ],
)
def test_ambient_environment_spreading_is_rejected(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "src/jacobian/worker.py", source)

    assert _violations(tmp_path, "environ-spreading") == ["src/jacobian/worker.py"]


def test_selective_environment_reads_are_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/process.py",
        "import os\npath = os.environ.get('PATH')\n",
    )

    assert check_architecture(tmp_path).ok


def test_direct_canonical_wire_conversion_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "numerator = int(value.num)\ndenominator = str(value.den)\n",
    )

    assert _violations(tmp_path, "unsafe-canonical-conversion") == [
        "src/jacobian/math/example.py",
        "src/jacobian/math/example.py",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "sympify(caller_input)\n",
        "sympy.sympify(caller_input)\n",
        "parse_expr(caller_input)\n",
        "eval(caller_input)\n",
        "exec(caller_input)\n",
        "lambdify(axis, caller_input)\n",
        "import builtins\nbuiltins.eval(caller_input)\n",
        "import builtins as b\nb.exec(caller_input)\n",
        "from builtins import eval as evaluate\nevaluate(caller_input)\n",
        "from sympy import sympify as parse\nparse(caller_input)\n",
        "import builtins\nevaluate = builtins.eval\nevaluate(caller_input)\n",
        (
            "import builtins\n"
            "evaluate = execute = builtins.eval\n"
            "evaluate(caller_input)\n"
        ),
        "import sympy\nparse = sympy.sympify\nparse(caller_input)\n",
        (
            "from sympy import sympify as parse\n"
            "evaluate: object = parse\n"
            "evaluate(caller_input)\n"
        ),
    ],
)
def test_evaluator_capable_parsers_are_forbidden_in_math_tree(
    tmp_path: Path, source: str
) -> None:
    _write(tmp_path, "src/jacobian/math/example.py", source)

    assert _violations(tmp_path, "evaluator-capable-parser") == [
        "src/jacobian/math/example.py"
    ]


def test_backend_eval_methods_are_not_confused_with_python_eval(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "value = polynomial.eval(point)\n"
        "result = model.eval(variable)\n"
        "items.append(value)\n"
        "mapping.keys()\n",
    )

    assert check_architecture(tmp_path).ok


def test_contracts_and_values_cannot_reenter_their_own_operations(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "from jacobian.math.example.operations import compute\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/relative/_models.py",
        "from .operations import compute\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/imported/_models.py",
        "import jacobian.math.imported._operations as operations\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/values.py",
        "import importlib as loader\n"
        "operation = loader.import_module('jacobian.math.example._operations')\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/other.py",
        "from jacobian.math.example.operations import compute\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/other/_models.py",
        "from jacobian.math.example.operations import compute\n",
    )

    assert _violations(tmp_path, "owner-operation-reentry") == [
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/values.py",
        "src/jacobian/math/imported/_models.py",
        "src/jacobian/math/relative/_models.py",
    ]


def test_result_validators_cannot_replay_known_owner_kernels(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "class ExampleResult:\n"
        "    @model_validator(mode='after')\n"
        "    def replay(self):\n"
        "        return factorizations((3, 5), 15)\n"
        "class ExampleRequest:\n"
        "    @model_validator(mode='after')\n"
        "    def admission(self):\n"
        "        return factorizations((3, 5), 15)\n",
    )

    assert _violations(tmp_path, "result-validator-replay") == [
        "src/jacobian/math/example/_models.py"
    ]


def test_model_validators_cannot_import_backends_or_processes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "class ExampleValue:\n"
        "    @model_validator(mode='after')\n"
        "    def verify(self):\n"
        "        from flint import fmpz_mat\n"
        "        return fmpz_mat(1, 1)\n"
        "class ExampleRequest:\n"
        "    @model_validator(mode='before')\n"
        "    def launch(cls, value):\n"
        "        import subprocess\n"
        "        return value\n",
    )

    assert _violations(tmp_path, "validator-backend-import") == [
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_models.py",
    ]


def test_model_validators_cannot_hide_relative_backend_imports(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "class ExampleValue:\n"
        "    @model_validator(mode='after')\n"
        "    def verify(self):\n"
        "        from ._backend import solve\n"
        "        from ._process import run\n"
        "        from . import _backend\n"
        "        from jacobian import process\n"
        "        return self\n",
    )

    assert _violations(tmp_path, "validator-backend-import") == [
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_models.py",
    ]


def test_profiles_and_certificates_cannot_hide_kernel_replay(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "class ExampleProfile:\n"
        "    @model_validator(mode='after')\n"
        "    def replay(self):\n"
        "        return factor_list(self.polynomial)\n"
        "class ExampleCertificate:\n"
        "    @model_validator(mode='after')\n"
        "    def replay(self):\n"
        "        return minimal_polynomial(self.value)\n",
    )

    assert _violations(tmp_path, "result-validator-replay") == [
        "src/jacobian/math/example/_models.py",
        "src/jacobian/math/example/_models.py",
    ]


@pytest.mark.parametrize(
    "export_source",
    (
        "from .native import value\n",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from .native import value\n",
    ),
    ids=("eager", "lazy-typed"),
)
def test_exported_native_functions_do_not_construct_wire_models(
    tmp_path: Path,
    export_source: str,
) -> None:
    _write(tmp_path, "src/jacobian/math/__init__.py", "__all__ = ['example']\n")
    _write(
        tmp_path,
        "src/jacobian/math/example/__init__.py",
        export_source + "__all__ = ['value']\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/native.py",
        "from jacobian.math.example._models import ExampleRequest\n"
        "def value():\n"
        "    return ExampleRequest()\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/_mcp_adapter.py",
        "from jacobian.math.example._models import ExampleRequest\n"
        "def parse():\n"
        "    return ExampleRequest()\n",
    )

    assert _violations(tmp_path, "native-wire-boundary") == [
        "src/jacobian/math/example/native.py"
    ]


@pytest.mark.parametrize(
    "export_source",
    (
        "from .api import value\n",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from .api import value\n",
    ),
    ids=("eager", "lazy-typed"),
)
def test_exported_native_functions_do_not_annotate_wire_models(
    tmp_path: Path,
    export_source: str,
) -> None:
    _write(tmp_path, "src/jacobian/math/__init__.py", "__all__ = ['example']\n")
    _write(
        tmp_path,
        "src/jacobian/math/example/__init__.py",
        export_source + "__all__ = ['value']\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/api.py",
        "from jacobian.math.example._models import ExampleRequest\n"
        "def value(request: ExampleRequest) -> ExampleRequest:\n"
        "    return request\n",
    )

    assert _violations(tmp_path, "native-wire-boundary") == [
        "src/jacobian/math/example/api.py",
        "src/jacobian/math/example/api.py",
    ]


def test_operations_modules_do_not_use_wire_models(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/operations.py",
        "class ExampleRequest: pass\n"
        "def compute(request: ExampleRequest) -> ExampleRequest:\n"
        "    return ExampleRequest()\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/_tools.py",
        "class ExampleRequest: pass\n"
        "def compute(request: ExampleRequest) -> ExampleRequest:\n"
        "    return ExampleRequest()\n",
    )

    assert _violations(tmp_path, "operations-wire-boundary") == [
        "src/jacobian/math/example/operations.py",
        "src/jacobian/math/example/operations.py",
        "src/jacobian/math/example/operations.py",
    ]


def test_wire_inputs_cannot_bypass_validation_with_model_construct(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example/_models.py",
        "class ExampleRequest:\n"
        "    @classmethod\n"
        "    def model_construct(cls): return cls()\n"
        "class ExampleResult:\n"
        "    @classmethod\n"
        "    def model_construct(cls): return cls()\n"
        "def trusted():\n"
        "    request = ExampleRequest.model_construct()\n"
        "    result = ExampleResult.model_construct()\n",
    )

    assert _violations(tmp_path, "trusted-wire-construction") == [
        "src/jacobian/math/example/_models.py"
    ]


def test_imported_native_domain_must_be_in_root_surface(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/__init__.py",
        "from jacobian.math import example\n__all__ = []\n",
    )
    _write(tmp_path, "src/jacobian/math/example/__init__.py", "__all__ = []\n")

    assert _violations(tmp_path, "native-root-export") == [
        "src/jacobian/math/__init__.py"
    ]


def test_exported_native_functions_do_not_call_public_compute_adapters(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "src/jacobian/math/__init__.py", "__all__ = ['example']\n")
    _write(
        tmp_path,
        "src/jacobian/math/example/__init__.py",
        "from .native import value\n__all__ = ['value']\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/native.py",
        "from jacobian.math.example.operations import compute_value\n"
        "def value():\n"
        "    return compute_value()\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/operations.py",
        "def compute_value():\n    return 1\n",
    )
    _write(
        tmp_path,
        "src/jacobian/math/example/private.py",
        "from jacobian.math.example._operations import compute_value\n"
        "def helper():\n"
        "    return compute_value()\n",
    )

    assert _violations(tmp_path, "native-wire-boundary") == [
        "src/jacobian/math/example/native.py"
    ]


def test_direct_rational_result_formatting_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "result = CanonicalRational(\n"
        "    num=str(value.numerator), den=str(value.denominator)\n"
        ")\n",
    )

    assert len(_violations(tmp_path, "unsafe-canonical-rational-output")) == 2


def test_canonical_result_formatter_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "result = CanonicalRational(\n"
        "    num=format_canonical_integer(value.numerator),\n"
        "    den=format_canonical_integer(value.denominator),\n"
        ")\n",
    )

    assert check_architecture(tmp_path).ok


def test_report_is_sorted_and_assertion_raises(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/z.py", "import subprocess\n")
    _write(tmp_path, "src/jacobian/a.py", "import subprocess\n")

    report = check_architecture(tmp_path)
    assert [item.path for item in report.violations] == [
        "src/jacobian/a.py",
        "src/jacobian/z.py",
    ]
    assert "subprocess-confined" in report.render()
    with pytest.raises(ArchitecturePolicyError):
        assert_architecture(tmp_path)


def test_file_rules_share_one_module_walk_without_changing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path,
        "src/jacobian/math/example.py",
        "import subprocess\n"
        "import os\n"
        "import shutil\n"
        "from jacobian.process import run_bounded_process\n"
        "from sympy import sympify as parse\n"
        "\n"
        "def compute(value):\n"
        "    assert value > 0\n"
        "    environment = dict(os.environ)\n"
        "    shutil.which('solver')\n"
        "    parsed = parse(value)\n"
        "    run_bounded_process(['solver'])\n"
        "    return str(value.num)\n",
    )

    original_walk = ast.walk
    module_walks = 0

    def count_module_walks(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal module_walks
        if isinstance(node, ast.Module):
            module_walks += 1
        yield from original_walk(node)

    monkeypatch.setattr(ast, "walk", count_module_walks)
    report = check_architecture(tmp_path)

    assert module_walks == 1
    assert [(item.code, item.line, item.message) for item in report.violations] == [
        (
            "subprocess-confined",
            1,
            "direct subprocess use belongs in jacobian.process",
        ),
        (
            "bounded-process-gateway",
            4,
            "run_bounded_process requires a concrete external-tool owner",
        ),
        (
            "semantic-production-assert",
            8,
            "production code must use an explicit stable failure, not assert",
        ),
        (
            "environ-spreading",
            9,
            "copy only explicitly allowed environment variables",
        ),
        (
            "shutil-which-resolver",
            10,
            "external executable discovery requires a concrete tool owner",
        ),
        (
            "evaluator-capable-parser",
            11,
            "parse is forbidden in public mathematical input flows",
        ),
        (
            "bounded-process-gateway",
            12,
            "run_bounded_process requires a concrete external-tool owner",
        ),
        (
            "unsafe-canonical-conversion",
            13,
            "use the canonical conversion API for rational wire components",
        ),
    ]
