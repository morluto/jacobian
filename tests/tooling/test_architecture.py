"""Focused tests for Jacobian's remaining custom source rules."""

from __future__ import annotations

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
    _write(tmp_path, "src/jacobian/math/logic/_operations.py", source)
    _write(tmp_path, "src/jacobian/math/example/_operations.py", source)

    assert _violations(tmp_path, "bounded-process-gateway") == [
        "src/jacobian/math/example/_operations.py",
        "src/jacobian/math/example/_operations.py",
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
