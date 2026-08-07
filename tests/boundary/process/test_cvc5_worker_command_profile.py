from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from jacobian.canonical import loads_strict_json
from jacobian.contracts.smt import SmtUnsatProofFindRequest

_FAKE_CVC5 = """
class InputLanguage:
    SMT_LIB_2_6 = object()


class Solver:
    def setOption(self, _name, _value):
        pass

    def getLogic(self):
        return "QF_UF"


class _Command:
    def __init__(self, name):
        self.name = name

    def isNull(self):
        return self.name is None

    def getCommandName(self):
        return self.name

    def invoke(self, _solver, _symbol_manager):
        return "sat" if self.name == "check-sat" else ""


class InputParser:
    def __init__(self, _solver):
        self.commands = []
        self.index = 0

    def setStringInput(self, _language, text, _name):
        self.commands = [
            line.strip()[1:].split(None, 1)[0].rstrip(")")
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(";")
        ]

    def nextCommand(self):
        if self.index == len(self.commands):
            return _Command(None)
        command = _Command(self.commands[self.index])
        self.index += 1
        return command

    def getSymbolManager(self):
        return object()
"""


def _run_worker(tmp_path: Path, smtlib_text: str) -> subprocess.CompletedProcess[bytes]:
    (tmp_path / "cvc5.py").write_text(_FAKE_CVC5, encoding="utf-8")
    input_path = tmp_path / "input.smt2"
    input_path.write_text(smtlib_text, encoding="ascii")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.sat_smt.cvc5_worker",
            str(input_path),
            str(tmp_path / "proof.alethe"),
            "QF_UF",
            "1000",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_cvc5_worker_accepts_declare_const_from_the_supported_profile(
    tmp_path: Path,
) -> None:
    request = SmtUnsatProofFindRequest(
        logic="QF_UF",
        smtlib_text=(
            "(set-logic QF_UF)\n(declare-const a Bool)\n(assert a)\n(check-sat)\n"
        ),
        resource_budget={"wall_seconds": 1},
    )

    completed = _run_worker(tmp_path, request.smtlib_text)

    assert completed.returncode == 0
    assert completed.stderr == b""
    assert loads_strict_json(completed.stdout) == {
        "protocol": "jacobian.cvc5-worker/v1",
        "solver_status": "SATISFIABLE",
        "proof_written": False,
        "alethe_hole_count": None,
    }


def test_cvc5_worker_still_rejects_commands_outside_the_supported_profile(
    tmp_path: Path,
) -> None:
    completed = _run_worker(
        tmp_path,
        "(set-logic QF_UF)\n(push 1)\n(check-sat)\n",
    )

    assert completed.returncode == 2
    assert completed.stderr == b""
    assert loads_strict_json(completed.stdout) == {
        "protocol": "jacobian.cvc5-worker/v1",
        "error_code": "CVC5_COMMAND_OUTSIDE_PROFILE",
    }
