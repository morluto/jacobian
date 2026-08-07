"""Isolated cvc5 process used by the bounded Alethe producer adapter."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CVC5_WORKER_PROTOCOL = "jacobian.cvc5-worker/v1"
CVC5_INPUT_LIMIT = 1_000_000
CVC5_PROOF_LIMIT = 6_000_000
_ALLOWED_PARSED_COMMANDS = frozenset(
    {
        "set-logic",
        "declare-sort",
        "declare-fun",
        "declare-const",
        "assert",
        "check-sat",
    }
)
_STATUS_MAP = {
    "sat": "SATISFIABLE",
    "unsat": "UNSATISFIABLE",
    "unknown": "UNKNOWN",
}


class Cvc5WorkerError(RuntimeError):
    """One worker request could not produce usable solver evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _emit(payload: dict[str, object]) -> None:
    encoded = json.dumps(
        {"protocol": CVC5_WORKER_PROTOCOL, **payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _read_bounded(path: Path) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(CVC5_INPUT_LIMIT + 1)
    if len(value) > CVC5_INPUT_LIMIT:
        raise Cvc5WorkerError("CVC5_INPUT_LIMIT_EXCEEDED")
    return value


def _run(
    *,
    input_path: Path,
    proof_path: Path,
    expected_logic: str,
    wall_milliseconds: int,
) -> dict[str, object]:
    try:
        raw_input = _read_bounded(input_path)
        smtlib_text = raw_input.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Cvc5WorkerError("CVC5_INPUT_NOT_ASCII") from exc
    try:
        cvc5: Any = importlib.import_module("cvc5")
        solver = cvc5.Solver()
        solver.setOption("produce-proofs", "true")
        solver.setOption("proof-format-mode", "alethe")
        solver.setOption("tlimit-per", str(wall_milliseconds))
        parser = cvc5.InputParser(solver)
        parser.setStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6,
            smtlib_text,
            "jacobian-input.smt2",
        )
        command_names: list[str] = []
        status_text: str | None = None
        while True:
            command = parser.nextCommand()
            if command.isNull():
                break
            name = str(command.getCommandName())
            if name not in _ALLOWED_PARSED_COMMANDS:
                raise Cvc5WorkerError("CVC5_COMMAND_OUTSIDE_PROFILE")
            if status_text is not None:
                raise Cvc5WorkerError("CVC5_COMMAND_AFTER_CHECK_SAT")
            result = str(command.invoke(solver, parser.getSymbolManager()))
            command_names.append(name)
            if name == "set-logic" and str(solver.getLogic()) != expected_logic:
                raise Cvc5WorkerError("CVC5_LOGIC_MISMATCH")
            if name == "check-sat":
                status_text = result.strip()
    except Cvc5WorkerError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as exc:
        raise Cvc5WorkerError("CVC5_REJECTED_INPUT_OR_FAILED") from exc
    if (
        not command_names
        or command_names[0] != "set-logic"
        or command_names.count("set-logic") != 1
        or command_names[-1] != "check-sat"
        or command_names.count("check-sat") != 1
        or status_text not in _STATUS_MAP
    ):
        raise Cvc5WorkerError("CVC5_QUERY_OUTSIDE_PROFILE")
    solver_status = _STATUS_MAP[status_text]
    if solver_status != "UNSATISFIABLE":
        return {
            "solver_status": solver_status,
            "proof_written": False,
            "alethe_hole_count": None,
        }
    try:
        proofs = solver.getProof(cvc5.ProofComponent.FULL)
        if len(proofs) != 1:
            raise Cvc5WorkerError("CVC5_PROOF_COUNT_INVALID")
        proof = solver.proofToString(proofs[0], cvc5.ProofFormat.ALETHE)
        if not isinstance(proof, bytes):
            raise Cvc5WorkerError("CVC5_PROOF_NOT_BYTES")
        if len(proof) > CVC5_PROOF_LIMIT:
            raise Cvc5WorkerError("CVC5_PROOF_LIMIT_EXCEEDED")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(proof_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(proof)
        finally:
            os.close(descriptor)
    except Cvc5WorkerError:
        raise
    except (OSError, RuntimeError, TypeError) as exc:
        raise Cvc5WorkerError("CVC5_PROOF_CAPTURE_FAILED") from exc
    return {
        "solver_status": solver_status,
        "proof_written": True,
        "alethe_hole_count": proof.count(b":rule hole"),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 4:
        _emit({"error_code": "CVC5_WORKER_ARGUMENTS_INVALID"})
        return 2
    input_path = Path(arguments[0])
    proof_path = Path(arguments[1])
    expected_logic = arguments[2]
    try:
        wall_milliseconds = int(arguments[3])
    except ValueError:
        _emit({"error_code": "CVC5_WORKER_ARGUMENTS_INVALID"})
        return 2
    if (
        expected_logic not in {"QF_UF", "QF_LIA", "QF_LRA"}
        or wall_milliseconds < 1
        or wall_milliseconds > 300_000
    ):
        _emit({"error_code": "CVC5_WORKER_ARGUMENTS_INVALID"})
        return 2
    try:
        result = _run(
            input_path=input_path,
            proof_path=proof_path,
            expected_logic=expected_logic,
            wall_milliseconds=wall_milliseconds,
        )
    except Cvc5WorkerError as exc:
        _emit({"error_code": exc.code})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
