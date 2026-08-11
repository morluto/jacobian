"""Isolated cvc5 process used by the bounded Alethe producer adapter."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from jacobian.sat_smt.cvc5_protocol import (
    CVC5_WORKER_PROTOCOL,
    Cvc5SatisfiableWorkerResult,
    Cvc5UnknownWorkerResult,
    Cvc5UnsatisfiableWorkerResult,
    Cvc5WorkerResult,
)

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


def _parse_and_solve(
    solver: Any,
    parser: Any,
    expected_logic: str,
) -> tuple[list[str], str | None]:
    """Parse and invoke SMT-LIB commands, returning command names and status."""

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
    return command_names, status_text


def _validate_query_profile(
    command_names: list[str],
    status_text: str | None,
) -> str:
    """Validate the command sequence and return the mapped solver status."""

    if (
        not command_names
        or command_names[0] != "set-logic"
        or command_names.count("set-logic") != 1
        or command_names[-1] != "check-sat"
        or command_names.count("check-sat") != 1
        or status_text not in _STATUS_MAP
    ):
        raise Cvc5WorkerError("CVC5_QUERY_OUTSIDE_PROFILE")
    assert status_text is not None
    return _STATUS_MAP[status_text]


def _capture_cvc5_proof(
    cvc5: Any,
    solver: Any,
    proof_path: Path,
) -> int:
    """Capture the Alethe proof and return the hole count."""

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
        descriptor = -1
        try:
            descriptor = os.open(proof_path, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(proof)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except Cvc5WorkerError:
        raise
    except (OSError, RuntimeError, TypeError) as exc:
        raise Cvc5WorkerError("CVC5_PROOF_CAPTURE_FAILED") from exc
    return proof.count(b":rule hole")


def _run(
    *,
    input_path: Path,
    proof_path: Path,
    expected_logic: str,
    wall_milliseconds: int,
) -> Cvc5WorkerResult:
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
        command_names, status_text = _parse_and_solve(solver, parser, expected_logic)
    except Cvc5WorkerError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as exc:
        raise Cvc5WorkerError("CVC5_REJECTED_INPUT_OR_FAILED") from exc
    solver_status = _validate_query_profile(command_names, status_text)
    if solver_status == "SATISFIABLE":
        return Cvc5SatisfiableWorkerResult(
            protocol="jacobian.cvc5-worker/v1",
            solver_status="SATISFIABLE",
            proof_written=False,
            alethe_hole_count=None,
        )
    if solver_status == "UNKNOWN":
        return Cvc5UnknownWorkerResult(
            protocol="jacobian.cvc5-worker/v1",
            solver_status="UNKNOWN",
            proof_written=False,
            alethe_hole_count=None,
        )
    alethe_hole_count = _capture_cvc5_proof(cvc5, solver, proof_path)
    return Cvc5UnsatisfiableWorkerResult(
        protocol="jacobian.cvc5-worker/v1",
        solver_status="UNSATISFIABLE",
        proof_written=True,
        alethe_hole_count=alethe_hole_count,
    )


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
    _emit(result.model_dump(mode="json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
