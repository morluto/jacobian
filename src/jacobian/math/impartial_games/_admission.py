"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "game.impartial.birthday.compute",
        AdmissionDecision.KEEP,
        "exact birthday rank certificate for a bounded impartial game position",
    ),
    OperationAdmission(
        "game.impartial.grundy_table.compute",
        AdmissionDecision.KEEP,
        "complete Grundy table over a bounded impartial game state space",
    ),
    OperationAdmission(
        "game.subtraction.grundy_prefix.compute",
        AdmissionDecision.KEEP,
        "complete Grundy prefix with periodic tail certificate for a bounded subtraction game",
    ),
    OperationAdmission(
        "game.nim.nim_sum.compute",
        AdmissionDecision.KEEP,
        "exact bitwise xor nim sum determining the P/N outcome of a Nim position",
    ),
    OperationAdmission(
        "game.impartial.outcome_profile.compute",
        AdmissionDecision.KEEP,
        "complete P/N position partition with Grundy values and terminal positions",
    ),
)
