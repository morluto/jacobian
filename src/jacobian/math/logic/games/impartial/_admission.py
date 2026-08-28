"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.logic.games.impartial._tools import TOOLS

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
        "game.nim.options.compute",
        AdmissionDecision.KEEP,
        "complete deduplicated one-move relation for a bounded canonical Nim multiset with indexed source transport",
    ),
    OperationAdmission(
        "game.impartial.outcome_profile.compute",
        AdmissionDecision.KEEP,
        "complete P/N position partition with Grundy values and terminal positions",
    ),
    OperationAdmission(
        "game.impartial.disjunctive_sum.compute",
        AdmissionDecision.KEEP,
        "exact Grundy value of a disjunctive sum by XOR of component Grundy values",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
