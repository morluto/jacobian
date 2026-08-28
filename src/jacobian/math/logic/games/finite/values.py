"""Canonical values for finite deterministic terminal-payoff games."""

from __future__ import annotations

import unicodedata
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math._labels import OpaqueLabel

MAX_TERMINAL_GAME_POSITIONS = 4_096
MAX_TERMINAL_GAME_MOVES = 65_536
MAX_TERMINAL_GAME_WORK_UNITS = 3_000_000
MAX_TERMINAL_GAME_RESULT_BYTES = 4 * 1024 * 1024

PositionOwner = Literal["MAX", "MIN", "TERMINAL"]


class DeterministicGamePosition(StrictModel):
    """One position, its controller, and its payoff when terminal."""

    label: OpaqueLabel
    owner: PositionOwner
    payoff: CanonicalRational | None = Field(
        default=None,
        description=(
            "Exact payoff to MAX when owner is TERMINAL; omit it for MAX and "
            "MIN positions."
        ),
    )

    @model_validator(mode="after")
    def require_payoff_exactly_at_terminals(self) -> Self:
        if self.owner == "TERMINAL" and self.payoff is None:
            raise PydanticCustomError(
                "finite_game.terminal_payoff_required",
                "terminal positions require an exact payoff",
            )
        if self.owner != "TERMINAL" and self.payoff is not None:
            raise PydanticCustomError(
                "finite_game.nonterminal_payoff_forbidden",
                "nonterminal positions must not carry a payoff",
            )
        return self


class DeterministicGameMove(StrictModel):
    """One directed move in a deterministic turn-based arena."""

    source: OpaqueLabel
    target: OpaqueLabel


class DeterministicTerminalGame(StrictModel):
    """A complete finite deterministic two-player terminal-payoff game.

    MAX maximizes and MIN minimizes the exact payoff. A play stops on first
    reaching a terminal position. A play that never reaches a terminal has
    ``draw_payoff``. Position order is the public axis for values and strategy
    tie-breaking; moves must be sorted by their endpoint indices on that axis.
    """

    positions: tuple[DeterministicGamePosition, ...] = Field(
        min_length=1,
        max_length=MAX_TERMINAL_GAME_POSITIONS,
        description=(
            "Distinct NFC-normalized labeled positions in the declared order "
            "used by every returned profile."
        ),
    )
    moves: tuple[DeterministicGameMove, ...] = Field(
        max_length=MAX_TERMINAL_GAME_MOVES,
        description=(
            "Distinct directed moves in lexicographic (source index, target "
            "index) order relative to positions; self-loops are allowed only "
            "at nonterminal positions."
        ),
    )
    draw_payoff: CanonicalRational = Field(
        description="Exact payoff to MAX for every infinite play."
    )

    @model_validator(mode="after")
    def require_complete_bounded_arena(self) -> Self:
        labels = tuple(position.label for position in self.positions)
        if any(not unicodedata.is_normalized("NFC", label) for label in labels):
            raise PydanticCustomError(
                "finite_game.labels_not_nfc", "position labels must use Unicode NFC"
            )
        if len(set(labels)) != len(labels):
            raise PydanticCustomError(
                "finite_game.labels_not_distinct", "position labels must be distinct"
            )

        index = {label: offset for offset, label in enumerate(labels)}
        edge_pairs = tuple((move.source, move.target) for move in self.moves)
        if any(
            source not in index or target not in index for source, target in edge_pairs
        ):
            raise PydanticCustomError(
                "finite_game.move_endpoint_unknown",
                "every move endpoint must be a declared position",
            )
        if len(set(edge_pairs)) != len(edge_pairs):
            raise PydanticCustomError(
                "finite_game.moves_not_distinct", "game moves must be distinct"
            )
        canonical_edges = tuple(
            sorted(edge_pairs, key=lambda edge: (index[edge[0]], index[edge[1]]))
        )
        if edge_pairs != canonical_edges:
            raise PydanticCustomError(
                "finite_game.moves_not_canonical",
                "game moves must use canonical declared-position order",
            )

        outgoing = dict.fromkeys(labels, 0)
        for source, _ in edge_pairs:
            outgoing[source] += 1
        for position in self.positions:
            if position.owner == "TERMINAL":
                if outgoing[position.label] != 0:
                    raise PydanticCustomError(
                        "finite_game.terminal_has_moves",
                        "terminal positions must have no outgoing moves",
                    )
            elif outgoing[position.label] == 0:
                raise PydanticCustomError(
                    "finite_game.nonterminal_without_move",
                    "every nonterminal position must have an outgoing move",
                )

        return self


class TerminalGameValueClass(StrictModel):
    """Every declared position having one exact minimax payoff."""

    payoff: CanonicalRational
    positions: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TERMINAL_GAME_POSITIONS,
        description="Positions in their declared game order.",
    )


class StationaryChoice(StrictModel):
    """The successor selected at one player-owned position."""

    position: OpaqueLabel
    target: OpaqueLabel


class DeterministicTerminalGameSolution(StrictModel):
    """All position values and one canonical optimal stationary strategy pair.

    The source game is retained for composition. Deserialization checks the
    bounded result shape only; use the owner-local verifier to check an
    independently supplied minimax claim. Kernel output uses the trusted
    factory below. Value classes are ordered by payoff and list members in
    declared position order. Strategy choices follow the corresponding
    player's declared positions.
    """

    game: DeterministicTerminalGame
    value_classes: tuple[TerminalGameValueClass, ...] = Field(
        min_length=1,
        max_length=MAX_TERMINAL_GAME_POSITIONS,
        description="Nonempty payoff classes in strictly increasing payoff order.",
    )
    max_strategy: tuple[StationaryChoice, ...] = Field(
        max_length=MAX_TERMINAL_GAME_POSITIONS,
        description="One optimal move for every MAX position in declared order.",
    )
    min_strategy: tuple[StationaryChoice, ...] = Field(
        max_length=MAX_TERMINAL_GAME_POSITIONS,
        description="One optimal move for every MIN position in declared order.",
    )

    @model_validator(mode="after")
    def require_bounded_solution_shape(self) -> Self:
        labels = tuple(position.label for position in self.game.positions)
        label_set = set(labels)
        if any(
            not value_class.positions
            or any(label not in label_set for label in value_class.positions)
            for value_class in self.value_classes
        ):
            raise PydanticCustomError(
                "finite_game.value_class_position_unknown",
                "every value-class position must be declared by game",
            )
        flattened = tuple(
            label
            for value_class in self.value_classes
            for label in value_class.positions
        )
        if len(set(flattened)) != len(flattened) or set(flattened) != label_set:
            raise PydanticCustomError(
                "finite_game.value_classes_not_partition",
                "value classes must partition the declared positions",
            )
        payoff_values = tuple(
            value_class.payoff.as_fraction() for value_class in self.value_classes
        )
        if any(later <= earlier for earlier, later in pairwise(payoff_values)):
            raise PydanticCustomError(
                "finite_game.value_classes_not_ordered",
                "value classes must be ordered by strictly increasing payoff",
            )
        successors: dict[OpaqueLabel, set[OpaqueLabel]] = {
            label: set() for label in labels
        }
        for move in self.game.moves:
            successors[move.source].add(move.target)
        for choices, owner, error_code in (
            (self.max_strategy, "MAX", "max_strategy"),
            (self.min_strategy, "MIN", "min_strategy"),
        ):
            expected = tuple(
                position.label
                for position in self.game.positions
                if position.owner == owner
            )
            if tuple(choice.position for choice in choices) != expected:
                raise PydanticCustomError(
                    f"finite_game.{error_code}_positions",
                    f"{error_code} must cover {owner} positions in declared order",
                )
            if any(
                choice.target not in successors[choice.position] for choice in choices
            ):
                raise PydanticCustomError(
                    f"finite_game.{error_code}_move_unknown",
                    f"every {error_code} choice must be a declared move",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        game: DeterministicTerminalGame,
        value_classes: tuple[TerminalGameValueClass, ...],
        max_strategy: tuple[StationaryChoice, ...],
        min_strategy: tuple[StationaryChoice, ...],
    ) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            game=game,
            value_classes=value_classes,
            max_strategy=max_strategy,
            min_strategy=min_strategy,
        )


def _require_terminal_game_envelope(game: DeterministicTerminalGame) -> None:
    """Preflight threshold work and a conservative exact-result wire bound."""

    payoffs = [game.draw_payoff]
    payoffs.extend(
        position.payoff for position in game.positions if position.payoff is not None
    )
    threshold_count = len({(payoff.num, payoff.den) for payoff in payoffs})
    terminal_count = len(payoffs) - 1
    max_digits = max(
        max(len(payoff.num.lstrip("-")), len(payoff.den)) for payoff in payoffs
    )
    position_count = len(game.positions)
    move_count = len(game.moves)

    # Each threshold performs at most four full vertex passes (target choice,
    # attractor setup, safety complement, and value update), one edge pass, and
    # the solve performs one final O(V+E) strategy pass. Result construction
    # replays the same kernel.
    # Forming threshold targets compares every terminal payoff with every
    # threshold. Sorting k exact rationals uses at most k*ceil(log2(k)) more
    # comparisons. Each comparison is conservatively charged quadratically in
    # the largest scalar digit count, while parsing/hashing every supplied
    # scalar is charged linearly.
    comparison_levels = (threshold_count - 1).bit_length()
    rational_work = (
        len(payoffs) * max_digits
        + (threshold_count * terminal_count + threshold_count * comparison_levels)
        * max_digits
        * max_digits
    )
    work_units = 2 * (
        threshold_count * (4 * position_count + move_count)
        + position_count
        + move_count
        + rational_work
    )
    if work_units > MAX_TERMINAL_GAME_WORK_UNITS:
        raise PydanticCustomError(
            "finite_game.threshold_work_exceeded",
            "terminal-game threshold work exceeds the exact work bound "
            f"({work_units} > {MAX_TERMINAL_GAME_WORK_UNITS}); reduce distinct "
            "payoff levels, arena size, or payoff digit length",
        )

    labels = tuple(position.label for position in game.positions)
    longest_label = max(labels, key=lambda label: len(canonicalize_json([label])))
    widest_payoff = max(
        payoffs,
        key=lambda payoff: len(payoff.num) + len(payoff.den),
    )
    max_positions = tuple(
        position for position in game.positions if position.owner == "MAX"
    )
    min_positions = tuple(
        position for position in game.positions if position.owner == "MIN"
    )
    result_upper_bound = {
        "game": game.model_dump(mode="json"),
        "value_classes": [
            {
                "payoff": widest_payoff.model_dump(mode="json"),
                "positions": [position.label],
            }
            for position in game.positions
        ],
        "max_strategy": [
            {"position": position.label, "target": longest_label}
            for position in max_positions
        ],
        "min_strategy": [
            {"position": position.label, "target": longest_label}
            for position in min_positions
        ],
    }
    try:
        canonicalize_json(
            result_upper_bound,
            limits=CanonicalLimits(
                max_output_bytes=MAX_TERMINAL_GAME_RESULT_BYTES,
            ),
        )
    except ValueError as exc:
        raise PydanticCustomError(
            "finite_game.result_size_exceeded",
            "terminal-game solution exceeds the exact result-size bound",
        ) from exc


__all__ = [
    "MAX_TERMINAL_GAME_MOVES",
    "MAX_TERMINAL_GAME_POSITIONS",
    "MAX_TERMINAL_GAME_RESULT_BYTES",
    "MAX_TERMINAL_GAME_WORK_UNITS",
    "DeterministicGameMove",
    "DeterministicGamePosition",
    "DeterministicTerminalGame",
    "DeterministicTerminalGameSolution",
    "PositionOwner",
    "StationaryChoice",
    "TerminalGameValueClass",
]
