"""Typed contracts for all-shortest Petri-net firing sequences."""

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.values import (
    FiringSequence,
    Marking,
    PetriNet,
)

MAX_SHORTEST_PATH_SEQUENCES = 100_000
MAX_SHORTEST_PATH_LENGTH = 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"petri_net.shortest_sequences.{reason}", message)


class ShortestFiringSequencesRequest(StrictModel):
    """Find every minimum-length transition word to a target marking."""

    source_graph: ReachabilityResult
    target_marking: Marking

    @model_validator(mode="after")
    def require_target_axis(self) -> Self:
        if (
            self.target_marking.net is not None
            and self.target_marking.net != self.source_graph.net
        ):
            raise _validation_error(
                "target_parent", "target marking belongs to a different Petri net"
            )
        if len(self.target_marking.tokens) != self.source_graph.net.place_count:
            raise _validation_error(
                "target_axis", "target marking must match the source graph place axis"
            )
        return self


class ShortestFiringSequencesResult(StrictModel):
    """All shortest transition sequences, with their source/target context."""

    net: PetriNet
    initial_marking: Marking
    target_marking: Marking
    shortest_length: int | None = Field(default=None, ge=0, le=MAX_SHORTEST_PATH_LENGTH)
    sequences: tuple[FiringSequence, ...] = Field(
        max_length=MAX_SHORTEST_PATH_SEQUENCES
    )

    @model_validator(mode="after")
    def require_complete_family_shape(self) -> Self:
        if (
            (
                self.initial_marking.net is not None
                and self.initial_marking.net != self.net
            )
            or (
                self.target_marking.net is not None
                and self.target_marking.net != self.net
            )
            or len(self.initial_marking.tokens) != self.net.place_count
            or len(self.target_marking.tokens) != self.net.place_count
        ):
            raise _validation_error(
                "marking_axis", "result markings must match the source net place axis"
            )
        if self.shortest_length is None:
            if self.sequences:
                raise _validation_error(
                    "unreachable_family", "unreachable targets have no sequences"
                )
            return self
        if not self.sequences or any(
            len(sequence.transitions) != self.shortest_length
            or any(
                type(transition) is not int
                or not 0 <= transition < self.net.transition_count
                for transition in sequence.transitions
            )
            for sequence in self.sequences
        ):
            raise _validation_error(
                "sequence_shape",
                "every returned sequence must have the shortest length",
            )
        if tuple(sequence.transitions for sequence in self.sequences) != tuple(
            sorted({sequence.transitions for sequence in self.sequences})
        ):
            raise _validation_error(
                "sequence_order",
                "sequences must be unique and lexicographically ordered",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> "ShortestFiringSequencesResult":
        """Build the admitted path family without replaying it in validation."""
        return cls.model_construct(**values)
