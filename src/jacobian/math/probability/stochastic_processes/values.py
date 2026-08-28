"""Provider-independent values for exact finite stochastic processes."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_SAMPLES = 64


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite stochastic values."""

    return PydanticCustomError(f"finite_stochastic_process.{reason}", message)


class FiniteProbabilitySpace(StrictModel):
    """An immutable finite probability space with positive-mass atoms.

    ``samples`` are unique labels. ``masses`` are positive canonical rationals
    that sum to exactly one.
    """

    samples: tuple[str, ...] = Field(min_length=1, max_length=MAX_SAMPLES)
    masses: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_SAMPLES,
    )

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.samples) != len(self.masses):
            raise _validation_error(
                "sample_mass_length_mismatch",
                "samples and masses must have equal length",
            )
        if len(set(self.samples)) != len(self.samples):
            raise _validation_error("sample_duplicate", "sample labels must be unique")
        total = sum((mass.as_fraction() for mass in self.masses), start=0)
        for mass in self.masses:
            require_bounded_rational(
                mass,
                max_digits=256,
                label="probability mass",
            )
            if mass.as_fraction() <= 0:
                raise _validation_error("mass_nonpositive", "masses must be positive")
        if total != 1:
            raise _validation_error("mass_sum_invalid", "masses must sum to exactly 1")
        return self


class FiniteRandomVariable(StrictModel):
    """An immutable finite random variable on a probability space.

    ``values`` is a tuple of canonical rationals, one per sample, in the same
    order as the probability space's samples.
    """

    space: FiniteProbabilitySpace
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_SAMPLES,
    )

    @model_validator(mode="after")
    def require_valid_rv(self) -> Self:
        if len(self.values) != len(self.space.samples):
            raise _validation_error(
                "random_variable_length_mismatch",
                "values must have one entry per sample",
            )
        return self


class FiniteSigmaAlgebra(StrictModel):
    """An immutable finite sigma algebra represented by its atom partition.

    ``blocks`` is a tuple of frozensets of sample labels. The blocks partition
    the sample space (disjoint, nonempty, union = all samples).
    """

    space: FiniteProbabilitySpace
    blocks: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_partition(self) -> Self:
        all_samples = set()
        seen: set[str] = set()
        for block in self.blocks:
            if not block:
                raise _validation_error(
                    "partition_block_empty", "partition blocks must be nonempty"
                )
            for s in block:
                if s in seen:
                    raise _validation_error(
                        "partition_blocks_overlap", "partition blocks must be disjoint"
                    )
                if s not in self.space.samples:
                    raise _validation_error(
                        "partition_element_outside_space",
                        "block element not in sample space",
                    )
                seen.add(s)
                all_samples.add(s)
        if all_samples != set(self.space.samples):
            raise _validation_error(
                "partition_incomplete", "blocks must partition the entire sample space"
            )
        return self


__all__ = [
    "MAX_SAMPLES",
    "FiniteProbabilitySpace",
    "FiniteRandomVariable",
    "FiniteSigmaAlgebra",
]
