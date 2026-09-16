"""Contracts for the one-prime square-free local factor operation."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.affine_forms.values import AffineFormId
from jacobian.math.number_theory.squarefree_affine_forms._kernel import (
    form_solution_profile,
    profile_residues,
)
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_LEDGER_ROWS,
    MAX_LOCAL_FACTOR_PRIME,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    MAX_SQUAREFREE_FORMS,
    SquarefreeAffineFamily,
)


def _invariant_error(message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "number_theory.squarefree_affine.ledger_invariant", message
    )


class SquarefreeLocalFactorRequest(StrictModel):
    """Compute the complete p^2 residue ledger and exact local factor."""

    source: SquarefreeAffineFamily
    prime: StrictInt = Field(
        ge=2,
        description=(
            "Prime modulus base. Primality and the p^2 enumeration envelope "
            f"(at most {MAX_LOCAL_FACTOR_PRIME}) are admitted at execution."
        ),
    )


class SquarefreeFormBadRow(StrictModel):
    """One form's closed-form bad-residue coset modulo p^2."""

    form_id: AffineFormId
    bad_count: StrictInt = Field(ge=0)
    root: StrictInt | None = Field(default=None, ge=0)
    stride: StrictInt | None = Field(default=None, ge=1)


class SquarefreeBadLedgerRow(StrictModel):
    """One bad residue and the sorted forms whose p^2 divisibility excludes it."""

    residue: StrictInt = Field(ge=0)
    form_ids: tuple[AffineFormId, ...] = Field(
        min_length=1, max_length=MAX_SQUAREFREE_FORMS
    )

    @model_validator(mode="after")
    def require_canonical_ids(self) -> Self:
        if self.form_ids != tuple(sorted(set(self.form_ids))):
            raise _invariant_error("ledger form IDs must be distinct and sorted")
        return self


class SquarefreeLocalFactorResult(StrictModel):
    """Complete canonical p^2-residue ledger and the exact local factor."""

    source: SquarefreeAffineFamily
    prime: StrictInt = Field(ge=2, le=MAX_LOCAL_FACTOR_PRIME)
    modulus: StrictInt = Field(ge=4)
    form_rows: tuple[SquarefreeFormBadRow, ...] = Field(
        min_length=1, max_length=MAX_SQUAREFREE_FORMS
    )
    covers_all_residues: StrictBool
    bad_residues: tuple[SquarefreeBadLedgerRow, ...] = Field(max_length=MAX_LEDGER_ROWS)
    bad_count: StrictInt = Field(ge=0)
    valid_count: StrictInt = Field(ge=0)
    overlap_count: StrictInt = Field(ge=0)
    local_factor: CanonicalRational
    has_local_obstruction: StrictBool

    @model_validator(mode="after")
    def require_complete_replayed_ledger(self) -> Self:
        """Replay the defining congruences and the complete partition."""

        modulus = self.prime * self.prime
        if self.modulus != modulus:
            raise _invariant_error("modulus must equal the square of the prime")
        if tuple(row.form_id for row in self.form_rows) != tuple(
            form.form_id for form in self.source.forms
        ):
            raise _invariant_error("form rows must align with the canonical source")
        by_id = {form.form_id: form for form in self.source.forms}
        expected_union: dict[int, list[str]] = {}
        covers_all = False
        for row in self.form_rows:
            expected = form_solution_profile(by_id[row.form_id], self.prime)
            if (row.bad_count, row.root, row.stride) != expected:
                raise _invariant_error(
                    "form bad-residue coset does not replay its defining "
                    "congruence modulo p^2"
                )
            if row.bad_count == modulus:
                covers_all = True
            for residue in profile_residues(expected):
                expected_union.setdefault(residue, []).append(row.form_id)
        if self.covers_all_residues != covers_all:
            raise _invariant_error(
                "full residue coverage must equal one identically vanishing form"
            )
        expected_ledger = (
            ()
            if covers_all
            else tuple(
                SquarefreeBadLedgerRow(
                    residue=residue, form_ids=tuple(sorted(form_ids))
                )
                for residue, form_ids in sorted(expected_union.items())
            )
        )
        if self.bad_residues != expected_ledger:
            raise _invariant_error(
                "bad-residue ledger must equal the complete replayed union of "
                "the per-form cosets"
            )
        expected_bad = modulus if covers_all else len(expected_ledger)
        if self.bad_count != expected_bad:
            raise _invariant_error("bad count must equal the bad-residue union")
        if self.bad_count + self.valid_count != modulus:
            raise _invariant_error(
                "valid and bad counts must partition every residue modulo p^2 "
                "exactly once"
            )
        expected_overlap = sum(1 for row in expected_ledger if len(row.form_ids) > 1)
        if self.overlap_count != expected_overlap:
            raise _invariant_error(
                "overlap count must equal the ledger rows excluded by multiple forms"
            )
        if (
            self.local_factor.as_integer_ratio()
            != Fraction(self.valid_count, modulus).as_integer_ratio()
        ):
            raise _invariant_error(
                "local factor must equal valid_count / p^2 in lowest terms"
            )
        if self.has_local_obstruction != (self.valid_count == 0):
            raise _invariant_error("local obstruction flag must equal valid_count == 0")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: SquarefreeAffineFamily,
        prime: int,
        profiles: tuple[tuple[int, int | None, int | None], ...],
        ledger: tuple[tuple[int, tuple[str, ...]], ...],
        covers_all: bool,
    ) -> Self:
        modulus = prime * prime
        bad_count = modulus if covers_all else len(ledger)
        valid_count = modulus - bad_count
        return cls.model_construct(
            source=source,
            prime=prime,
            modulus=modulus,
            form_rows=tuple(
                SquarefreeFormBadRow(
                    form_id=form.form_id,
                    bad_count=count,
                    root=root,
                    stride=stride,
                )
                for form, (count, root, stride) in zip(
                    source.forms, profiles, strict=True
                )
            ),
            covers_all_residues=covers_all,
            bad_residues=tuple(
                SquarefreeBadLedgerRow(residue=residue, form_ids=form_ids)
                for residue, form_ids in ledger
            ),
            bad_count=bad_count,
            valid_count=valid_count,
            overlap_count=sum(1 for _, form_ids in ledger if len(form_ids) > 1),
            local_factor=CanonicalRational.from_fraction(
                Fraction(valid_count, modulus)
            ),
            has_local_obstruction=valid_count == 0,
        )


__all__ = [
    "SquarefreeBadLedgerRow",
    "SquarefreeFormBadRow",
    "SquarefreeLocalFactorRequest",
    "SquarefreeLocalFactorResult",
]
