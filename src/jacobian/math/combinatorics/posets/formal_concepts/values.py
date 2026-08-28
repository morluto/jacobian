"""Provider-independent values for exact formal concept analysis."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_OBJECTS = 64
MAX_ATTRIBUTES = 64
MAX_IMPLICATIONS = 256
MAX_IMPLICATION_MEMBERSHIPS = 4_096

# One synchronous forward-chaining round rescans the whole canonical family,
# and the closing satisfaction scan does so once more; every productive round
# adds at least one carrier attribute, so exact logical work is at most
# ``(carrier_size + 1) * (rows + memberships)``.  Admission below bounds
# exactly that product, so the accepted implication-system carrier axis is
# limited by predicted work and serialized-result size rather than by a fixed
# attribute count.  This budget admits a full-budget family (256 rows and
# 4,096 memberships) over a 64-attribute carrier and proportionally smaller
# families over larger carriers.
MAX_FORWARD_CHAIN_WORK = 2 * 65 * (MAX_IMPLICATIONS + MAX_IMPLICATION_MEMBERSHIPS)
MAX_CANONICAL_CLOSURE_WORK = MAX_FORWARD_CHAIN_WORK // 2
MAX_IMPLICATION_CLOSURE_RESULT_BYTES = 128 * 1_024

# Attribute indices are carrier members: each owning model validator below
# checks them against the declared attribute axis instead of a fixed ceiling.
_AttributeIndex = Annotated[StrictInt, Field(ge=0)]


class AttributeImplication(StrictModel):
    """One finite implication ``premise -> conclusion`` over attribute indices.

    Member order is immaterial and is canonicalized.  Attributes already in the
    premise are removed from the conclusion, so the stored conclusion contains
    exactly the attributes the implication can add.
    """

    premise: tuple[_AttributeIndex, ...] = Field(
        default=(),
        description=(
            "Attribute indices in the implication premise; order is immaterial "
            "and duplicate indices are invalid."
        ),
    )
    conclusion: tuple[_AttributeIndex, ...] = Field(
        default=(),
        description=(
            "Attribute indices concluded by the premise; premise members are "
            "removed and order is immaterial."
        ),
    )

    @model_validator(mode="after")
    def canonicalize_members(self) -> Self:
        if len(set(self.premise)) != len(self.premise):
            raise PydanticCustomError(
                "formal_concept_analysis.implication_premise_not_unique",
                "implication premise indices must be unique",
            )
        if len(set(self.conclusion)) != len(self.conclusion):
            raise PydanticCustomError(
                "formal_concept_analysis.implication_conclusion_not_unique",
                "implication conclusion indices must be unique",
            )
        premise = tuple(sorted(self.premise))
        conclusion = tuple(sorted(set(self.conclusion) - set(premise)))
        object.__setattr__(self, "premise", premise)
        object.__setattr__(self, "conclusion", conclusion)
        return self


class FiniteAttributeImplicationSystem(StrictModel):
    """A bounded finite attribute carrier with a canonical implication family.

    The normalized family has at most 256 duplicate-free rows and at most 4,096
    premise-plus-conclusion memberships in aggregate.  The carrier axis has no
    fixed count: admission derives its envelope from predicted forward-chaining
    work and serialized-result size.
    """

    attributes: tuple[OpaqueLabel, ...] = Field(
        description=(
            "Ordered unique attribute labels. Implication and subset indices "
            "refer to this axis. The accepted carrier size is bounded by the "
            "predicted work and serialized-result budgets enforced below, not "
            "by a fixed attribute count."
        ),
    )
    implications: tuple[AttributeImplication, ...] = Field(
        default=(),
        max_length=MAX_IMPLICATIONS,
        description=(
            f"Finite implication family with at most {MAX_IMPLICATION_MEMBERSHIPS:,} aggregate premise and "
            "normalized-conclusion memberships. Row order is immaterial and is "
            "canonicalized lexicographically; duplicate rows after normalization "
            "are invalid."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_canonical_system(self) -> Self:
        if len(set(self.attributes)) != len(self.attributes):
            raise PydanticCustomError(
                "formal_concept_analysis.attribute_labels_not_unique",
                "attribute labels must be unique",
            )

        attribute_count = len(self.attributes)
        canonical = tuple(
            sorted(
                self.implications,
                key=lambda implication: (
                    implication.premise,
                    implication.conclusion,
                ),
            )
        )
        if len(set(canonical)) != len(canonical):
            raise PydanticCustomError(
                "formal_concept_analysis.implication_rows_not_unique",
                "implication rows must be duplicate-free after normalization",
            )

        memberships = 0
        for implication in canonical:
            for attribute in implication.premise + implication.conclusion:
                if attribute >= attribute_count:
                    raise PydanticCustomError(
                        "formal_concept_analysis.implication_attribute_out_of_range",
                        "implication attribute index is outside the declared carrier",
                    )
            memberships += len(implication.premise) + len(implication.conclusion)
        if memberships > MAX_IMPLICATION_MEMBERSHIPS:
            raise PydanticCustomError(
                "formal_concept_analysis.membership_budget_exceeded",
                "implication memberships exceed the bounded aggregate membership "
                f"limit of {MAX_IMPLICATION_MEMBERSHIPS}",
            )

        predicted_work = 2 * (attribute_count + 1) * (len(canonical) + memberships)
        if predicted_work > MAX_FORWARD_CHAIN_WORK:
            raise PydanticCustomError(
                "formal_concept_analysis.forward_chain_work_exceeded",
                "predicted forward-chaining work exceeds the bounded limit of "
                f"{MAX_FORWARD_CHAIN_WORK}",
            )

        # Conservative strict-JSON envelope: four UTF-8 bytes per label code
        # point, row framing plus carrier indices for the retained system, and
        # enough per-attribute space for seed/closure/added indices and a
        # complete three-field lineage row.  The fixed allowance covers result
        # keys and the largest admitted work counters.
        predicted_result_bytes = (
            4_096
            + sum(4 * len(label) for label in self.attributes)
            + 96 * len(canonical)
            + 12 * memberships
            + 128 * attribute_count
        )
        if predicted_result_bytes > MAX_IMPLICATION_CLOSURE_RESULT_BYTES:
            raise PydanticCustomError(
                "formal_concept_analysis.result_size_exceeded",
                "predicted closure result exceeds the aggregate serialized-result "
                f"limit of {MAX_IMPLICATION_CLOSURE_RESULT_BYTES} bytes",
            )

        object.__setattr__(self, "implications", canonical)
        return self

    @property
    def total_memberships(self) -> int:
        """Return the total premise and normalized-conclusion memberships."""

        return sum(
            len(implication.premise) + len(implication.conclusion)
            for implication in self.implications
        )


class ImplicationDerivation(StrictModel):
    """The first canonical implication deriving one added attribute."""

    attribute: _AttributeIndex
    implication_index: StrictInt = Field(ge=0, lt=MAX_IMPLICATIONS)
    # Each productive round adds at least one carrier attribute, so the
    # lineage replay below rejects any round count the carrier cannot justify.
    activation_round: StrictInt = Field(ge=1)


class ImplicationClosureWork(StrictModel):
    """Exact cost of canonical synchronous closure construction.

    These logical counts are defined by the public replay procedure, independent
    of whether the private closure kernel uses repeated scans or counters.
    """

    productive_rounds: StrictInt = Field(ge=0)
    canonical_implication_checks: StrictInt = Field(
        ge=0,
        le=MAX_CANONICAL_CLOSURE_WORK,
    )
    canonical_membership_checks: StrictInt = Field(
        ge=0,
        le=MAX_CANONICAL_CLOSURE_WORK,
    )
    total_logical_work: StrictInt = Field(
        ge=0,
        le=MAX_CANONICAL_CLOSURE_WORK,
    )

    @model_validator(mode="after")
    def bind_total_logical_work(self) -> Self:
        if self.total_logical_work != (
            self.canonical_implication_checks + self.canonical_membership_checks
        ):
            raise PydanticCustomError(
                "formal_concept_analysis.logical_work_mismatch",
                "total_logical_work must equal implication plus membership checks",
            )
        return self


def _require_canonical_carrier_subset(
    name: str,
    values: tuple[int, ...],
    carrier_size: int,
) -> None:
    if values != tuple(sorted(set(values))):
        raise PydanticCustomError(
            "formal_concept_analysis.subset_not_canonical",
            f"{name} must be sorted and duplicate-free",
            {"name": name},
        )
    if any(attribute >= carrier_size for attribute in values):
        raise PydanticCustomError(
            "formal_concept_analysis.subset_attribute_out_of_range",
            f"{name} contains an attribute outside the carrier",
            {"name": name},
        )


class ImplicationClosureResult(StrictModel):
    """The least source-bound closure of a seed under finite implications."""

    system: FiniteAttributeImplicationSystem
    seed: tuple[_AttributeIndex, ...]
    closure: tuple[_AttributeIndex, ...]
    added: tuple[_AttributeIndex, ...]
    lineage: tuple[ImplicationDerivation, ...]
    work: ImplicationClosureWork

    @classmethod
    def _from_kernel(
        cls,
        *,
        system: FiniteAttributeImplicationSystem,
        seed: tuple[int, ...],
        closure: tuple[int, ...],
        added: tuple[int, ...],
        lineage: tuple[ImplicationDerivation, ...],
        work: ImplicationClosureWork,
    ) -> Self:
        """Build a result emitted by the owner-local closure kernel."""

        return cls(
            system=system,
            seed=seed,
            closure=closure,
            added=added,
            lineage=lineage,
            work=work,
        )


class FormalContext(StrictModel):
    """An immutable finite formal context K = (G, M, I).

    ``objects`` is a tuple of unique object labels.  ``attributes`` is a tuple
    of unique attribute labels.  ``incidence`` is a tuple of ``(object_index,
    attribute_index)`` pairs, each denoting that object ``objects[oi]`` has
    attribute ``attributes[ai]``.
    """

    objects: tuple[str, ...] = Field(min_length=1, max_length=MAX_OBJECTS)
    attributes: tuple[str, ...] = Field(min_length=1, max_length=MAX_ATTRIBUTES)
    incidence: tuple[tuple[int, int], ...] = Field(default=())

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(set(self.objects)) != len(self.objects):
            raise PydanticCustomError(
                "formal_concept_analysis.object_labels_not_unique",
                "object labels must be unique",
            )
        if len(set(self.attributes)) != len(self.attributes):
            raise PydanticCustomError(
                "formal_concept_analysis.attribute_labels_not_unique",
                "attribute labels must be unique",
            )
        seen: set[tuple[int, int]] = set()
        for oi, ai in self.incidence:
            if not 0 <= oi < len(self.objects):
                raise PydanticCustomError(
                    "formal_concept_analysis.incidence_object_out_of_range",
                    "incidence object index out of range",
                )
            if not 0 <= ai < len(self.attributes):
                raise PydanticCustomError(
                    "formal_concept_analysis.incidence_attribute_out_of_range",
                    "incidence attribute index out of range",
                )
            pair = (oi, ai)
            if pair in seen:
                raise PydanticCustomError(
                    "formal_concept_analysis.incidence_pairs_not_unique",
                    "incidence pairs must be duplicate-free",
                )
            seen.add(pair)
        return self


__all__ = [
    "MAX_ATTRIBUTES",
    "MAX_CANONICAL_CLOSURE_WORK",
    "MAX_FORWARD_CHAIN_WORK",
    "MAX_IMPLICATIONS",
    "MAX_IMPLICATION_CLOSURE_RESULT_BYTES",
    "MAX_IMPLICATION_MEMBERSHIPS",
    "MAX_OBJECTS",
    "AttributeImplication",
    "FiniteAttributeImplicationSystem",
    "FormalContext",
    "ImplicationClosureResult",
    "ImplicationClosureWork",
    "ImplicationDerivation",
]
