"""Typed wire contracts for finite semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_ELEMENTS = 50


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite semigroups."""

    return PydanticCustomError(f"finite_semigroup.{reason}", message)


class FiniteSemigroup(StrictModel):
    """A finite semigroup: a finite set with an associative binary operation."""

    elements: tuple[OpaqueLabel, ...] = Field(min_length=1, max_length=MAX_ELEMENTS)
    multiplication: tuple[tuple[OpaqueLabel, ...], ...]

    @model_validator(mode="after")
    def require_valid_semigroup(self) -> Self:
        labels = set(self.elements)
        if len(labels) != len(self.elements):
            raise _validation_error(
                "element_labels_not_distinct", "element labels must be distinct"
            )
        if len(self.multiplication) != len(self.elements):
            raise _validation_error(
                "multiplication_row_count",
                "multiplication table must have one row per element",
            )
        for row in self.multiplication:
            if len(row) != len(self.elements):
                raise _validation_error(
                    "multiplication_not_square", "multiplication table must be square"
                )
            for cell in row:
                if cell not in labels:
                    raise _validation_error(
                        "product_not_declared",
                        "every product must be a declared element",
                    )
        self._check_associativity(labels)
        return self

    def _check_associativity(self, labels: set[str]) -> None:
        idx = {label: i for i, label in enumerate(self.elements)}
        n = len(self.elements)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ij = self.multiplication[i][j]
                    jk = self.multiplication[j][k]
                    left = self.multiplication[idx[ij]][k]
                    right = self.multiplication[i][idx[jk]]
                    if left != right:
                        raise _validation_error(
                            "not_associative",
                            f"semigroup must be associative: "
                            f"({self.elements[i]}*{self.elements[j]})*{self.elements[k]} "
                            f"!= {self.elements[i]}*({self.elements[j]}*{self.elements[k]})",
                        )


class PowerProfileRequest(StrictModel):
    """Request the power profile of one element in a finite semigroup."""

    semigroup: FiniteSemigroup
    element: OpaqueLabel

    @model_validator(mode="after")
    def require_element_exists(self) -> Self:
        if self.element not in set(self.semigroup.elements):
            raise _validation_error(
                "element_not_in_semigroup", "element must be in the semigroup"
            )
        return self


class PowerProfileResult(StrictModel):
    """The power profile of one element in a finite semigroup.

    The supplied semigroup and element are carried on the result so the
    bound model can re-run the exact native kernel and verify the power
    sequence, index, period, idempotent, and cyclic subsemigroup.  ``index``
    is the smallest positive exponent whose power first repeats; ``powers``
    is ``a, a^2, a^3, ...`` in one-based exponent order.
    """

    semigroup: FiniteSemigroup
    element: OpaqueLabel
    powers: tuple[OpaqueLabel, ...]
    index: int = Field(ge=1)
    period: int = Field(ge=1)
    idempotent: OpaqueLabel
    cyclic_subsemigroup: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def bind_power_profile(self) -> Self:
        from jacobian.math.finite_semigroups._operations import _power_profile_data

        powers, index, period, idempotent, cyclic = _power_profile_data(
            self.semigroup.elements,
            self.semigroup.multiplication,
            self.element,
        )
        if self.powers != powers:
            raise _validation_error(
                "powers_mismatch",
                "powers must be the exact power sequence of the element",
            )
        if self.index != index:
            raise _validation_error(
                "index_mismatch", "index must be the first repeated power exponent"
            )
        if self.period != period:
            raise _validation_error(
                "period_mismatch", "period must be the power cycle length"
            )
        if self.idempotent != idempotent:
            raise _validation_error(
                "idempotent_mismatch", "idempotent must be the unique idempotent power"
            )
        if self.cyclic_subsemigroup != cyclic:
            raise _validation_error(
                "cyclic_subsemigroup_mismatch",
                "cyclic_subsemigroup must be the exact closure of the element",
            )
        return self


class GeneratedSubsemigroupRequest(StrictModel):
    """Request the subsemigroup generated by a set of elements."""

    semigroup: FiniteSemigroup
    generators: tuple[OpaqueLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_generators_exist(self) -> Self:
        labels = set(self.semigroup.elements)
        for gen in self.generators:
            if gen not in labels:
                raise _validation_error(
                    "generator_not_in_semigroup",
                    "every generator must be in the semigroup",
                )
        return self


class GeneratedSubsemigroupResult(StrictModel):
    """The subsemigroup generated by a set of elements."""

    generators: tuple[OpaqueLabel, ...]
    elements: tuple[OpaqueLabel, ...]


class ElementPowerRequest(StrictModel):
    """Request the exact power ``element^exponent`` in a finite semigroup."""

    semigroup: FiniteSemigroup
    element: OpaqueLabel
    exponent: int = Field(ge=1)

    @model_validator(mode="after")
    def require_element_exists(self) -> Self:
        if self.element not in set(self.semigroup.elements):
            raise _validation_error(
                "element_not_in_semigroup", "element must be in the semigroup"
            )
        return self


class ElementPowerResult(StrictModel):
    """The exact power ``element^exponent`` in a finite semigroup."""

    semigroup: FiniteSemigroup
    element: OpaqueLabel
    exponent: int = Field(ge=1)
    power: OpaqueLabel

    @model_validator(mode="after")
    def bind_power(self) -> Self:
        from jacobian.math.finite_semigroups._operations import _element_power

        power = _element_power(
            self.semigroup.elements,
            self.semigroup.multiplication,
            self.element,
            self.exponent,
        )
        if self.power != power:
            raise _validation_error(
                "power_mismatch",
                "power must be the exact iterated product of the element",
            )
        return self


class IdempotentsRequest(StrictModel):
    """Request all idempotent elements ``e`` with ``e*e = e``."""

    semigroup: FiniteSemigroup


class IdempotentsResult(StrictModel):
    """All idempotent elements of a finite semigroup."""

    semigroup: FiniteSemigroup
    idempotents: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def bind_idempotents(self) -> Self:
        from jacobian.math.finite_semigroups._operations import _idempotents

        idempotents = _idempotents(
            self.semigroup.elements, self.semigroup.multiplication
        )
        if self.idempotents != idempotents:
            raise _validation_error(
                "idempotents_mismatch",
                "idempotents must be exactly the elements with e*e = e",
            )
        return self


class PrincipalIdealsRequest(StrictModel):
    """Request the principal ideal of each listed element."""

    semigroup: FiniteSemigroup
    elements: tuple[OpaqueLabel, ...] = Field(min_length=1, max_length=MAX_ELEMENTS)

    @model_validator(mode="after")
    def require_elements_exist(self) -> Self:
        labels = set(self.semigroup.elements)
        for element in self.elements:
            if element not in labels:
                raise _validation_error(
                    "element_not_in_semigroup", "every element must be in the semigroup"
                )
        if len(set(self.elements)) != len(self.elements):
            raise _validation_error(
                "requested_elements_not_distinct", "requested elements must be distinct"
            )
        declared_order = tuple(
            element for element in self.semigroup.elements if element in self.elements
        )
        if self.elements != declared_order:
            raise _validation_error(
                "requested_elements_wrong_order",
                "requested elements must use declared semigroup order",
            )
        return self


class PrincipalIdealsResult(StrictModel):
    """The principal ideals of the requested elements.

    The principal two-sided ideal of ``a`` is ``S^1 a S^1``.
    """

    semigroup: FiniteSemigroup
    elements: tuple[OpaqueLabel, ...]
    ideals: tuple[tuple[OpaqueLabel, ...], ...]

    @model_validator(mode="after")
    def bind_ideals(self) -> Self:
        from jacobian.math.finite_semigroups._operations import _principal_ideals

        ideals = _principal_ideals(
            self.semigroup.elements,
            self.semigroup.multiplication,
            self.elements,
        )
        if self.ideals != ideals:
            raise _validation_error(
                "ideals_mismatch",
                "ideals must be the exact principal ideals of the elements",
            )
        return self


class GreenRelationsRequest(StrictModel):
    """Request the Green relations L, R, H, D, J of a finite semigroup."""

    semigroup: FiniteSemigroup


class GreenRelationsResult(StrictModel):
    """Green relations of a finite semigroup.

    Each relation is a tuple of equivalence-class tuples, in declared
    element order, partitioning the semigroup elements.  ``L`` and ``R``
    are Green's left and right equivalences; ``H = L ∩ R``;
    ``D = L ∘ R`` (the join); ``J`` is the two-sided Green relation.
    """

    semigroup: FiniteSemigroup
    L: tuple[tuple[str, ...], ...]
    R: tuple[tuple[str, ...], ...]
    H: tuple[tuple[str, ...], ...]
    D: tuple[tuple[str, ...], ...]
    J: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def bind_green_relations(self) -> Self:
        from jacobian.math.finite_semigroups._operations import _green_relations

        L, R, H, D, J = _green_relations(  # noqa: N806
            self.semigroup.elements, self.semigroup.multiplication
        )
        if self.L != L:
            raise _validation_error(
                "green_l_mismatch", "L must be the exact Green L-relation"
            )
        if self.R != R:
            raise _validation_error(
                "green_r_mismatch", "R must be the exact Green R-relation"
            )
        if self.H != H:
            raise _validation_error(
                "green_h_mismatch", "H must be the exact Green H-relation"
            )
        if self.D != D:
            raise _validation_error(
                "green_d_mismatch", "D must be the exact Green D-relation"
            )
        if self.J != J:
            raise _validation_error(
                "green_j_mismatch", "J must be the exact Green J-relation"
            )
        return self
