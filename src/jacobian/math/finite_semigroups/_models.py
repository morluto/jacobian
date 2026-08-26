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

    ``index`` is the smallest positive exponent whose power first repeats;
    ``powers`` is ``a, a^2, a^3, ...`` in one-based exponent order.
    Deserialization checks the bounded result shape only.  Exact replay is
    available through the owner-local verifier for independently supplied
    claims; kernel output uses the trusted factory below.
    """

    semigroup: FiniteSemigroup
    element: OpaqueLabel
    powers: tuple[OpaqueLabel, ...]
    index: int = Field(ge=1)
    period: int = Field(ge=1)
    idempotent: OpaqueLabel
    cyclic_subsemigroup: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def require_bounded_shape(self) -> Self:
        labels = set(self.semigroup.elements)
        if (
            not self.powers
            or len(self.powers) > len(labels)
            or self.element not in labels
            or self.powers[0] != self.element
            or len(set(self.powers)) != len(self.powers)
            or any(power not in labels for power in self.powers)
        ):
            raise _validation_error(
                "power_profile_shape",
                "powers must be a bounded distinct sequence starting at element",
            )
        if self.index + self.period - 1 != len(self.powers):
            raise _validation_error(
                "power_profile_length",
                "powers length must equal index plus period minus one",
            )
        if (
            self.idempotent not in self.powers
            or self.cyclic_subsemigroup != self.powers
        ):
            raise _validation_error(
                "power_profile_values",
                "idempotent and cyclic subsemigroup must use the declared power profile",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PowerProfileRequest,
        powers: tuple[OpaqueLabel, ...],
        index: int,
        period: int,
        idempotent: OpaqueLabel,
        cyclic_subsemigroup: tuple[OpaqueLabel, ...],
    ) -> Self:
        return cls(
            semigroup=request.semigroup,
            element=request.element,
            powers=powers,
            index=index,
            period=period,
            idempotent=idempotent,
            cyclic_subsemigroup=cyclic_subsemigroup,
        )


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
    def require_declared_power(self) -> Self:
        labels = set(self.semigroup.elements)
        if self.element not in labels or self.power not in labels:
            raise _validation_error(
                "power_not_declared",
                "element and power must be declared semigroup elements",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: ElementPowerRequest, power: OpaqueLabel) -> Self:
        return cls(
            semigroup=request.semigroup,
            element=request.element,
            exponent=request.exponent,
            power=power,
        )


class IdempotentsRequest(StrictModel):
    """Request all idempotent elements ``e`` with ``e*e = e``."""

    semigroup: FiniteSemigroup


class IdempotentsResult(StrictModel):
    """All idempotent elements of a finite semigroup."""

    semigroup: FiniteSemigroup
    idempotents: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def require_idempotent_shape(self) -> Self:
        declared = tuple(
            element
            for element in self.semigroup.elements
            if element in self.idempotents
        )
        if self.idempotents != declared:
            raise _validation_error(
                "idempotents_not_canonical",
                "idempotents must be distinct declared elements in semigroup order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: IdempotentsRequest, idempotents: tuple[OpaqueLabel, ...]
    ) -> Self:
        return cls(semigroup=request.semigroup, idempotents=idempotents)


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
    def require_ideal_shape(self) -> Self:
        declared_elements = tuple(
            element for element in self.semigroup.elements if element in self.elements
        )
        if self.elements != declared_elements or len(self.ideals) != len(self.elements):
            raise _validation_error(
                "principal_ideal_shape",
                "elements and ideals must have matching canonical length",
            )
        for element, ideal in zip(self.elements, self.ideals, strict=True):
            declared_ideal = tuple(
                candidate for candidate in self.semigroup.elements if candidate in ideal
            )
            if not ideal or ideal != declared_ideal or element not in ideal:
                raise _validation_error(
                    "principal_ideal_values",
                    "every ideal must be a nonempty canonical declared set containing its element",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrincipalIdealsRequest,
        ideals: tuple[tuple[OpaqueLabel, ...], ...],
    ) -> Self:
        return cls(
            semigroup=request.semigroup, elements=request.elements, ideals=ideals
        )


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
    def require_partition_shapes(self) -> Self:
        for relation in (self.L, self.R, self.H, self.D, self.J):
            flattened = tuple(element for block in relation for element in block)
            if (
                not relation
                or any(not block for block in relation)
                or flattened != self.semigroup.elements
            ):
                raise _validation_error(
                    "green_partition_shape",
                    "every Green relation must partition elements in declared order",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: GreenRelationsRequest,
        L: tuple[tuple[str, ...], ...],  # noqa: N803
        R: tuple[tuple[str, ...], ...],  # noqa: N803
        H: tuple[tuple[str, ...], ...],  # noqa: N803
        D: tuple[tuple[str, ...], ...],  # noqa: N803
        J: tuple[tuple[str, ...], ...],  # noqa: N803
    ) -> Self:
        return cls(semigroup=request.semigroup, L=L, R=R, H=H, D=D, J=J)
