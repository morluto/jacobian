"""Typed input and output values for finite CSP domain consistency."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.relational_structures._models import FiniteCspInstance
from jacobian.math.logic.relational_structures.values import MAX_RELATIONAL_CARRIER


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.csp.consistency.{code}", message)


class CspDomainRequest(StrictModel):
    """A finite CSP instance with one allowed template subset per variable."""

    instance: FiniteCspInstance
    domains: tuple[Annotated[tuple[StrictInt, ...], Field(max_length=MAX_RELATIONAL_CARRIER)], ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description="One ordered, duplicate-free subset of template labels per variable.",
    )

    @model_validator(mode="after")
    def require_domain_axes(self) -> Self:
        carrier_size = self.instance.template.carrier_size
        if len(self.domains) != self.instance.variable_count:
            raise _error("domain_axis", "domains must have one entry per CSP variable")
        if any(
            len(domain) > carrier_size
            or len(set(domain)) != len(domain)
            or any(not 0 <= value < carrier_size for value in domain)
            for domain in self.domains
        ):
            raise _error(
                "domain_values",
                "each domain must contain distinct labels from the template carrier",
            )
        return self


class CspDomainConsistency(StrictModel):
    """The greatest generalized-arc-consistent subdomains of the input domains.

    Empty domain indices certify inconsistency of the current domains. A
    nonempty result says only that this local consistency test reached a fixed
    point; it does not assert that a global CSP solution exists.
    """

    instance: FiniteCspInstance
    initial_domains: tuple[tuple[StrictInt, ...], ...]
    domains: tuple[tuple[StrictInt, ...], ...]
    empty_domain_variables: tuple[StrictInt, ...]
    false_nullary_constraint_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_consistency(self) -> Self:
        count = self.instance.variable_count
        if len(self.initial_domains) != count or len(self.domains) != count:
            raise _error("result_axis", "domain axes must match the instance variable count")
        carrier_size = self.instance.template.carrier_size
        if any(
            len(initial) > carrier_size
            or tuple(sorted(set(initial))) != initial
            or any(not 0 <= value < carrier_size for value in initial)
            or tuple(sorted(set(domain))) != domain
            or not set(domain).issubset(initial)
            or any(not 0 <= value < carrier_size for value in domain)
            for domain, initial in zip(self.domains, self.initial_domains, strict=True)
        ):
            raise _error("result_domains", "final domains must be canonical subsets of initial domains")
        expected_empty = tuple(i for i, domain in enumerate(self.domains) if not domain)
        if self.empty_domain_variables != expected_empty:
            raise _error("result_empty_domains", "empty-domain metadata must match the final domains")
        symbols = {symbol.symbol_id: i for i, symbol in enumerate(self.instance.template.signature)}
        false_ids = {
            constraint.constraint_id
            for constraint in self.instance.constraints
            if not constraint.scope
            and not self.instance.template.relation_tables[symbols[constraint.symbol_id]]
        }
        if (
            tuple(sorted(set(self.false_nullary_constraint_ids))) != self.false_nullary_constraint_ids
            or set(self.false_nullary_constraint_ids) != false_ids
        ):
            raise _error("result_nullary_metadata", "false nullary metadata must name exactly false nullary constraints")
        return self
