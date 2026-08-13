"""Behavioral tests for the explicit typed portfolio plan."""

from __future__ import annotations

import pytest

from jacobian.domain_bundles import DomainBundle
from jacobian.portfolio.builtin import (
    build_builtin_portfolio,
    build_builtin_portfolio_components,
)
from jacobian.portfolio.model import PortfolioPlan


def test_builtin_portfolio_is_an_explicit_plan_of_components() -> None:
    plan = build_builtin_portfolio()

    assert isinstance(plan, PortfolioPlan)
    # The plan is a literal ordered tuple, not discovered or registered.
    assert plan.components == build_builtin_portfolio_components()
    assert plan.domain_ids == tuple(
        bundle.domain_id for bundle in build_builtin_portfolio_components()
    )


def test_validate_accepts_the_builtin_plan() -> None:
    # Must not raise.
    build_builtin_portfolio().validate()
    assert build_builtin_portfolio().domain_ids


def test_bundle_for_returns_the_declared_bundle_or_none() -> None:
    plan = build_builtin_portfolio()
    arithmetic = plan.component_for("arithmetic")
    assert isinstance(arithmetic, DomainBundle)
    assert arithmetic.domain_id == "arithmetic"
    assert plan.component_for("absent.domain") is None


def test_validate_rejects_duplicate_domain_bundles() -> None:
    bundle = build_builtin_portfolio_components()[0]
    plan = PortfolioPlan(components=(bundle, bundle))

    with pytest.raises(ValueError, match="duplicate domain bundles"):
        plan.validate()


def test_validate_rejects_non_domain_bundle_entries() -> None:
    impostor = object()
    plan = PortfolioPlan(components=(impostor,))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="domain bundles"):
        plan.validate()


def test_empty_plan_validates_and_exposes_no_domains() -> None:
    plan = PortfolioPlan(components=())

    plan.validate()
    assert plan.components == ()
    assert plan.domain_ids == ()
    assert plan.component_for("anything") is None
