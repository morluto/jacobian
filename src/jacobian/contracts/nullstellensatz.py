"""Bounded exact Nullstellensatz certificates for normalized Jacobian slices."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.results import ContractModel

NULLSTELLENSATZ_VARIABLE_LIMIT = 16
NULLSTELLENSATZ_EXPONENT_LIMIT = 32
NULLSTELLENSATZ_TERMS_PER_POLYNOMIAL_LIMIT = 1024
NULLSTELLENSATZ_TERMS_PER_CHART_LIMIT = 4096
NULLSTELLENSATZ_TERMS_PER_BUNDLE_LIMIT = 16384
NULLSTELLENSATZ_COEFFICIENT_DIGIT_LIMIT = 256


class BoundedRationalPolynomialTerm(ContractModel):
    """One nonzero bounded QQ monomial in an explicitly ordered ring."""

    coefficient: CanonicalRational
    exponents: tuple[int, ...] = Field(
        min_length=1,
        max_length=NULLSTELLENSATZ_VARIABLE_LIMIT,
    )

    @model_validator(mode="after")
    def require_bounded_nonzero_term(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero polynomial terms must be omitted")
        if (
            any(
                exponent < 0 or exponent > NULLSTELLENSATZ_EXPONENT_LIMIT
                for exponent in self.exponents
            )
            or sum(self.exponents) > NULLSTELLENSATZ_EXPONENT_LIMIT
        ):
            raise ValueError("certificate polynomial exponent limit exceeded")
        if (
            len(self.coefficient.num.lstrip("-"))
            > NULLSTELLENSATZ_COEFFICIENT_DIGIT_LIMIT
            or len(self.coefficient.den) > NULLSTELLENSATZ_COEFFICIENT_DIGIT_LIMIT
        ):
            raise ValueError("certificate coefficient digit limit exceeded")
        return self


class BoundedRationalPolynomial(ContractModel):
    """Canonical sparse QQ polynomial used by systems and multipliers."""

    terms: tuple[BoundedRationalPolynomialTerm, ...] = Field(
        default=(),
        max_length=NULLSTELLENSATZ_TERMS_PER_POLYNOMIAL_LIMIT,
    )

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise ValueError("polynomial exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise ValueError("polynomial terms must use descending lexicographic order")
        return self


class NamedBoundedRationalPolynomial(ContractModel):
    polynomial_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        min_length=1,
        max_length=64,
    )
    polynomial: BoundedRationalPolynomial


class JacobianDegreeChart(ContractModel):
    chart_id: str = Field(pattern=r"^a(?:20|11|02)-b(?:30|21|12|03)$")
    selected_quadratic_coefficient: Literal["a20", "a11", "a02"]
    selected_cubic_coefficient: Literal["b30", "b21", "b12", "b03"]
    variables: tuple[str, ...] = Field(min_length=1, max_length=16)
    generators: tuple[NamedBoundedRationalPolynomial, ...] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def require_one_bounded_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("chart variables must be unique")
        if any(
            len(term.exponents) != len(self.variables)
            for generator in self.generators
            for term in generator.polynomial.terms
        ):
            raise ValueError("every monomial must match the chart variable order")
        if len({item.polynomial_id for item in self.generators}) != len(
            self.generators
        ):
            raise ValueError("chart generator IDs must be unique")
        return self


class NormalizedJacobianDegreeSliceSystem(ContractModel):
    system_schema_version: Literal["1"] = "1"
    statement_id: Literal["normalized-bivariate-jacobian-degree-2-3"] = (
        "normalized-bivariate-jacobian-degree-2-3"
    )
    coefficient_domain: Literal["QQ"] = "QQ"
    source_characteristic: Literal["0"] = "0"
    component_degrees: tuple[Literal[2, 3], Literal[2, 3]] = (2, 3)
    normalization: Literal["F(0)=0;JF(0)=I;det(JF)=1"] = "F(0)=0;JF(0)=I;det(JF)=1"
    chart_encoding: Literal["rabinowitsch-product-cover"] = "rabinowitsch-product-cover"
    chart_count: Literal[12] = 12
    charts: tuple[JacobianDegreeChart, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def require_complete_chart_cover(self) -> Self:
        if self.component_degrees != (2, 3):
            raise ValueError("the frozen component degrees must be exactly (2,3)")
        expected = {
            f"{a}-{b}"
            for a in ("a20", "a11", "a02")
            for b in ("b30", "b21", "b12", "b03")
        }
        if {chart.chart_id for chart in self.charts} != expected:
            raise ValueError("system must contain the complete 3 by 4 chart cover")
        if len({chart.chart_id for chart in self.charts}) != 12:
            raise ValueError("system chart IDs must be unique")
        return self


class JacobianDegreeSliceMaterializeRequest(ContractModel):
    statement_id: Literal["normalized-bivariate-jacobian-degree-2-3"] = (
        "normalized-bivariate-jacobian-degree-2-3"
    )
    coefficient_domain: Literal["QQ"] = "QQ"


class JacobianDegreeSliceMaterializeOutput(ContractModel):
    system_uri: ArtifactUri
    system_digest: Sha256Digest
    chart_count: Literal[12] = 12
    generator_count_per_chart: Literal[10] = 10
    coefficient_domain: Literal["QQ"] = "QQ"


class NullstellensatzResourceBudget(ContractModel):
    wall_seconds: int = Field(default=120, ge=1, le=300, strict=True)
    maximum_degree: int = Field(default=32, ge=1, le=32, strict=True)
    maximum_terms_per_multiplier: int = Field(default=1024, ge=1, le=1024, strict=True)
    maximum_terms_per_chart: int = Field(default=4096, ge=1, le=4096, strict=True)
    maximum_terms_per_bundle: int = Field(default=16384, ge=1, le=16384, strict=True)
    maximum_coefficient_digits: int = Field(default=256, ge=1, le=256, strict=True)
    maximum_output_bytes: int = Field(
        default=2_000_000, ge=1024, le=2_000_000, strict=True
    )


class NullstellensatzCertificateRequest(ContractModel):
    system_uri: ArtifactUri
    resource_budget: NullstellensatzResourceBudget = Field(
        default_factory=NullstellensatzResourceBudget
    )


class NullstellensatzMultiplier(ContractModel):
    generator_id: str = Field(min_length=1, max_length=64)
    multiplier: BoundedRationalPolynomial


class NullstellensatzChartCertificate(ContractModel):
    chart_id: str = Field(pattern=r"^a(?:20|11|02)-b(?:30|21|12|03)$")
    variable_order: tuple[str, ...] = Field(min_length=1, max_length=16)
    generators: tuple[NamedBoundedRationalPolynomial, ...] = Field(
        min_length=1,
        max_length=16,
    )
    multipliers: tuple[NullstellensatzMultiplier, ...] = Field(
        min_length=1,
        max_length=16,
    )
    identity_rhs: CanonicalRational = CanonicalRational(num="1", den="1")

    @model_validator(mode="after")
    def require_paired_bounded_identity(self) -> Self:
        if len(set(self.variable_order)) != len(self.variable_order):
            raise ValueError("certificate variable order must contain unique names")
        if any(
            len(term.exponents) != len(self.variable_order)
            for generator in self.generators
            for term in generator.polynomial.terms
        ) or any(
            len(term.exponents) != len(self.variable_order)
            for multiplier in self.multipliers
            for term in multiplier.multiplier.terms
        ):
            raise ValueError("certificate monomials must match the variable order")
        generator_ids = tuple(item.polynomial_id for item in self.generators)
        if tuple(item.generator_id for item in self.multipliers) != generator_ids:
            raise ValueError("multipliers must pair with generators in declared order")
        total_terms = sum(len(item.multiplier.terms) for item in self.multipliers)
        if total_terms > NULLSTELLENSATZ_TERMS_PER_CHART_LIMIT:
            raise ValueError("chart multiplier term limit exceeded")
        if self.identity_rhs.as_fraction() != 1:
            raise ValueError("Nullstellensatz identity right-hand side must be one")
        return self


class NullstellensatzCertificateBundle(ContractModel):
    certificate_schema_version: Literal["1"] = "1"
    certificate_format: Literal["polynomial.nullstellensatz.chart-cover"] = (
        "polynomial.nullstellensatz.chart-cover"
    )
    format_version: Literal["1"] = "1"
    coefficient_domain: Literal["QQ"] = "QQ"
    system_uri: ArtifactUri
    system_digest: Sha256Digest
    producer: Literal["singular"] = "singular"
    producer_version: str = Field(min_length=1, max_length=64)
    producer_digest: Sha256Digest
    charts: tuple[NullstellensatzChartCertificate, ...] = Field(
        min_length=12,
        max_length=12,
    )

    @model_validator(mode="after")
    def require_complete_bounded_bundle(self) -> Self:
        chart_ids = tuple(chart.chart_id for chart in self.charts)
        if len(set(chart_ids)) != 12:
            raise ValueError("certificate must contain 12 unique charts")
        if (
            sum(
                len(multiplier.multiplier.terms)
                for chart in self.charts
                for multiplier in chart.multipliers
            )
            > NULLSTELLENSATZ_TERMS_PER_BUNDLE_LIMIT
        ):
            raise ValueError("bundle multiplier term limit exceeded")
        return self


class NullstellensatzCertificateOutput(ContractModel):
    system_uri: ArtifactUri
    certificate_bundle_uri: ArtifactUri
    chart_count: Literal[12] = 12
    conclusion: Literal["INFEASIBLE"] = "INFEASIBLE"
    identity: Literal["sum(h_i*f_i)=1"] = "sum(h_i*f_i)=1"
    producer: Literal["singular"] = "singular"
    producer_version: str = Field(min_length=1, max_length=64)


class NullstellensatzVerificationRequest(ContractModel):
    system_uri: ArtifactUri
    certificate_bundle_uri: ArtifactUri
    timeout_seconds: int = Field(default=30, ge=1, le=105, strict=True)


class NullstellensatzVerificationOutput(ContractModel):
    system_uri: ArtifactUri
    certificate_bundle_uri: ArtifactUri
    evidence_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    claim: Literal["SYSTEM_INFEASIBLE"] = "SYSTEM_INFEASIBLE"
    conclusion: Literal["TRUE", "UNKNOWN"]
    checked_chart_count: int = Field(ge=0, le=12)

    @model_validator(mode="after")
    def bind_verification_evidence(self) -> Self:
        verified = self.conclusion == "TRUE"
        if verified != (
            self.verification_record_uri is not None and self.checker_id is not None
        ):
            raise ValueError("a true conclusion requires a checker-backed record")
        if self.checked_chart_count != (12 if verified else 0):
            raise ValueError("checked chart count must agree with verification")
        return self


__all__ = [
    "BoundedRationalPolynomial",
    "BoundedRationalPolynomialTerm",
    "JacobianDegreeChart",
    "JacobianDegreeSliceMaterializeOutput",
    "JacobianDegreeSliceMaterializeRequest",
    "NamedBoundedRationalPolynomial",
    "NormalizedJacobianDegreeSliceSystem",
    "NullstellensatzCertificateBundle",
    "NullstellensatzCertificateOutput",
    "NullstellensatzCertificateRequest",
    "NullstellensatzChartCertificate",
    "NullstellensatzMultiplier",
    "NullstellensatzResourceBudget",
    "NullstellensatzVerificationOutput",
    "NullstellensatzVerificationRequest",
]
