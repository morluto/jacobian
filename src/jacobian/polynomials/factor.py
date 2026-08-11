"""Adapter implementations for sparse rational polynomial-map capabilities."""

from __future__ import annotations

import time
from typing import Any, cast

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.polynomials import (
    PolynomialFactorizationArtifact,
    PolynomialFactorOutput,
    PolynomialFactorRecord,
    PolynomialFactorRequest,
)
from jacobian.domains._examples import example
from jacobian.polynomials._support import (
    _computed_result,
    _sympy_polynomial,
    _validate_request,
    _wire_polynomial,
    _wire_rational,
)
from jacobian.polynomials._sympy import _sympy
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.schema_registry import model_schema


class PolynomialFactorAdapter:
    """Factor one univariate sparse polynomial over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.factor.compute",
            version="1",
            title="Factor a univariate rational polynomial",
            description=(
                "Compute a coefficient and multiplicity-bearing factor list over QQ, "
                "together with an exact reconstructed product."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("univariate-polynomial-factorization",),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialFactorRequest),
            output_schema=model_schema(PolynomialFactorOutput),
            tags=("polynomial", "factorization", "exact-computation"),
            invocation_examples=(
                example(
                    "factor_x_squared_minus_one",
                    "Factor x^2-1 over QQ.",
                    {
                        "variable": "x",
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialFactorRequest,
            request.input,
            code="INVALID_POLYNOMIAL_FACTOR_REQUEST",
            operation="factorization",
        )
        started = time.monotonic()
        source = self.resources.artifacts.put(
            schema_uri=self.resources.installation.polynomial_schema_uri,
            semantics_uri=self.resources.installation.polynomial_semantics_uri,
            payload=validated.polynomial.model_dump(mode="json"),
            summary="univariate sparse rational polynomial",
        )
        sp = _sympy.get()
        generator = cast(tuple[Any, ...], sp.symbols(validated.variable, seq=True))
        polynomial = _sympy_polynomial(validated.polynomial, generator)
        coefficient_value, raw_factors = polynomial.factor_list()
        factors = tuple(
            PolynomialFactorRecord(
                factor=_wire_polynomial(factor),
                multiplicity=multiplicity,
            )
            for factor, multiplicity in raw_factors
        )
        reconstructed_expression = sp.Rational(coefficient_value)
        for factor, multiplicity in raw_factors:
            reconstructed_expression *= factor.as_expr() ** multiplicity
        reconstructed = _wire_polynomial(
            sp.Poly(
                sp.expand(reconstructed_expression),
                *generator,
                domain=sp.QQ,
            )
        )
        artifact_payload = PolynomialFactorizationArtifact(
            variable=validated.variable,
            source_polynomial_uri=source.artifact_uri,
            coefficient=_wire_rational(coefficient_value),
            factors=factors,
            reconstructed=reconstructed,
            backend_version=SYMPY_VERSION,
        )
        factorization = self.resources.artifacts.put(
            schema_uri=self.resources.installation.factorization_schema_uri,
            semantics_uri=self.resources.installation.factorization_semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            parents=(source.artifact_uri,),
            summary="computed univariate rational polynomial factorization",
        )
        output = PolynomialFactorOutput(
            source_polynomial_uri=source.artifact_uri,
            factorization_uri=factorization.artifact_uri,
            variable=validated.variable,
            coefficient=artifact_payload.coefficient,
            factors=factors,
            reconstructed=reconstructed,
            backend_version=SYMPY_VERSION,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one univariate polynomial over QQ",
                parameters={"variable": validated.variable},
                artifact_uri=source.artifact_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.factorization-of",
                    source_artifact_uris=(source.artifact_uri,),
                    target_artifact_uris=(factorization.artifact_uri,),
                ),
            ),
            artifact_uris=(source.artifact_uri, factorization.artifact_uri),
            completeness_basis=(
                "SymPy returned a factor list and its product was reconstructed exactly"
            ),
            assurance_basis=(
                "exact SymPy factorization and product reconstruction over QQ; "
                "factor irreducibility was not independently verified"
            ),
        )
