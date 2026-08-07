"""Bounded Singular producer for chart-cover Nullstellensatz certificates."""

from __future__ import annotations

import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityMode,
    CapabilityProviderRuntime,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.nullstellensatz import (
    BoundedRationalPolynomial,
    BoundedRationalPolynomialTerm,
    JacobianDegreeChart,
    NormalizedJacobianDegreeSliceSystem,
    NullstellensatzCertificateBundle,
    NullstellensatzCertificateOutput,
    NullstellensatzCertificateRequest,
    NullstellensatzChartCertificate,
    NullstellensatzMultiplier,
    NullstellensatzResourceBudget,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains.polynomial_nullstellensatz.core import MATERIALIZE_CAPABILITY_ID
from jacobian.domains.polynomial_nullstellensatz.system import (
    materialize_degree_23_system,
)
from jacobian.installation.context import InstallationContext
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResourceLimits,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.schema_registry import model_schema
from jacobian.storage.errors import ArtifactNotFoundError, StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.worker_environment import worker_environment

PRODUCE_CAPABILITY_ID = "polynomial.nullstellensatz.infeasibility_certificate.compute"
_INTEGER_OR_RATIONAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?$")
_STDERR_LIMIT = 64_000


def _diagnostic(code: str, message: str) -> CapabilityDiagnostic:
    return CapabilityDiagnostic(
        code=code,
        stage="nullstellensatz_certificate_production",
        message=message,
        hint="Inspect the bounded Singular provider status and retry with the frozen system artifact.",
    )


def _singular_script() -> bytes:
    chart_rows: list[str] = []
    letter = {
        "a20": "A",
        "a11": "B",
        "a02": "C",
        "b20": "D",
        "b11": "E",
        "b02": "F",
        "b30": "G",
        "b21": "H",
        "b12": "I",
        "b03": "J",
    }
    for quadratic in ("a20", "a11", "a02"):
        for cubic in ("b30", "b21", "b12", "b03"):
            chart_id = f"{quadratic}-{cubic}"
            chart_rows.extend(
                (
                    f'print("JCB_BEGIN|{chart_id}");',
                    (f"ideal generators=base,T*{letter[quadratic]}*{letter[cubic]}-1;"),
                    "matrix certificate=lift(generators,ideal(1));",
                    "for (int k=1; k<=10; k++) {",
                    '  print("JCB_MULT|"+string(k));',
                    "  poly p=certificate[k,1];",
                    "  while (p!=0) {",
                    '    print("JCB_TERM|"+string(leadcoef(p))+"|"+string(leadexp(p)));',
                    "    p=p-lead(p);",
                    "  }",
                    "}",
                    f'print("JCB_END|{chart_id}");',
                )
            )
    script = "\n".join(
        (
            "ring r=0,(A,B,C,D,E,F,G,H,I,J,T),dp;",
            (
                "ideal base=2*A*H-3*B*G,4*A*I-B*H-6*C*G,"
                "6*A*J+B*I-4*C*H,3*B*J-2*C*I,H+2*A*E-2*B*D,"
                "2*I+4*A*F-4*C*D,3*J+2*B*F-2*C*E,E+2*A,2*F+B;"
            ),
            *chart_rows,
            "quit;",
        )
    )
    return script.encode("ascii")


def _parse_coefficient(value: str) -> CanonicalRational:
    if _INTEGER_OR_RATIONAL.fullmatch(value) is None:
        raise ValueError("Singular returned a non-rational coefficient")
    parsed = Fraction(value)
    return CanonicalRational(num=str(parsed.numerator), den=str(parsed.denominator))


class _TaggedOutputParser:
    def __init__(self) -> None:
        self.chart_terms: dict[str, list[list[BoundedRationalPolynomialTerm]]] = {}
        self.chart_multipliers: dict[str, set[int]] = {}
        self.current_chart: str | None = None
        self.current_multiplier: int | None = None

    def feed(self, line: str) -> None:
        if line.startswith("JCB_BEGIN|"):
            self._begin(line.removeprefix("JCB_BEGIN|"))
        elif line.startswith("JCB_MULT|"):
            self._multiplier(line.removeprefix("JCB_MULT|"))
        elif line.startswith("JCB_TERM|"):
            self._term(line)
        elif line.startswith("JCB_END|"):
            self._end(line.removeprefix("JCB_END|"))
        elif line.startswith("JCB_"):
            raise ValueError("unknown Singular output marker")

    def _begin(self, chart_id: str) -> None:
        if self.current_chart is not None or chart_id in self.chart_terms:
            raise ValueError("duplicate or nested chart output")
        self.current_chart = chart_id
        self.current_multiplier = None
        self.chart_terms[chart_id] = [[] for _ in range(10)]
        self.chart_multipliers[chart_id] = set()

    def _multiplier(self, value: str) -> None:
        if self.current_chart is None:
            raise ValueError("multiplier appeared outside a chart")
        multiplier = int(value) - 1
        if not 0 <= multiplier < 10:
            raise ValueError("invalid multiplier index")
        if multiplier in self.chart_multipliers[self.current_chart]:
            raise ValueError("duplicate multiplier marker")
        self.chart_multipliers[self.current_chart].add(multiplier)
        self.current_multiplier = multiplier

    def _term(self, line: str) -> None:
        if self.current_chart is None or self.current_multiplier is None:
            raise ValueError("term appeared outside a multiplier")
        _, coefficient_text, exponent_text = line.split("|", 2)
        exponents = tuple(
            int(value) for value in re.findall(r"-?[0-9]+", exponent_text)
        )
        if len(exponents) != 11:
            raise ValueError("Singular exponent vector has the wrong dimension")
        self.chart_terms[self.current_chart][self.current_multiplier].append(
            BoundedRationalPolynomialTerm(
                coefficient=_parse_coefficient(coefficient_text),
                exponents=exponents,
            )
        )

    def _end(self, chart_id: str) -> None:
        if chart_id != self.current_chart:
            raise ValueError("chart end marker mismatch")
        if self.chart_multipliers[chart_id] != set(range(10)):
            raise ValueError("chart does not contain all multiplier markers")
        self.current_chart = None
        self.current_multiplier = None

    def finish(self) -> dict[str, list[list[BoundedRationalPolynomialTerm]]]:
        if self.current_chart is not None:
            raise ValueError("unterminated chart output")
        return self.chart_terms


def _chart_certificate(
    chart: JacobianDegreeChart,
    raw_multipliers: list[list[BoundedRationalPolynomialTerm]],
) -> NullstellensatzChartCertificate:
    multipliers = tuple(
        NullstellensatzMultiplier(
            generator_id=generator.polynomial_id,
            multiplier=BoundedRationalPolynomial(
                terms=tuple(
                    sorted(terms, key=lambda item: item.exponents, reverse=True)
                )
            ),
        )
        for generator, terms in zip(
            chart.generators,
            raw_multipliers,
            strict=True,
        )
    )
    return NullstellensatzChartCertificate(
        chart_id=chart.chart_id,
        variable_order=chart.variables,
        generators=chart.generators,
        multipliers=multipliers,
    )


def _parse_output(
    stdout: bytes,
    system: NormalizedJacobianDegreeSliceSystem,
) -> tuple[NullstellensatzChartCertificate, ...]:
    parser = _TaggedOutputParser()
    for line in stdout.decode("ascii").splitlines():
        parser.feed(line.strip())
    chart_terms = parser.finish()
    system_by_id = {chart.chart_id: chart for chart in system.charts}
    if set(chart_terms) != set(system_by_id):
        raise ValueError("Singular output does not cover every chart")
    return tuple(
        _chart_certificate(chart, chart_terms[chart.chart_id])
        for chart in system.charts
    )


def _within_declared_budget(
    charts: tuple[NullstellensatzChartCertificate, ...],
    budget: NullstellensatzResourceBudget,
) -> bool:
    bundle_terms = 0
    for chart in charts:
        chart_terms = 0
        for multiplier in chart.multipliers:
            terms = multiplier.multiplier.terms
            if len(terms) > budget.maximum_terms_per_multiplier:
                return False
            if any(
                sum(term.exponents) > budget.maximum_degree
                or len(term.coefficient.num.lstrip("-"))
                > budget.maximum_coefficient_digits
                or len(term.coefficient.den) > budget.maximum_coefficient_digits
                for term in terms
            ):
                return False
            chart_terms += len(terms)
        if chart_terms > budget.maximum_terms_per_chart:
            return False
        bundle_terms += chart_terms
    return bundle_terms <= budget.maximum_terms_per_bundle


def _process_failure(
    completed: ProcessResult,
) -> tuple[ExecutionStatus, CapabilityDiagnostic] | None:
    if completed.termination is ProcessTermination.TIMED_OUT:
        return ExecutionStatus.TIMEOUT, _diagnostic(
            "SINGULAR_TIMEOUT",
            "Singular did not complete within the declared wall budget.",
        )
    if completed.termination is ProcessTermination.CANCELLED:
        return ExecutionStatus.CANCELLED, _diagnostic(
            "SINGULAR_CANCELLED",
            "Singular was cancelled before a complete certificate was available.",
        )
    if completed.termination is ProcessTermination.OUTPUT_LIMIT_EXCEEDED:
        return ExecutionStatus.ERROR, _diagnostic(
            "SINGULAR_OUTPUT_LIMIT_EXCEEDED",
            "Singular exceeded the bounded output protocol.",
        )
    if completed.termination is ProcessTermination.START_FAILED:
        return ExecutionStatus.ERROR, _diagnostic(
            "SINGULAR_START_FAILED",
            "The bounded Singular process could not start.",
        )
    if completed.returncode != 0:
        return ExecutionStatus.ERROR, _diagnostic(
            "SINGULAR_COMPUTATION_FAILED",
            "Singular did not produce a complete certificate bundle.",
        )
    return None


def _certificate_payload(
    charts: tuple[NullstellensatzChartCertificate, ...],
    system_artifact: StoredArtifact,
    runtime: CapabilityProviderRuntime,
    budget: NullstellensatzResourceBudget,
) -> dict[str, Any]:
    if not _within_declared_budget(charts, budget):
        raise ValueError("certificate exceeds the declared algebraic budget")
    if runtime.digest is None:
        raise ValueError("Singular runtime digest is unavailable")
    bundle = NullstellensatzCertificateBundle(
        system_uri=system_artifact.artifact_uri,
        system_digest=system_artifact.manifest.object_digest,
        producer_version=runtime.version or "unknown",
        producer_digest=runtime.digest,
        charts=charts,
    )
    payload = bundle.model_dump(mode="json")
    if len(canonicalize_json(payload)) > budget.maximum_output_bytes:
        raise ValueError("canonical certificate exceeds the declared byte limit")
    return payload


class SingularNullstellensatzCertificateAdapter:
    def __init__(
        self,
        context: InstallationContext,
        dependency: InstalledDomainBundle,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.context = context
        self.dependency = dependency
        self.provider_runtime = provider_runtime
        self.system_schema_uri = dependency.result_schema_uris[
            MATERIALIZE_CAPABILITY_ID
        ]
        self.bundle_schema_uri = dependency.obligation_schema_uris[
            "nullstellensatz_certificate_bundle"
        ]
        self._descriptor = CapabilityDescriptor(
            capability_id=PRODUCE_CAPABILITY_ID,
            version="1",
            title="Compute a bounded Nullstellensatz infeasibility certificate",
            description=(
                "Use pinned Singular lift computations to produce one exact "
                "sum(h_i*f_i)=1 certificate for each chart in a bound degree slice."
            ),
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(NullstellensatzCertificateRequest),
            output_schema=model_schema(NullstellensatzCertificateOutput),
            tags=(
                "polynomial",
                "nullstellensatz",
                "singular",
                "certificate",
                "bounded",
            ),
            accepted_input_kinds=(CapabilityInputKind.TYPED_ARTIFACT,),
            accepted_artifact_types=(self.system_schema_uri,),
            produced_artifact_types=(self.bundle_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def _failure(
        self,
        request: CapabilityRequest,
        status: ExecutionStatus,
        diagnostic: CapabilityDiagnostic,
        started: float,
    ) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=status,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                detail=diagnostic.message,
            ),
            output={"error": diagnostic.model_dump(mode="json", exclude_none=True)},
            diagnostics=(diagnostic,),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.UNKNOWN,
                basis="producer failure establishes no chart identity coverage",
                assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.HEURISTIC,
                basis="producer did not complete; no infeasibility conclusion",
            ),
        )

    def _resolve_request(
        self,
        request: CapabilityRequest,
    ) -> tuple[
        NullstellensatzCertificateRequest,
        StoredArtifact,
        NormalizedJacobianDegreeSliceSystem,
    ]:
        try:
            validated = NullstellensatzCertificateRequest.model_validate(request.input)
            system_artifact = self.context.store.get(validated.system_uri)
            identity = (
                system_artifact.manifest.schema_uri,
                system_artifact.manifest.semantics_uri,
            )
            if identity != (self.system_schema_uri, self.dependency.semantics_uri):
                raise ValueError("system artifact has the wrong schema or semantics")
            system = NormalizedJacobianDegreeSliceSystem.model_validate(
                system_artifact.payload
            )
            if system != materialize_degree_23_system():
                raise ValueError("system artifact differs from the frozen degree slice")
            return validated, system_artifact, system
        except (
            ValidationError,
            ValueError,
            ArtifactNotFoundError,
            StorageError,
        ) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_NULLSTELLENSATZ_CERTIFICATE_REQUEST",
                    stage="artifact_resolution",
                    message="The request does not name the frozen producer-owned system artifact.",
                    hint="Invoke the materialization capability and pass its system_uri.",
                )
            ) from exc

    @staticmethod
    def _run_singular(
        executable: str,
        request: NullstellensatzCertificateRequest,
    ) -> ProcessResult:
        budget = request.resource_budget
        return execute_process(
            ProcessRequest(
                executable=executable,
                arguments=("-q",),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                timeout_seconds=float(budget.wall_seconds),
                stdin_bytes=_singular_script(),
                stdout_limit_bytes=budget.maximum_output_bytes,
                stderr_limit_bytes=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=budget.wall_seconds + 1,
                    address_space_bytes=2 * 1024 * 1024 * 1024,
                    file_size_bytes=budget.maximum_output_bytes,
                ),
            )
        )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated, system_artifact, system = self._resolve_request(request)
        started = time.monotonic()
        executable = self.provider_runtime.configuration.get("executable")
        if not isinstance(executable, str):
            return self._failure(
                request,
                ExecutionStatus.ERROR,
                _diagnostic(
                    "SINGULAR_RUNTIME_INVALID",
                    "Singular executable identity is unavailable.",
                ),
                started,
            )
        completed = self._run_singular(executable, validated)
        failure = _process_failure(completed)
        if failure is not None:
            status, diagnostic = failure
            return self._failure(request, status, diagnostic, started)
        try:
            charts = _parse_output(completed.stdout, system)
            payload = _certificate_payload(
                charts,
                system_artifact,
                self.provider_runtime,
                validated.resource_budget,
            )
        except (UnicodeDecodeError, ValidationError, ValueError):
            return self._failure(
                request,
                ExecutionStatus.ERROR,
                _diagnostic(
                    "SINGULAR_PROTOCOL_INVALID",
                    "Singular returned malformed or oversized certificate output.",
                ),
                started,
            )
        stored = self.context.artifacts.put(
            schema_uri=self.bundle_schema_uri,
            semantics_uri=self.dependency.semantics_uri,
            payload=payload,
            parents=(system_artifact.artifact_uri,),
            summary="Singular Nullstellensatz certificate bundle for 12 degree charts",
            producer_write=True,
        )
        output = NullstellensatzCertificateOutput(
            system_uri=system_artifact.artifact_uri,
            certificate_bundle_uri=stored.artifact_uri,
            producer_version=self.provider_runtime.version or "unknown",
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="all 12 charts in the bound normalized degree slice",
                parameters={
                    "chart_count": 12,
                    "maximum_degree": validated.resource_budget.maximum_degree,
                },
                artifact_uri=system_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="Singular returned one multiplier for every generator in every chart",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.infeasibility-certificate-for",
                    source_artifact_uris=(stored.artifact_uri,),
                    target_artifact_uris=(system_artifact.artifact_uri,),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="pinned Singular lift output; independent replay not yet invoked",
            ),
            artifact_uris=(system_artifact.artifact_uri, stored.artifact_uri),
        )


def install_singular_producer(
    context: InstallationContext,
    dependency: InstalledDomainBundle,
    provider_runtime: CapabilityProviderRuntime,
) -> InstalledDomainBundle:
    adapter = SingularNullstellensatzCertificateAdapter(
        context,
        dependency,
        provider_runtime,
    )
    return InstalledDomainBundle(
        adapters=(adapter,),
        semantics_uri=dependency.semantics_uri,
        input_schema_uris={
            NullstellensatzCertificateRequest: adapter.system_schema_uri
        },
        result_schema_uris={PRODUCE_CAPABILITY_ID: adapter.bundle_schema_uri},
        obligation_schema_uris={},
    )


__all__ = [
    "PRODUCE_CAPABILITY_ID",
    "SingularNullstellensatzCertificateAdapter",
    "install_singular_producer",
]
