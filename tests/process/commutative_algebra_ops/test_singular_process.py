"""Failure-mode tests for the one-shot Singular process boundary."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from fractions import Fraction
from pathlib import Path

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.commutative_algebra_ops._models import IdealComputationBudget
from jacobian.math.commutative_algebra_ops._singular import (
    _minimal_primes_stdout_limit,
    run_singular_ideal_operation,
    run_singular_minimal_primes,
    run_singular_minimal_primes_verification,
    run_singular_saturation_verification,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import (
    BoundedProcessResult,
    ProcessResourceLimits,
    bounded_process_cancellation,
)


def _ideal() -> RationalPolynomialIdeal:
    variables = ("x",)
    return RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(2,),
                        ),
                    )
                ),
            ),
        ),
    )


def _eight_var_ideal() -> RationalPolynomialIdeal:
    """A source in eight variables exercising the aggregate decode envelopes."""

    variables = tuple(f"v{index}" for index in range(8))
    return RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=tuple(2 for _ in variables),
                        ),
                    )
                ),
            ),
        ),
    )


def _executable(tmp_path: Path, body: str) -> str:
    path = tmp_path / "fake-singular"
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return os.fspath(path)


def _select_executable(monkeypatch: pytest.MonkeyPatch, executable: str) -> None:
    monkeypatch.setattr(
        "jacobian.math._singular.shutil.which",
        lambda name: executable if name == "Singular" else None,
    )


def test_timeout_is_not_reported_as_a_mathematical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(wall_seconds=1),
    )
    assert result.outcome == "TIMEOUT"
    assert result.ideal is None


def test_cancellation_is_preserved_as_its_own_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        result = run_singular_ideal_operation(
            "radical",
            _ideal(),
            None,
            IdealComputationBudget(),
        )

    assert result.outcome == "CANCELLED"
    assert result.ideal is None


def test_minimal_prime_producing_cancellation_keeps_its_typed_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "CANCELLED"
    assert result.components is None
    assert result.detail == (
        "Singular execution was cancelled before producing a result."
    )


def test_minimal_prime_verification_cancellation_is_a_typed_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The independent verification pass preserves cancellation too."""

    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        verdict = run_singular_minimal_primes_verification(
            _ideal(), (_ideal(),), IdealComputationBudget()
        )

    assert verdict == "CANCELLED"


def test_saturation_verification_cancellation_is_a_typed_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    variables = ("x", "y")
    source = _monomial_ideal(variables, (1, 1))
    saturator = RationalPolynomialIdeal(
        variables=variables,
        generators=(_single_term_poly(variables, (1, 0)),),
    )
    claimed = _monomial_ideal(variables, (1, 0))
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        verdict = run_singular_saturation_verification(
            source, saturator, claimed, IdealComputationBudget()
        )

    assert verdict == "CANCELLED"


def test_missing_backend_is_a_typed_unavailable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math._singular.shutil.which",
        lambda name: None,
    )

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "UNAVAILABLE"
    assert result.ideal is None


def test_minimal_prime_family_protocol_is_typed_and_canonically_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        "2",
        "COMPONENT",
        "1",
        "GENERATOR",
        "1|1",
        "END_GENERATOR",
        "END_COMPONENT",
        "COMPONENT",
        "1",
        "GENERATOR",
        "1|2",
        "END_GENERATOR",
        "END_COMPONENT",
        "END",
    )
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert tuple(
        component.model_dump_json() for component in result.components
    ) == tuple(sorted(component.model_dump_json() for component in result.components))


def test_narrowed_replay_allowance_reaches_the_bounded_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def bounded_process(
        command: object,
        *,
        timeout_seconds: float,
        resource_limits: ProcessResourceLimits | None = None,
        **kwargs: object,
    ) -> BoundedProcessResult:
        seen["timeout_seconds"] = timeout_seconds
        seen["cpu_seconds"] = (
            resource_limits.cpu_seconds if resource_limits is not None else None
        )
        return BoundedProcessResult(
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
            cancelled=False,
        )

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        bounded_process,
    )
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: "/usr/bin/Singular" if name == "Singular" else None,
    )

    result = run_singular_minimal_primes(
        _ideal(), IdealComputationBudget(wall_seconds=60), wall_seconds=6.75
    )

    assert result.outcome == "TIMEOUT"
    assert seen["timeout_seconds"] == 6.75
    assert seen["cpu_seconds"] == 7


def _family_records(component_generator_counts: list[int], exponents: str = "0") -> str:
    lines = [
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        str(len(component_generator_counts)),
    ]
    for count in component_generator_counts:
        lines.append("COMPONENT")
        lines.append(str(count))
        for _ in range(count):
            lines.extend(("GENERATOR", f"1|{exponents}", "END_GENERATOR"))
        lines.append("END_COMPONENT")
    lines.append("END")
    return "\n".join(lines)


_EIGHT_EXPONENTS = ",".join("0" for _ in range(8))


def test_aggregate_generator_limit_is_enforced_across_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path, f"print({_family_records([8] * 9, _EIGHT_EXPONENTS)!r})"
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_eight_var_ideal(), IdealComputationBudget())

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.components is None


def test_family_at_the_aggregate_generator_limit_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path, f"print({_family_records([8] * 8, _EIGHT_EXPONENTS)!r})"
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_eight_var_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert all(len(component.generators) == 8 for component in result.components)


def test_family_at_the_aggregate_generator_limit_with_placeholders_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path,
        f"print({_family_records([0, 8, 8, 8, 8, 8, 8, 8, 7], _EIGHT_EXPONENTS)!r})",
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_eight_var_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert sorted(len(component.generators) for component in result.components) == [
        1,
        7,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ]


def test_zero_placeholder_padding_cannot_exceed_the_generator_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path, f"print({_family_records([8] * 8 + [0], _EIGHT_EXPONENTS)!r})"
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_eight_var_ideal(), IdealComputationBudget())

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.components is None


def test_component_wider_than_the_ring_dimension_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, f"print({_family_records([3], '0,0')!r})")
    _select_executable(monkeypatch, executable)

    variables = ("x", "y")
    source = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(1, 1),
                        ),
                    )
                ),
            ),
        ),
    )

    result = run_singular_minimal_primes(source, IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert [len(component.generators) for component in result.components] == [3]


def test_component_at_the_ring_dimension_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, f"print({_family_records([2, 2], '0,0')!r})")
    _select_executable(monkeypatch, executable)

    variables = ("x", "y")
    source = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(1, 1),
                        ),
                    )
                ),
            ),
        ),
    )

    result = run_singular_minimal_primes(source, IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert [len(component.generators) for component in result.components] == [2, 2]


def test_exhausted_replay_allowance_times_out_without_launching_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(command: object, **kwargs: object) -> BoundedProcessResult:
        raise AssertionError("Singular must not launch on an exhausted allowance")

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        forbidden,
    )

    result = run_singular_minimal_primes(
        _ideal(), IdealComputationBudget(), wall_seconds=-0.5
    )

    assert result.outcome == "TIMEOUT"
    assert result.components is None


def test_caller_cannot_narrow_the_exact_result_contract() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1024"):
        IdealComputationBudget(maximum_output_terms=1)


def test_temporary_directory_failure_is_a_typed_unavailable_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not reached")')
    _select_executable(monkeypatch, executable)

    def unavailable_directory(*args: object, **kwargs: object) -> None:
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(
        "jacobian.math._singular.tempfile.TemporaryDirectory",
        unavailable_directory,
    )

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "UNAVAILABLE"


def test_relative_path_backend_is_resolved_before_entering_worker_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(_executable(tmp_path, 'print("not the protocol")'))
    monkeypatch.chdir(tmp_path)
    _select_executable(monkeypatch, executable.name)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "ERROR"
    assert (
        result.detail == "Singular returned an invalid or unsupported result encoding."
    )


def test_relative_prlimit_path_is_resolved_before_entering_worker_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(_executable(tmp_path, 'print("not the protocol")'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "jacobian.math._singular.shutil.which",
        lambda name: executable.name if name in {"Singular", "prlimit"} else None,
    )

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "ERROR"


def test_nonzero_exit_is_a_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "raise SystemExit(7)")
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None
    assert result.detail == "Singular failed without producing an exact ideal."


def test_malformed_success_output_is_a_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None
    assert (
        result.detail == "Singular returned an invalid or unsupported result encoding."
    )


def test_unsupported_backend_version_is_typed_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path,
        "print('\\n'.join(("
        "'JACOBIAN_SINGULAR_IDEAL_V1', '45000', '1', 'GENERATOR', "
        "'1|2', 'END_GENERATOR', 'END'))) ",
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "UNAVAILABLE"
    assert result.ideal is None
    assert result.detail == "The installed Singular release is unsupported."


def test_invocation_is_hermetic_and_version_precedes_algebra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44105",
        "1",
        "GENERATOR",
        "1|2",
        "END_GENERATOR",
        "END",
    )
    body = (
        "import sys\n"
        "required={'-q','-t','--no-rc','--no-shell','--no-stdlib'}\n"
        "if not required.issubset(sys.argv): raise SystemExit(7)\n"
        f"print({chr(10).join(records)!r})"
    )
    executable = _executable(tmp_path, body)
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "COMPUTED"
    source = (
        __import__(
            "jacobian.math.commutative_algebra_ops._singular",
            fromlist=["_script"],
        )
        ._script("radical", _ideal(), None)
        .decode("ascii")
    )
    assert source.index('system("version")') < source.index('LIB "primdec.lib"')


_HERMETIC_ARGUMENTS = ("-q", "-t", "--no-rc", "--no-shell", "--no-stdlib")


def test_minimal_prime_producing_invocation_is_hermetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44105",
        "1",
        "COMPONENT",
        "1",
        "GENERATOR",
        "1|2",
        "END_GENERATOR",
        "END_COMPONENT",
        "END",
    )
    body = (
        "import sys\n"
        f"required={set(_HERMETIC_ARGUMENTS)!r}\n"
        "if not required.issubset(sys.argv): raise SystemExit(7)\n"
        f"print({chr(10).join(records)!r})"
    )
    executable = _executable(tmp_path, body)
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None


def test_minimal_prime_verifying_invocation_is_hermetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44105",
        "VERDICT 1",
        "END",
    )
    body = (
        "import sys\n"
        f"required={set(_HERMETIC_ARGUMENTS)!r}\n"
        "if not required.issubset(sys.argv): raise SystemExit(7)\n"
        f"print({chr(10).join(records)!r})"
    )
    executable = _executable(tmp_path, body)
    _select_executable(monkeypatch, executable)

    verdict = run_singular_minimal_primes_verification(
        _ideal(), (_ideal(),), IdealComputationBudget()
    )

    assert verdict == "VERIFIED"


def test_adapter_scripts_gate_the_version_before_any_algebra() -> None:
    from jacobian.math.commutative_algebra_ops import _singular as adapter

    variables = ("x",)
    source = _ideal()
    saturator = RationalPolynomialIdeal(
        variables=variables,
        generators=(_single_term_poly(variables, (1,)),),
    )
    claimed = (_monomial_ideal(variables, (2,)),)
    scripts = [
        adapter._minimal_primes_script(source).decode("ascii"),
        adapter._minimal_primes_verification_script(source, claimed).decode("ascii"),
        adapter._verification_script(source, saturator, claimed[0]).decode("ascii"),
    ]

    for script in scripts:
        assert script.count('system("version")') == 1
        assert script.index('system("version")') < script.index('LIB "primdec.lib"')


def test_unsupported_version_quit_before_algebra_is_typed_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4.x release quits at the preamble without running decomposition."""

    executable = _executable(
        tmp_path,
        f"print({chr(10).join(('JACOBIAN_SINGULAR_IDEAL_V1', '45000'))!r})",
    )
    _select_executable(monkeypatch, executable)

    produced = run_singular_minimal_primes(_ideal(), IdealComputationBudget())
    verified = run_singular_minimal_primes_verification(
        _ideal(), (_ideal(),), IdealComputationBudget()
    )

    assert produced.outcome == "UNAVAILABLE"
    assert produced.components is None
    assert produced.detail == "The installed Singular release is unsupported."
    assert verified == "UNAVAILABLE"


def test_exact_result_limit_is_not_reported_as_invalid_backend_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        "1",
        "GENERATOR",
        *(f"1|{exponent}" for exponent in range(1_025)),
        "END_GENERATOR",
        "END",
    ]
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "LIMIT_EXCEEDED"
    assert (
        result.detail == "The exact Singular ideal exceeds the declared result bound."
    )


def test_large_canonical_coefficients_decode_without_the_digit_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical coefficient beyond Python's 4300-digit int() cap decodes.

    MAX_CANONICAL_RATIONAL_DIGITS admits 32,768-digit components, so a
    minAssGTZE result carrying a 5,001-digit numerator must be decoded by
    the chunked canonical parser instead of ``int(text)``.
    """

    numerator = format_canonical_integer(10**5000)
    assert len(numerator) > 4300
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        "1",
        "COMPONENT",
        "1",
        "GENERATOR",
        f"{numerator}/3|0",
        "END_GENERATOR",
        "END_COMPONENT",
        "END",
    )
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    coefficient = result.components[0].generators[0].polynomial.terms[0].coefficient
    assert coefficient == CanonicalRational.from_fraction(Fraction(10**5000, 3))


def test_near_envelope_high_digit_family_decodes_fully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admitted family may encode above any fixed capture cap.

    Sixteen terms whose canonical coefficient components approach the
    accepted 32,768-digit representation limit encode to roughly 1 MiB of
    protocol — over the former 512 KiB ceiling yet far inside the admitted
    64-generator/1,024-term envelope — so the producing pass sizes its
    capture allowance from the admitted envelope and decodes the complete
    family instead of reporting LIMIT_EXCEEDED.
    """

    numerator = "9" * 32_000
    denominator = "9" * 31_999 + "8"
    records = ["JACOBIAN_SINGULAR_IDEAL_V1", "44000", "1", "COMPONENT", "16"]
    for exponent in range(16):
        term = f"{numerator}/{denominator}|{exponent}," + ",".join(
            "0" for _ in range(7)
        )
        records.extend(("GENERATOR", term, "END_GENERATOR"))
    records.extend(("END_COMPONENT", "END"))
    protocol = "\n".join(records) + "\n"
    encoded = protocol.encode("ascii")
    assert len(encoded) > 512 * 1024

    executable = _executable(tmp_path, f"import sys; sys.stdout.write({protocol!r})")
    _select_executable(monkeypatch, executable)
    budget = IdealComputationBudget()

    result = run_singular_minimal_primes(_eight_var_ideal(), budget)

    assert len(encoded) <= _minimal_primes_stdout_limit(_eight_var_ideal(), budget)
    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert len(result.components) == 1
    generators = result.components[0].generators
    assert len(generators) == 16
    expected = CanonicalRational.from_fraction(
        Fraction(
            parse_canonical_integer(numerator),
            parse_canonical_integer(denominator),
        )
    )
    assert [
        generator.polynomial.terms[0].exponents[0] for generator in generators
    ] == list(range(16))
    assert all(len(generator.polynomial.terms) == 1 for generator in generators)
    assert all(
        generator.polynomial.terms[0].coefficient == expected
        for generator in generators
    )


def test_family_over_canonical_output_limit_is_typed_result_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representable family cannot escape the canonical output envelope."""

    numerator = "9" * 32_768
    denominator = "9" * 32_767 + "8"
    records = ["JACOBIAN_SINGULAR_IDEAL_V1", "44000", "1", "COMPONENT", "1"]
    records.append("GENERATOR")
    records.extend(
        f"{numerator}/{denominator}|{exponent}," + ",".join("0" for _ in range(7))
        for exponent in range(161)
    )
    records.extend(("END_GENERATOR", "END_COMPONENT", "END"))
    protocol = "\n".join(records) + "\n"
    assert len(protocol.encode("ascii")) > CanonicalLimits().max_output_bytes

    executable = _executable(tmp_path, f"import sys; sys.stdout.write({protocol!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.components is None
    assert result.detail == (
        "The exact Singular minimal-prime family exceeds the declared result bound."
    )


def test_stderr_on_zero_exit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path,
        'import sys; print("warning", file=sys.stderr)',
    )
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None


def test_oversized_stdout_is_a_typed_result_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("x" * 600_000)')
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.ideal is None
    assert (
        result.detail == "The exact Singular ideal exceeds the declared result bound."
    )


def test_request_scoped_directory_is_removed_after_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)
    created: list[Path] = []

    class RecordingTemporaryDirectory(tempfile.TemporaryDirectory[str]):
        def __enter__(self) -> str:
            directory = super().__enter__()
            created.append(Path(directory))
            return directory

    monkeypatch.setattr(
        "jacobian.math._singular.tempfile.TemporaryDirectory",
        RecordingTemporaryDirectory,
    )

    run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert created
    assert all(not directory.exists() for directory in created)


def _single_term_poly(
    variables: tuple[str, ...], exponents: tuple[int, ...]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"),
                    exponents=exponents,
                ),
            )
        ),
    )


def _zero_poly(variables: tuple[str, ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(terms=()),
    )


def _monomial_ideal(
    variables: tuple[str, ...], monomial: tuple[int, ...]
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=variables,
        generators=(_single_term_poly(variables, monomial),),
    )


def test_verification_narrows_the_allowance_like_the_producing_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def bounded_process(
        command: object,
        *,
        timeout_seconds: float,
        resource_limits: ProcessResourceLimits | None = None,
        **kwargs: object,
    ) -> BoundedProcessResult:
        seen["timeout_seconds"] = timeout_seconds
        seen["cpu_seconds"] = (
            resource_limits.cpu_seconds if resource_limits is not None else None
        )
        return BoundedProcessResult(
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
            cancelled=False,
        )

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        bounded_process,
    )
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: "/usr/bin/Singular" if name == "Singular" else None,
    )

    verdict = run_singular_minimal_primes_verification(
        _ideal(),
        (_ideal(),),
        IdealComputationBudget(wall_seconds=60),
        wall_seconds=6.75,
    )

    assert verdict == "TIMEOUT"
    assert seen["timeout_seconds"] == 6.75
    assert seen["cpu_seconds"] == 7


def test_exhausted_verification_allowance_times_out_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(command: object, **kwargs: object) -> BoundedProcessResult:
        raise AssertionError("Singular must not launch on an exhausted allowance")

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        forbidden,
    )

    verdict = run_singular_minimal_primes_verification(
        _ideal(), (_ideal(),), IdealComputationBudget(), wall_seconds=-0.5
    )

    assert verdict == "TIMEOUT"


def test_missing_backend_is_a_typed_unavailable_verification_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: None,
    )

    verdict = run_singular_minimal_primes_verification(
        _ideal(), (_ideal(),), IdealComputationBudget()
    )

    assert verdict == "UNAVAILABLE"


def test_malformed_verification_output_is_an_error_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)

    verdict = run_singular_minimal_primes_verification(
        _ideal(), (_ideal(),), IdealComputationBudget()
    )

    assert verdict == "ERROR"


def _sorted_family(
    *components: RationalPolynomialIdeal,
) -> tuple[RationalPolynomialIdeal, ...]:
    return tuple(sorted(components, key=lambda ideal: ideal.model_dump_json()))


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_verification_accepts_the_true_axes_family() -> None:
    variables = ("x", "y")
    source = _monomial_ideal(variables, (1, 1))
    claimed = _sorted_family(
        _monomial_ideal(variables, (1, 0)),
        _monomial_ideal(variables, (0, 1)),
    )

    verdict = run_singular_minimal_primes_verification(
        source, claimed, IdealComputationBudget()
    )

    assert verdict == "VERIFIED"


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_verification_refutes_a_non_prime_single_component() -> None:
    variables = ("x", "y")
    source = _monomial_ideal(variables, (1, 1))
    claimed = _sorted_family(_monomial_ideal(variables, (1, 1)))

    verdict = run_singular_minimal_primes_verification(
        source, claimed, IdealComputationBudget()
    )

    assert verdict == "REFUTED"


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_verification_refutes_a_family_missing_a_component() -> None:
    variables = ("x", "y")
    source = _monomial_ideal(variables, (1, 1))
    claimed = _sorted_family(_monomial_ideal(variables, (1, 0)))

    verdict = run_singular_minimal_primes_verification(
        source, claimed, IdealComputationBudget()
    )

    assert verdict == "REFUTED"


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_verification_refutes_a_non_minimal_family() -> None:
    variables = ("x", "y")
    source = _monomial_ideal(variables, (1, 0))
    claimed = _sorted_family(
        _monomial_ideal(variables, (1, 0)),
        _monomial_ideal(variables, (1, 1)),
    )

    verdict = run_singular_minimal_primes_verification(
        source, claimed, IdealComputationBudget()
    )

    assert verdict == "REFUTED"


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_verification_decides_degenerate_sources_structurally() -> None:
    variables = ("x", "y")
    unit = _monomial_ideal(variables, (0, 0))
    zero_source = RationalPolynomialIdeal(
        variables=variables, generators=(_zero_poly(variables),)
    )
    zero_family = _sorted_family(
        RationalPolynomialIdeal(
            variables=variables, generators=(_zero_poly(variables),)
        )
    )

    empty_unit = run_singular_minimal_primes_verification(
        unit, (), IdealComputationBudget()
    )
    populated_unit = run_singular_minimal_primes_verification(
        unit, (unit,), IdealComputationBudget()
    )
    zero_verdict = run_singular_minimal_primes_verification(
        zero_source, zero_family, IdealComputationBudget()
    )

    assert empty_unit == "VERIFIED"
    assert populated_unit == "REFUTED"
    assert zero_verdict == "VERIFIED"
