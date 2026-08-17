"""Domain tests for the Boolean function analysis operations."""

from __future__ import annotations

import pytest

from jacobian.contracts.boolean_analysis import (
    ErasureNoiseRequest,
    ErasureNoiseResult,
    FourierSpectrumRequest,
    FourierSpectrumResult,
    MultilinearExtensionRequest,
    MultilinearExtensionResult,
    TruthTableRequest,
    TruthTableResult,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.domains.boolean_analysis.operations import (
    compute_erasure_noise,
    compute_fourier_spectrum,
    compute_multilinear_extension,
    compute_truth_table,
)


def _zero() -> CanonicalRational:
    return CanonicalRational(num="0", den="1")


def _one() -> CanonicalRational:
    return CanonicalRational(num="1", den="1")


def _truth_table(values: list[int]) -> tuple[CanonicalRational, ...]:
    return tuple(_one() if v == 1 else _zero() for v in values)


def _rational(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=str(num), den=str(den))


# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------


def test_truth_table_single_variable() -> None:
    result = compute_truth_table(TruthTableRequest(truth_table=_truth_table([0, 1])))
    assert isinstance(result, TruthTableResult)
    assert result.variable_count == 1
    assert [entry.as_fraction() for entry in result.truth_table] == [0, 1]


def test_truth_table_two_variables() -> None:
    result = compute_truth_table(
        TruthTableRequest(truth_table=_truth_table([1, 0, 1, 1]))
    )
    assert result.variable_count == 2
    assert len(result.truth_table) == 4
    assert result.convention == "NATURAL_ORDER"


def test_truth_table_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        TruthTableRequest(truth_table=_truth_table([0, 1, 1]))


def test_truth_table_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least 2 items"):
        TruthTableRequest.model_validate({"truth_table": []})


def test_truth_table_rejects_non_boolean_values() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        TruthTableRequest.model_validate(
            {"truth_table": [{"num": "2", "den": "1"}, {"num": "1", "den": "1"}]}
        )


# ---------------------------------------------------------------------------
# Fourier spectrum
# ---------------------------------------------------------------------------


def test_fourier_spectrum_constant_zero_is_all_zeros() -> None:
    result = compute_fourier_spectrum(
        FourierSpectrumRequest(truth_table=_truth_table([0, 0, 0, 0]))
    )
    assert isinstance(result, FourierSpectrumResult)
    assert result.variable_count == 2
    assert [c.as_integer_ratio()[0] for c in result.spectrum] == [0, 0, 0, 0]


def test_fourier_spectrum_constant_one() -> None:
    result = compute_fourier_spectrum(
        FourierSpectrumRequest(truth_table=_truth_table([1, 1, 1, 1]))
    )
    # W[0] = sum of all entries = 4; all other coefficients are zero.
    assert [c.as_integer_ratio()[0] for c in result.spectrum] == [4, 0, 0, 0]
    assert result.variable_count == 2


def test_fourier_spectrum_single_variable_identity() -> None:
    # f(x) = x: truth table [0, 1]; W = [1, -1]
    result = compute_fourier_spectrum(
        FourierSpectrumRequest(truth_table=_truth_table([0, 1]))
    )
    assert [c.as_integer_ratio()[0] for c in result.spectrum] == [1, -1]


def test_fourier_spectrum_and_function() -> None:
    # AND(0,0)=0, AND(0,1)=0, AND(1,0)=0, AND(1,1)=1
    result = compute_fourier_spectrum(
        FourierSpectrumRequest(truth_table=_truth_table([0, 0, 0, 1]))
    )
    # W[0] = sum = 1, W[1] = f00-f01... using popcount parity
    # Standard result: W = [1, -1, -1, 1]
    assert [c.as_integer_ratio()[0] for c in result.spectrum] == [1, -1, -1, 1]


def test_fourier_spectrum_matches_definition() -> None:
    # Verify W[k] = sum_x f(x) (-1)^popcount(x & k) against a random table.
    import random

    rng = random.Random(42)
    values = [rng.randint(0, 1) for _ in range(8)]
    result = compute_fourier_spectrum(
        FourierSpectrumRequest(truth_table=_truth_table(values))
    )
    spectrum = [c.as_integer_ratio()[0] for c in result.spectrum]
    for k in range(8):
        expected = 0
        for x in range(8):
            expected += values[x] * (-1) ** bin(x & k).count("1")
        assert spectrum[k] == expected


def test_fourier_spectrum_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        FourierSpectrumRequest(truth_table=_truth_table([0, 1, 1]))


# ---------------------------------------------------------------------------
# Multilinear extension
# ---------------------------------------------------------------------------


def test_multilinear_extension_identity() -> None:
    # f(0)=0, f(1)=1 => f~(x0) = x0
    result = compute_multilinear_extension(
        MultilinearExtensionRequest(truth_table=_truth_table([0, 1]))
    )
    assert isinstance(result, MultilinearExtensionResult)
    assert result.variable_count == 1
    assert result.polynomial == "x0"


def test_multilinear_extension_constant_one() -> None:
    # f = 1 on both inputs => f~ = 1
    result = compute_multilinear_extension(
        MultilinearExtensionRequest(truth_table=_truth_table([1, 1]))
    )
    assert result.polynomial == "1"


def test_multilinear_extension_constant_zero() -> None:
    result = compute_multilinear_extension(
        MultilinearExtensionRequest(truth_table=_truth_table([0, 0, 0, 0]))
    )
    assert result.polynomial == "0"


def test_multilinear_extension_and_function() -> None:
    # AND(0,0)=0, AND(0,1)=0, AND(1,0)=0, AND(1,1)=1 => f~ = x0*x1
    result = compute_multilinear_extension(
        MultilinearExtensionRequest(truth_table=_truth_table([0, 0, 0, 1]))
    )
    assert result.polynomial == "x0*x1"


def test_multilinear_extension_agrees_on_hypercube() -> None:
    import sympy

    truth = [0, 1, 1, 0, 1, 0, 0, 1]
    result = compute_multilinear_extension(
        MultilinearExtensionRequest(truth_table=_truth_table(truth))
    )
    assert result.variable_count == 3
    symbols = sympy.symbols("x0:3")
    poly = sympy.sympify(result.polynomial)
    for x in range(8):
        assignment = {symbols[i]: (x >> i) & 1 for i in range(3)}
        assert poly.subs(assignment) == truth[x], f"MLE disagrees at {x}"


# ---------------------------------------------------------------------------
# Erasure noise
# ---------------------------------------------------------------------------


def test_erasure_noise_p_one_is_mean() -> None:
    # f=[0,1] => spectrum [1,-1], f_hat=[1/2,-1/2].
    # E = sum_S f_hat(S) p^|S| = 1/2 + (-1/2)*1 = 0 when p=1.
    result = compute_erasure_noise(
        ErasureNoiseRequest(
            truth_table=_truth_table([0, 1]),
            probability=_rational(1, 1),
            base_input=(0,),
        )
    )
    assert isinstance(result, ErasureNoiseResult)
    assert result.expected_value.as_integer_ratio() == (0, 1)
    assert result.variable_count == 1


def test_erasure_noise_p_zero_is_half() -> None:
    # E = sum_S f_hat(S) * 0^|S| = f_hat(empty) = mean = 1/2 when p=0.
    result = compute_erasure_noise(
        ErasureNoiseRequest(
            truth_table=_truth_table([0, 1]),
            probability=_rational(0, 1),
            base_input=(0,),
        )
    )
    assert result.expected_value.as_integer_ratio() == (1, 2)


def test_erasure_noise_p_half() -> None:
    # f(0)=0, f(1)=1, p=1/2
    # E[f] = sum_S f_hat(S) p^|S|
    # f_hat(0) = 1/2, f_hat(1) = -1/2 (from spectrum [1, -1] / 2)
    # E = 1/2 * (1/2)^0 + (-1/2) * (1/2)^1 = 1/2 - 1/4 = 1/4
    result = compute_erasure_noise(
        ErasureNoiseRequest(
            truth_table=_truth_table([0, 1]),
            probability=_rational(1, 2),
            base_input=(0,),
        )
    )
    assert result.expected_value.as_integer_ratio() == (1, 4)


def test_erasure_noise_constant_one() -> None:
    result = compute_erasure_noise(
        ErasureNoiseRequest(
            truth_table=_truth_table([1, 1, 1, 1]),
            probability=_rational(3, 7),
            base_input=(0, 0),
        )
    )
    assert result.expected_value.as_fraction() == 1  # type: ignore[comparison]


def test_erasure_noise_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability must be in"):
        ErasureNoiseRequest(
            truth_table=_truth_table([0, 1]),
            probability=_rational(3, 2),
            base_input=(0,),
        )


def test_erasure_noise_rejects_negative_probability() -> None:
    with pytest.raises(ValueError, match="probability must be in"):
        ErasureNoiseRequest(
            truth_table=_truth_table([0, 1]),
            probability=_rational(-1, 2),
            base_input=(0,),
        )


def test_erasure_noise_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        ErasureNoiseRequest.model_validate(
            {
                "truth_table": [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ],
                "probability": {"num": "1", "den": "2"},
                "base_input": [0, 0, 0],
            }
        )
