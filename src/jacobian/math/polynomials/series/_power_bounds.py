"""Complete binary-power work and retained-coefficient bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jacobian.catalog.models import OperationResourceAdmissionError

if TYPE_CHECKING:
    from ._models import TruncatedSeries
MAX_POWER_ORDER = 2048
MAX_POWER_WORK = 1 << 30
MAX_POWER_LIMB_WORK = 1 << 36
MAX_POWER_SCRATCH_BITS = 1 << 18
MAX_POWER_STORAGE_BITS = 1 << 32


@dataclass(frozen=True)
class PowerEnvelope:
    """Conservative coefficient/limb policies, not CPU-time or RSS estimates."""

    work: int
    bits: int
    storage: int

    def checked(self) -> None:
        for name, amount, limit in (
            ("work", self.work, MAX_POWER_WORK),
            ("limb_work", self.work * ((self.bits + 63) // 64), MAX_POWER_LIMB_WORK),
            ("scratch", self.bits, MAX_POWER_SCRATCH_BITS),
            ("storage", self.storage, MAX_POWER_STORAGE_BITS),
        ):
            if amount > limit:
                raise OperationResourceAdmissionError(
                    location=(),
                    code=f"formal_power_series.power_{name}",
                    message=f"proved series power {name} {amount} exceeds {limit}",
                )


def power_envelope(
    source: TruncatedSeries, exponent: int, norm_bits: int, denominator_bits: int
) -> PowerEnvelope:
    """Price the owned binary loop, including untruncated FLINT products.

    Write f=A/D after bounded common-denominator clearing. Every base or
    accumulated result is a truncated f**j with j<=exponent. Its numerator
    l1 norm is <=max(1,||A||_1)**j and its denominator divides D**j. The
    next full product has the same bound at the summed exponent; truncation
    and canonical normalization can only decrease these bounds.

    The degree bound min(N-1, degree(f)*j) prices both operand arrays before
    multiplication. Eight units per coefficient pair plus eight per full
    output slot cover convolution, normalization and copies; 32*N reserves
    source recognition, denominator clearing, conversion and final decoding.
    At most 32*N coefficient slots cover retained inputs/outputs and the
    simultaneous pre-truncation product arrays. The limb-width-weighted cap
    rejects expensive high-height dense cases independently of result height.
    """
    n = source.truncation_order
    source_bits = max(
        max(abs(v.num).bit_length(), v.den.bit_length()) for v in source.coefficients
    )
    bits = max(1, source_bits, exponent * norm_bits, exponent * denominator_bits) + 2
    degree = max((i for i, v in enumerate(source.coefficients) if v.num), default=0)
    result_degree = 0
    work = 32 * n
    remaining = exponent
    while remaining:
        if remaining & 1:
            work += 8 * (result_degree + 1) * (degree + 1) + 8 * (
                result_degree + degree + 1
            )
            result_degree = min(n - 1, result_degree + degree)
        remaining >>= 1
        if remaining:
            work += 8 * (degree + 1) ** 2 + 8 * (2 * degree + 1)
            degree = min(n - 1, 2 * degree)
    return PowerEnvelope(work, bits, 32 * n * bits)
