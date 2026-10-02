"""Output and recursive integer-PRS admission for square-free decomposition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, prod
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationResourceAdmissionError

# Scalar arithmetic counts and exact coefficient storage, not wall-time or RSS.
MAX_SQUARE_FREE_WORK = 1 << 40
MAX_SQUARE_FREE_SCRATCH_BITS = 1 << 24
MAX_SQUARE_FREE_STORAGE_BITS = 1 << 33
MAX_SQUARE_FREE_CELLS = 1 << 20
_CANONICAL_BITS = MAX_CANONICAL_RATIONAL_DIGITS * 3_321_928 // 1_000_000
_CANONICAL_MAGNITUDE = 10**MAX_CANONICAL_RATIONAL_DIGITS


def refuse(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("polynomial",),
        code=f"polynomial.square_free_{reason}",
        message=message,
    )


def canonical_height(bits: int) -> None:
    if bits > _CANONICAL_BITS:
        refuse(
            "result_height",
            "proved square-free factor height exceeds the canonical rational envelope",
        )


def canonical_components(values: Iterable[Fraction]) -> None:
    if any(
        abs(q.numerator) >= _CANONICAL_MAGNITUDE
        or q.denominator >= _CANONICAL_MAGNITUDE
        for q in values
    ):
        refuse(
            "result_height",
            "exact monic normalization exceeds the canonical rational envelope",
        )


def divisor_bits(degrees: tuple[int, ...], height: int) -> int:
    """Mixed-radix Kronecker injection followed by the Mignotte bound.

    Every divisor has degree box at most the source box B. Its integer
    coefficient magnitude is at most 2^(B-1)*sqrt(B)*2^height; the integer
    expression below deliberately rounds upward. Primitive leading
    coefficients obey the same bound, hence so do monic denominators.
    """
    box = prod(degree + 1 for degree in degrees)
    return height + box + box.bit_length() + 1


@dataclass(frozen=True)
class Envelope:
    work: int = 0
    bits: int = 1
    storage: int = 0

    def checked(self) -> Envelope:
        for reason, value, limit in (
            ("work", self.work, MAX_SQUARE_FREE_WORK),
            ("scratch_height", self.bits, MAX_SQUARE_FREE_SCRATCH_BITS),
            ("coefficient_storage", self.storage, MAX_SQUARE_FREE_STORAGE_BITS),
        ):
            if value > limit:
                refuse(
                    reason,
                    f"proved square-free {reason} envelope {value} exceeds {limit}",
                )
        return self


@dataclass(frozen=True)
class IntegerSource:
    terms: dict[tuple[int, ...], int]
    scale: Fraction
    height: int
    envelope: Envelope


def source_storage(terms: dict[tuple[int, ...], Fraction]) -> int:
    return 4 * sum(
        abs(q.numerator).bit_length() + q.denominator.bit_length()
        for q in terms.values()
    )


def clearing_envelope(terms: dict[tuple[int, ...], Fraction]) -> Envelope:
    denominator_bits = sum(
        d.bit_length() for d in {q.denominator for q in terms.values()}
    )
    height = max(abs(q.numerator).bit_length() for q in terms.values())
    bits = height + denominator_bits + 1
    return Envelope(
        work=8 * len(terms),
        bits=bits,
        storage=source_storage(terms) + 4 * len(terms) * bits,
    ).checked()


def clear_source(terms: dict[tuple[int, ...], Fraction]) -> IntegerSource:
    """Clear denominators only after bounding the allocation and arithmetic."""
    envelope = clearing_envelope(terms)
    denominators = {value.denominator for value in terms.values()}
    denominator = 1
    for value in denominators:
        denominator = denominator // gcd(denominator, value) * value
    integers = {
        powers: value.numerator * (denominator // value.denominator)
        for powers, value in terms.items()
    }
    content = 0
    for value in integers.values():
        content = gcd(content, abs(value))
    if integers[max(integers)] < 0:
        content = -content
    primitive = {powers: value // content for powers, value in integers.items()}
    return IntegerSource(
        terms=primitive,
        scale=Fraction(content, denominator),
        height=max(abs(value).bit_length() for value in primitive.values()),
        envelope=envelope,
    )


def gcd_envelope(
    degrees: tuple[int, ...],
    height: int,
    cache: dict[tuple[tuple[int, ...], int], Envelope],
) -> Envelope:
    """Conservative complete controlled-content/PRS recursion envelope.

    Primitive factors divide their inputs. Completed subresultants are
    determinants of order <=2d with tail degrees <=2d*D_i. Pseudo-division
    and leading-coefficient powers need the larger (d+2)-scaled scratch
    box, but only completed values enter recursive content GCDs.
    """
    degrees = tuple(degree for degree in degrees if degree)
    if not degrees:
        return Envelope(work=8, bits=height, storage=8 * height).checked()
    key = degrees, height
    if key in cache:
        return cache[key]
    d, tail = degrees[0], degrees[1:]
    primitive = divisor_bits(degrees, height)
    determinant = (
        2
        * d
        * (primitive + prod(x + 1 for x in tail).bit_length() + (2 * d).bit_length())
        + 1
    )
    completed = tuple(2 * d * x for x in tail)
    scratch = (d + 1) * prod(2 * d * (d + 2) * x + 1 for x in tail)
    if scratch > MAX_SQUARE_FREE_CELLS:
        refuse(
            "scratch_support",
            f"proved square-free scratch support {scratch} exceeds {MAX_SQUARE_FREE_CELLS}",
        )
    # Dividing the last subresultant by its polynomial content may involve
    # a content coefficient larger than an individual determinant coefficient.
    # The quotient is a primitive GCD (hence an original-input divisor), while
    # the content divides a completed coefficient polynomial. Bound their raw
    # convolution explicitly instead of relying on slack in the PRS estimate.
    content_height = divisor_bits(completed, determinant)
    bits = max(
        (2 * d + 3) * (determinant + scratch.bit_length()) + 2 * primitive + 16,
        content_height + primitive + scratch.bit_length() + 2,
    )
    # At most d+1 PRS descents and d+1 pseudo-division cancellations.
    # Dense multiplication, powering and exact division in this fixed box
    # cost <=O(m*scratch^2); 256 includes primitive normalization and copies.
    work = 256 * len(degrees) * (d + 1) ** 2 * scratch**2
    completed_box = (d + 1) * prod(x + 1 for x in completed)
    storage = 8 * scratch * bits + 4 * (d + 1) * completed_box * determinant
    local = Envelope(work, bits, storage).checked()
    if tail:
        before = gcd_envelope(tail, divisor_bits(tail, height), cache)
        after = gcd_envelope(completed, divisor_bits(completed, determinant), cache)
        # Two initial contents plus their GCD: <=2d+1 recursive pairs.
        # The last subresultant contributes at most d further content pairs.
        local = Envelope(
            work=local.work + (2 * d + 1) * before.work + d * after.work,
            bits=max(local.bits, before.bits, after.bits),
            storage=local.storage + max(before.storage, after.storage),
        ).checked()
    cache[key] = local
    return local


def decomposition_envelope(degrees: tuple[int, ...], height: int) -> Envelope:
    active = tuple(degree for degree in degrees if degree)
    if not active:
        return Envelope(work=1, bits=height, storage=height).checked()
    box = prod(degree + 1 for degree in active)
    factor_height = divisor_bits(active, height)
    canonical_height(factor_height)
    gcd_bound = gcd_envelope(active, factor_height + max(active).bit_length(), {})
    calls = len(active) + max(active)
    # Initial gradient GCDs and one GCD per possible multiplicity. Every
    # subsequent polynomial is an integer divisor of the primitive source.
    # Exact divisions and all reconstruction products stay in its degree box.
    replay = 128 * (len(active) + 1) * (max(active) + 1) * box**2
    return Envelope(
        work=calls * gcd_bound.work + replay,
        bits=max(gcd_bound.bits, 2 * factor_height + box.bit_length() + 2),
        storage=gcd_bound.storage + 8 * box * factor_height,
    ).checked()
