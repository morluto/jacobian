"""Exact closed-form kernels for square-free affine local arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from jacobian.math.number_theory.squarefree_affine_forms._models import (
    CROSSCHECK_MAX_PRIME,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)


@dataclass(frozen=True, slots=True)
class _NoSolutions:
    """The empty solution set of one affine congruence."""


@dataclass(frozen=True, slots=True)
class _SolutionCoset:
    """A nonempty affine coset of solutions modulo ``p²``.

    ``count`` and ``stride`` are concrete, rather than nullable, so callers
    cannot accidentally treat an empty profile as a partially specified coset.
    The kernel additionally establishes ``count * stride == p²``.
    """

    count: int
    root: int
    stride: int

    def __post_init__(self) -> None:
        if self.count < 1 or self.stride < 1:
            raise ValueError("a solution coset must be nonempty")
        if not 0 <= self.root < self.stride:
            raise ValueError("a coset root must be canonical modulo its stride")


_SolutionProfile = _NoSolutions | _SolutionCoset


def form_solution_profile(
    form: SquarefreeAffineForm, prime: int
) -> _SolutionProfile:
    """Return the empty set or one concrete solution coset modulo ``p²``.

    The solution set of one affine congruence modulo ``m=p²`` is empty or a
    single coset ``{root + t*stride : 0 <= t < count}`` with
    ``count*stride = m`` and ``count = gcd(a, m)`` (or ``count = m`` when
    ``a = b = 0 mod m``). No residue enumeration is required.
    """

    modulus = prime * prime
    coefficient = form.coefficient % modulus
    constant = form.constant % modulus
    if coefficient == 0:
        return _SolutionCoset(modulus, 0, 1) if constant == 0 else _NoSolutions()
    divisor = gcd(coefficient, modulus)
    if constant % divisor != 0:
        return _NoSolutions()
    stride = modulus // divisor
    root = (-(constant // divisor) * pow(coefficient // divisor, -1, stride)) % stride
    return _SolutionCoset(divisor, root, stride)


def profile_residues(profile: _SolutionProfile) -> tuple[int, ...]:
    """Expand one closed-form solution profile into canonical residues."""

    if isinstance(profile, _NoSolutions):
        return ()
    return tuple(
        profile.root + offset * profile.stride for offset in range(profile.count)
    )


def replay_profile(
    form: SquarefreeAffineForm, prime: int, profile: _SolutionProfile
) -> None:
    """Replay every congruence of one profile exactly."""

    if isinstance(profile, _NoSolutions):
        return
    modulus = prime * prime
    for offset in range(profile.count):
        residue = profile.root + offset * profile.stride
        value = form.coefficient * residue + form.constant
        if not (0 <= residue < modulus and value % modulus == 0):
            raise RuntimeError("closed-form bad residue failed its defining congruence")


def enumerated_ledger(
    source: SquarefreeAffineFamily, prime: int
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Brute-force the complete residue ledger by direct p^2 enumeration."""

    modulus = prime * prime
    rows: list[tuple[int, tuple[str, ...]]] = []
    for residue in range(modulus):
        form_ids = tuple(
            sorted(
                form.form_id
                for form in source.forms
                if (form.coefficient * residue + form.constant) % modulus == 0
            )
        )
        if form_ids:
            rows.append((residue, form_ids))
    return tuple(rows)


def closed_form_ledger(
    source: SquarefreeAffineFamily, prime: int
) -> tuple[
    tuple[_SolutionProfile, ...],
    tuple[tuple[int, tuple[str, ...]], ...],
    bool,
]:
    """Return per-form profiles, the bad-residue ledger, and full coverage."""

    modulus = prime * prime
    profiles = tuple(form_solution_profile(form, prime) for form in source.forms)
    for form, profile in zip(source.forms, profiles, strict=True):
        replay_profile(form, prime, profile)
    covers_all = any(
        isinstance(profile, _SolutionCoset) and profile.count == modulus
        for profile in profiles
    )
    if covers_all:
        return profiles, (), True
    by_residue: dict[int, list[str]] = {}
    for form, profile in zip(source.forms, profiles, strict=True):
        for residue in profile_residues(profile):
            by_residue.setdefault(residue, []).append(form.form_id)
    ledger = tuple(
        (residue, tuple(sorted(form_ids)))
        for residue, form_ids in sorted(by_residue.items())
    )
    if prime <= CROSSCHECK_MAX_PRIME and ledger != enumerated_ledger(source, prime):
        raise RuntimeError(
            "closed-form ledger disagreed with complete residue enumeration"
        )
    return profiles, ledger, False


__all__ = [
    "closed_form_ledger",
    "enumerated_ledger",
    "form_solution_profile",
    "profile_residues",
    "replay_profile",
]
