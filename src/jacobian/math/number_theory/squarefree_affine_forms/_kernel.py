"""Exact closed-form kernels for square-free affine local arithmetic."""

from __future__ import annotations

from math import gcd

from jacobian.math.number_theory.squarefree_affine_forms._models import (
    CROSSCHECK_MAX_PRIME,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)


def form_solution_profile(
    form: SquarefreeAffineForm, prime: int
) -> tuple[int, int | None, int | None]:
    """Return ``(count, root, stride)`` for ``{r mod p^2 : p^2 | a*r+b}``.

    The solution set of one affine congruence modulo ``m=p^2`` is empty or a
    single coset ``{root + t*stride : 0 <= t < count}`` with
    ``count*stride = m`` and ``count = gcd(a, m)`` (or ``count = m`` when
    ``a = b = 0 mod m``). No residue enumeration is required.
    """

    modulus = prime * prime
    coefficient = form.coefficient % modulus
    constant = form.constant % modulus
    if coefficient == 0:
        return (modulus, 0, 1) if constant == 0 else (0, None, None)
    divisor = gcd(coefficient, modulus)
    if constant % divisor != 0:
        return (0, None, None)
    stride = modulus // divisor
    root = (-(constant // divisor) * pow(coefficient // divisor, -1, stride)) % stride
    return (divisor, root, stride)


def profile_residues(profile: tuple[int, int | None, int | None]) -> tuple[int, ...]:
    """Expand one closed-form solution profile into canonical residues."""

    count, root, stride = profile
    if count == 0 or root is None or stride is None:
        return ()
    return tuple(root + offset * stride for offset in range(count))


def replay_profile(
    form: SquarefreeAffineForm, prime: int, profile: tuple[int, int | None, int | None]
) -> None:
    """Replay every congruence of one profile exactly."""

    modulus = prime * prime
    count, root, stride = profile
    if count == 0:
        return
    assert root is not None and stride is not None
    for offset in range(count):
        residue = root + offset * stride
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
    tuple[tuple[int, int | None, int | None], ...],
    tuple[tuple[int, tuple[str, ...]], ...],
    bool,
]:
    """Return per-form profiles, the bad-residue ledger, and full coverage."""

    modulus = prime * prime
    profiles = tuple(form_solution_profile(form, prime) for form in source.forms)
    for form, profile in zip(source.forms, profiles, strict=True):
        replay_profile(form, prime, profile)
    covers_all = any(count == modulus for count, _, _ in profiles)
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
