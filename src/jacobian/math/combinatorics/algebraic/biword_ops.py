from __future__ import annotations

from typing import Any, Literal

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic._rsk import _row_insert
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
    BiwordNormalizeResult,
    BiwordRSKPair,
    GreeneResult,
    NonnegativeIntegerMatrix,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)


def _forward(
    b: Biword, source_kind: Literal["BIWORD", "MATRIX"] = "BIWORD"
) -> BiwordRSKPair:
    tr = {x: i for i, x in enumerate(b.top_alphabet, 1)}
    br = {x: i for i, x in enumerate(b.bottom_alphabet, 1)}
    p, q = _row_insert(tuple(br[x] for x in b.bottom))
    # _row_insert records positions; replace position with top rank.
    recording = []
    for row in q:
        recording.append(tuple(tr[b.top[x - 1]] for x in row))
    shape = IntegerPartition(parts=tuple(len(r) for r in p))
    return BiwordRSKPair(
        top_alphabet=b.top_alphabet,
        bottom_alphabet=b.bottom_alphabet,
        insertion_tableau=SemistandardYoungTableau(rows=p),
        recording_tableau=SemistandardYoungTableau(rows=tuple(recording)),
        shape=shape,
        source_kind=source_kind,
    )


def normalize_biword(request: Any) -> BiwordNormalizeResult:
    if len(request.top) != len(request.bottom):
        raise OperationDomainValidationError(
            location=("top",),
            code="algebraic_combinatorics.biword_length",
            message="biword rows must have equal length",
        )
    tr = {x: i for i, x in enumerate(request.top_alphabet)}
    br = {x: i for i, x in enumerate(request.bottom_alphabet)}
    if len(tr) != len(request.top_alphabet) or len(br) != len(request.bottom_alphabet):
        raise OperationDomainValidationError(
            location=("top_alphabet",),
            code="algebraic_combinatorics.biword_alphabet",
            message="biword alphabets must be unique",
        )
    if any(x not in tr for x in request.top) or any(
        x not in br for x in request.bottom
    ):
        raise OperationDomainValidationError(
            location=("top",),
            code="algebraic_combinatorics.biword_symbol",
            message="biword symbols must belong to their alphabets",
        )
    order = sorted(
        range(len(request.top)),
        key=lambda i: (tr[request.top[i]], br[request.bottom[i]], i),
    )
    top = tuple(request.top[i] for i in order)
    bottom = tuple(request.bottom[i] for i in order)
    return BiwordNormalizeResult(
        source_top=request.top,
        source_bottom=request.bottom,
        biword=Biword(
            top_alphabet=request.top_alphabet,
            bottom_alphabet=request.bottom_alphabet,
            top=top,
            bottom=bottom,
        ),
        permutation=tuple(i for i in order),
    )


def rsk_biword(b: Biword) -> BiwordRSKPair:
    return _forward(b)


def _admit_matrix(m: NonnegativeIntegerMatrix) -> NonnegativeIntegerMatrix:
    """Re-establish matrix shape, cell, mass, and expansion bounds."""
    try:
        # model_validate(instance) trusts an already-constructed model.  The
        # dump/reparse is intentional at this native admission boundary so a
        # forged model_construct() cannot reach list multiplication below.
        return NonnegativeIntegerMatrix.model_validate(m.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="algebraic_combinatorics.matrix_invalid",
            message=str(exc),
        ) from exc


def matrix_biword(m: NonnegativeIntegerMatrix) -> BiwordRSKPair:
    m = _admit_matrix(m)
    top = []
    bottom = []
    for i, a in enumerate(m.row_labels):
        for j, b in enumerate(m.column_labels):
            top.extend([a] * m.entries[i][j])
            bottom.extend([b] * m.entries[i][j])
    return _forward(
        Biword(
            top_alphabet=m.row_labels,
            bottom_alphabet=m.column_labels,
            top=tuple(top),
            bottom=tuple(bottom),
        ),
        "MATRIX",
    )


def _admit_pair(pair: BiwordRSKPair) -> BiwordRSKPair:
    """Re-establish authored pair invariants at the inverse boundary."""
    try:
        checked = BiwordRSKPair.model_validate(pair.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_incompatible",
            message=str(exc),
        ) from exc
    return checked


def _inverse(pair: BiwordRSKPair) -> Biword:
    pair = _admit_pair(pair)
    p = [list(r) for r in pair.insertion_tableau.rows]
    q = [list(r) for r in pair.recording_tableau.rows]
    reverse = []
    # Process recording labels from largest to smallest. Equal labels are
    # removed from the outer boundary right-to-left, exactly reversing the
    # lexicographically sorted equal-top biword block.
    max_label = max((x for r in q for x in r), default=0)
    for label in range(max_label, 0, -1):
        while True:
            candidates = [
                (i, len(r) - 1) for i, r in enumerate(q) if r and r[-1] == label
            ]
            if not candidates:
                break
            # For equal top labels, sorted bottom insertion appends the
            # latest pair at the uppermost available outer corner.
            i, _ = min(candidates)
            q[i].pop()
            current = p[i].pop()
            if not p[i]:
                p.pop(i)
                q.pop(i)
            for upper in range(i - 1, -1, -1):
                row = p[upper]
                pos = 0
                while pos < len(row) and row[pos] < current:
                    pos += 1
                if pos == 0:
                    raise ValueError("incompatible biword tableau pair")
                pos -= 1
                row[pos], current = current, row[pos]
            reverse.append((label, current))
    tr = pair.top_alphabet
    br = pair.bottom_alphabet
    pairs = list(reversed([(tr[a - 1], br[b - 1]) for a, b in reverse]))
    pairs.sort(key=lambda x: (tr.index(x[0]), br.index(x[1])))
    return Biword(
        top_alphabet=tr,
        bottom_alphabet=br,
        top=tuple(a for a, b in pairs),
        bottom=tuple(b for a, b in pairs),
    )


def inverse_biword(pair: BiwordRSKPair) -> Biword:
    try:
        return _inverse(pair)
    except OperationDomainValidationError:
        raise
    except (IndexError, KeyError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_incompatible",
            message=str(exc),
        ) from exc


def inverse_matrix(
    pair: BiwordRSKPair, rows: tuple[str, ...], columns: tuple[str, ...]
) -> NonnegativeIntegerMatrix:
    if pair.source_kind != "MATRIX":
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_source",
            message="matrix inverse requires a matrix RSK pair",
        )
    if rows != pair.top_alphabet or columns != pair.bottom_alphabet:
        raise OperationDomainValidationError(
            location=("row_labels",),
            code="algebraic_combinatorics.rsk_matrix_axes",
            message="inverse matrix axes must equal the pair's labelled alphabets",
        )
    try:
        b = _inverse(pair)
    except OperationDomainValidationError:
        raise
    except (IndexError, KeyError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair_incompatible",
            message=str(exc),
        ) from exc
    counts = {(a, c): 0 for a in rows for c in columns}
    for a, c in zip(b.top, b.bottom, strict=True):
        if a not in rows or c not in columns:
            raise ValueError("inverse labels do not match matrix axes")
        counts[a, c] += 1
    return NonnegativeIntegerMatrix(
        row_labels=rows,
        column_labels=columns,
        entries=tuple(tuple(counts[a, c] for c in columns) for a in rows),
    )


def greene(word: Any, requested_k: int | None = None) -> GreeneResult:
    ranks = {s: i for i, s in enumerate(word.alphabet)}
    vals = [ranks[x] for x in word.letters]
    p, _ = _row_insert(tuple(x + 1 for x in vals))
    shape = IntegerPartition(parts=tuple(len(r) for r in p))
    k = requested_k if requested_k is not None else min(20, max(1, len(shape.parts)))
    inc = tuple(sum(shape.parts[:i]) for i in range(1, min(k, len(shape.parts)) + 1))
    dec = tuple(
        sum(1 for r in shape.parts if r >= i)
        for i in range(1, min(k, shape.parts[0] if shape.parts else 0) + 1)
    )
    inc += (sum(shape.parts),) * (k - len(inc))
    dec += (sum(shape.parts),) * (k - len(dec))
    return GreeneResult(
        word=word, shape=shape, increasing_totals=inc, decreasing_totals=dec, k=k
    )


__all__ = [
    "greene",
    "inverse_biword",
    "inverse_matrix",
    "matrix_biword",
    "normalize_biword",
    "rsk_biword",
]
