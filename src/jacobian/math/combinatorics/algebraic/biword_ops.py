from __future__ import annotations

from typing import Any, Literal

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic._rsk import _row_insert
from jacobian.math.combinatorics.algebraic.biword import (
    MAX_BIWORD_AXIS,
    MAX_BIWORD_LABEL_BYTES,
    MAX_BIWORD_MASS,
    Biword,
    BiwordNormalizeResult,
    BiwordRSKPair,
    GreeneResult,
    NonnegativeIntegerMatrix,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    require_semistandard,
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
        raise ValueError("biword rows must have equal length")
    tr = {x: i for i, x in enumerate(request.top_alphabet)}
    br = {x: i for i, x in enumerate(request.bottom_alphabet)}
    if len(tr) != len(request.top_alphabet) or len(br) != len(request.bottom_alphabet):
        raise ValueError("biword alphabets must be unique")
    if any(x not in tr for x in request.top) or any(
        x not in br for x in request.bottom
    ):
        raise ValueError("biword symbols must belong to their alphabets")
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


def matrix_biword(m: NonnegativeIntegerMatrix) -> BiwordRSKPair:
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


def _validate_pair(pair: BiwordRSKPair) -> None:
    """Re-establish source/alphabet/tableau relations at the inverse boundary."""
    if not isinstance(pair, BiwordRSKPair):
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair",
            message="inverse RSK requires a canonical pair",
        )
    top = pair.top_alphabet
    bottom = pair.bottom_alphabet
    if (
        not isinstance(top, tuple)
        or not isinstance(bottom, tuple)
        or any(type(x) is not str for x in (*top, *bottom))
    ):
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_alphabet",
            message="pair alphabets are malformed",
        )
    if len(top) > MAX_BIWORD_AXIS or len(bottom) > MAX_BIWORD_AXIS:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_axes",
            message="pair alphabets exceed the axis envelope",
        )
    try:
        label_bytes = sum(len(x.encode("utf-8")) for x in (*top, *bottom))
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_alphabet",
            message="pair alphabet labels must be UTF-8",
        ) from exc
    if label_bytes > MAX_BIWORD_LABEL_BYTES:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_label_bytes",
            message="pair alphabet labels exceed the byte envelope",
        )
    if len(set(top)) != len(top) or len(set(bottom)) != len(bottom):
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_alphabet",
            message="pair alphabets must be unique",
        )
    try:
        if (
            pair.insertion_tableau.shape != pair.shape
            or pair.recording_tableau.shape != pair.shape
        ):
            raise ValueError("tableaux and shape disagree")
        require_semistandard(pair.insertion_tableau)
        require_semistandard(pair.recording_tableau)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_pair",
            message="tableaux do not satisfy the semistandard pair contract",
        ) from exc
    cells = sum(len(row) for row in pair.insertion_tableau.rows) + sum(
        len(row) for row in pair.recording_tableau.rows
    )
    if cells > 2 * MAX_BIWORD_MASS:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_cells",
            message="pair tableau cells exceed the inverse envelope",
        )
    for rows, limit, where in (
        (pair.insertion_tableau.rows, len(bottom), "insertion"),
        (pair.recording_tableau.rows, len(top), "recording"),
    ):
        if any(
            type(value) is not int or value < 1 or value > limit
            for row in rows
            for value in row
        ):
            raise OperationDomainValidationError(
                location=("pair", where),
                code="algebraic_combinatorics.rsk_content",
                message=f"{where} tableau entries do not address the declared alphabet",
            )


def _inverse(pair: BiwordRSKPair) -> Biword:
    _validate_pair(pair)
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
    return _inverse(pair)


def inverse_matrix(
    pair: BiwordRSKPair, rows: tuple[str, ...], columns: tuple[str, ...]
) -> NonnegativeIntegerMatrix:
    _validate_pair(pair)
    if pair.source_kind != "MATRIX":
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_source_kind",
            message="matrix inverse requires a MATRIX pair",
        )
    if (
        type(rows) is not tuple
        or type(columns) is not tuple
        or rows != pair.top_alphabet
        or columns != pair.bottom_alphabet
    ):
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_axes",
            message="matrix axes must equal the pair alphabets",
        )
    b = _inverse(pair)
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
