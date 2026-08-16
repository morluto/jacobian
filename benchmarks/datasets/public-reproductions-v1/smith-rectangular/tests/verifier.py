import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _gcd(left: int, right: int) -> int:
    left, right = abs(left), abs(right)
    while right:
        left, right = right, left % right
    return left or 1


def _swap_rows(matrix: list[list[int]], i: int, j: int) -> None:
    matrix[i], matrix[j] = matrix[j], matrix[i]


def _swap_cols(matrix: list[list[int]], i: int, j: int) -> None:
    for row in matrix:
        row[i], row[j] = row[j], row[i]


def _add_row(
    matrix: list[list[int]], cols: int, source: int, target: int, scale: int
) -> None:
    if scale:
        for col in range(cols):
            matrix[target][col] += scale * matrix[source][col]


def _add_col(matrix: list[list[int]], source: int, target: int, scale: int) -> None:
    if scale:
        for row in matrix:
            row[target] += scale * row[source]


def _smallest_pivot(
    matrix: list[list[int]], rank: int, rows: int, cols: int
) -> tuple[int, int] | None:
    pivot = None
    for i in range(rank, rows):
        for j in range(rank, cols):
            if matrix[i][j] != 0 and (
                pivot is None or abs(matrix[i][j]) < abs(matrix[pivot[0]][pivot[1]])
            ):
                pivot = (i, j)
    return pivot


def _reduce_pivot_column(
    matrix: list[list[int]], rank: int, rows: int, cols: int
) -> bool:
    changed = False
    for i in range(rank + 1, rows):
        if matrix[i][rank]:
            quotient, remainder = divmod(matrix[i][rank], matrix[rank][rank])
            _add_row(matrix, cols, rank, i, -quotient)
            if remainder:
                _swap_rows(matrix, rank, i)
                changed = True
    return changed


def _reduce_pivot_row(matrix: list[list[int]], rank: int, rows: int, cols: int) -> bool:
    changed = False
    for j in range(rank + 1, cols):
        if matrix[rank][j]:
            quotient, remainder = divmod(matrix[rank][j], matrix[rank][rank])
            _add_col(matrix, rank, j, -quotient)
            if remainder:
                _swap_cols(matrix, rank, j)
                changed = True
    return changed


def _mix_off_pivot(matrix: list[list[int]], rank: int, rows: int, cols: int) -> bool:
    for i in range(rank + 1, rows):
        for j in range(rank + 1, cols):
            if matrix[i][j] % matrix[rank][rank]:
                _add_row(matrix, cols, i, rank, 1)
                return True
    return False


def _reduce_around_pivot(
    matrix: list[list[int]], rank: int, rows: int, cols: int
) -> None:
    changed = True
    while changed:
        changed = False
        if _reduce_pivot_column(matrix, rank, rows, cols):
            changed = True
        if _reduce_pivot_row(matrix, rank, rows, cols):
            changed = True
        if _mix_off_pivot(matrix, rank, rows, cols):
            changed = True


def _clear_off_pivot(matrix: list[list[int]], rank: int, rows: int, cols: int) -> None:
    for i in range(rank + 1, rows):
        _add_row(matrix, cols, rank, i, -(matrix[i][rank] // matrix[rank][rank]))
    for j in range(rank + 1, cols):
        _add_col(matrix, rank, j, -(matrix[rank][j] // matrix[rank][rank]))


def _divisibility_chain(factors: list[int]) -> list[int]:
    cleaned = [factor for factor in factors if factor]
    for index in range(len(cleaned) - 1):
        divisor = _gcd(cleaned[index], cleaned[index + 1])
        if divisor != cleaned[index]:
            cleaned[index + 1] = cleaned[index] * cleaned[index + 1] // divisor
            cleaned[index] = divisor
    return cleaned


def _smith(entries: list[list[int]]) -> tuple[int, list[int]]:
    matrix = [row[:] for row in entries]
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    rank = 0
    factors: list[int] = []
    while rank < min(rows, cols):
        pivot = _smallest_pivot(matrix, rank, rows, cols)
        if pivot is None:
            break
        _swap_rows(matrix, rank, pivot[0])
        _swap_cols(matrix, rank, pivot[1])
        _reduce_around_pivot(matrix, rank, rows, cols)
        _clear_off_pivot(matrix, rank, rows, cols)
        value = abs(matrix[rank][rank])
        if not value:
            break
        factors.append(value)
        rank += 1
    return rank, _divisibility_chain(factors)


def _typed_matrix(x: object) -> list[list[int]] | None:
    matrix = x.get("matrix") if isinstance(x, dict) else None
    if not isinstance(matrix, dict):
        return None
    rows = matrix.get("entries")
    row_count = matrix.get("row_count")
    column_count = matrix.get("column_count")
    if (
        type(row_count) is not int
        or type(column_count) is not int
        or not isinstance(rows, list)
        or len(rows) != row_count
        or any(not isinstance(row, list) or len(row) != column_count for row in rows)
    ):
        return None
    try:
        return [[int(value) for value in row] for row in rows]
    except (TypeError, ValueError):
        return None


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict) or set(result) != {"rank", "invariant_factors"}:
        return False
    if type(result.get("rank")) is not int:
        return False
    factors = result.get("invariant_factors")
    if not isinstance(factors, list) or any(
        type(value) is not int for value in factors
    ):
        return False
    entries = _typed_matrix(x)
    if entries is None:
        return False
    rank, invariants = _smith(entries)
    return result["rank"] == rank and factors == invariants


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = _math(s, x) if protocol_ok else False
    reward = float(protocol_ok and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
