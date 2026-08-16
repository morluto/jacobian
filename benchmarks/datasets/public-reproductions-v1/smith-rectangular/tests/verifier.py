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


def _smith(entries: list[list[int]]) -> tuple[int, list[int]]:
    matrix = [row[:] for row in entries]
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0

    def swap_rows(i: int, j: int) -> None:
        matrix[i], matrix[j] = matrix[j], matrix[i]

    def swap_cols(i: int, j: int) -> None:
        for row in matrix:
            row[i], row[j] = row[j], row[i]

    def add_row(source: int, target: int, scale: int) -> None:
        if scale:
            for col in range(cols):
                matrix[target][col] += scale * matrix[source][col]

    def add_col(source: int, target: int, scale: int) -> None:
        if scale:
            for row in matrix:
                row[target] += scale * row[source]

    rank = 0
    factors: list[int] = []
    while rank < min(rows, cols):
        pivot = None
        for i in range(rank, rows):
            for j in range(rank, cols):
                if matrix[i][j] != 0 and (
                    pivot is None or abs(matrix[i][j]) < abs(matrix[pivot[0]][pivot[1]])
                ):
                    pivot = (i, j)
        if pivot is None:
            break
        swap_rows(rank, pivot[0])
        swap_cols(rank, pivot[1])
        changed = True
        while changed:
            changed = False
            for i in range(rank + 1, rows):
                if matrix[i][rank]:
                    quotient, remainder = divmod(matrix[i][rank], matrix[rank][rank])
                    add_row(rank, i, -quotient)
                    if remainder:
                        swap_rows(rank, i)
                        changed = True
            for j in range(rank + 1, cols):
                if matrix[rank][j]:
                    quotient, remainder = divmod(matrix[rank][j], matrix[rank][rank])
                    add_col(rank, j, -quotient)
                    if remainder:
                        swap_cols(rank, j)
                        changed = True
            for i in range(rank + 1, rows):
                for j in range(rank + 1, cols):
                    if matrix[i][j] % matrix[rank][rank]:
                        add_row(i, rank, 1)
                        changed = True
                        break
                if changed:
                    break
        for i in range(rank + 1, rows):
            add_row(rank, i, -(matrix[i][rank] // matrix[rank][rank]))
        for j in range(rank + 1, cols):
            add_col(rank, j, -(matrix[rank][j] // matrix[rank][rank]))
        value = abs(matrix[rank][rank])
        if not value:
            break
        factors.append(value)
        rank += 1
    cleaned = [factor for factor in factors if factor]
    for index in range(len(cleaned) - 1):
        divisor = _gcd(cleaned[index], cleaned[index + 1])
        if divisor != cleaned[index]:
            cleaned[index + 1] = cleaned[index] * cleaned[index + 1] // divisor
            cleaned[index] = divisor
    return rank, cleaned


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
    matrix = x.get("matrix") if isinstance(x, dict) else None
    if not isinstance(matrix, dict):
        return False
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
        return False
    try:
        entries = [[int(value) for value in row] for row in rows]
    except (TypeError, ValueError):
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
