import json
from collections import defaultdict
from itertools import combinations
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


def _collect_faces(
    vertices: list[str], facets: list[list[str]]
) -> dict[int, list[tuple[int, ...]]]:
    index = {name: i for i, name in enumerate(vertices)}
    faces: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    seen: set[tuple[int, ...]] = set()
    for facet in facets:
        ids = tuple(sorted(index[name] for name in facet))
        for dim in range(len(ids)):
            for combo in combinations(ids, dim + 1):
                if combo not in seen:
                    seen.add(combo)
                    faces[dim].append(combo)
    for dim in faces:
        faces[dim].sort()
    return faces


def _boundary_matrix(
    faces: dict[int, list[tuple[int, ...]]], dim: int, convention: str
) -> list[list[int]]:
    if dim == 0:
        if convention == "REDUCED":
            return [[1] * len(faces[0])]
        return []
    if dim not in faces or not faces[dim]:
        return []
    lower = faces.get(dim - 1, [])
    col_index = {simplex: i for i, simplex in enumerate(lower)}
    matrix = [[0] * len(faces[dim]) for _ in range(len(lower))]
    for col, simplex in enumerate(faces[dim]):
        for i, _vertex in enumerate(simplex):
            face = simplex[:i] + simplex[i + 1 :]
            matrix[col_index[face]][col] += 1 - 2 * (i % 2)
    return matrix


def _homology(vertices: list[str], facets: list[list[str]], convention: str):
    faces = _collect_faces(vertices, facets)
    max_dim = max(faces, default=-1)
    if convention == "REDUCED":
        faces[-1] = [()]
    ranks: dict[int, int] = {}
    factors: dict[int, list[int]] = {}
    start = -1 if convention == "REDUCED" else 0
    for dim in range(start, max_dim + 1):
        matrix = _boundary_matrix(faces, dim, convention)
        if not matrix or not matrix[0]:
            ranks[dim] = 0
            factors[dim] = []
            continue
        rank, invariants = _smith(matrix)
        ranks[dim] = rank
        factors[dim] = invariants
    top = max_dim if max_dim >= 0 else 0
    free = []
    torsion = []
    for degree in range(top + 1):
        cell_count = len(faces.get(degree, []))
        betti = cell_count - ranks.get(degree, 0) - ranks.get(degree + 1, 0)
        free.append(betti)
        torsion.append(
            [str(value) for value in factors.get(degree + 1, []) if value > 1]
        )
    return free, torsion


def _typed_presentation(x: object) -> tuple[list[str], list[list[str]], str] | None:
    presentation = x.get("presentation") if isinstance(x, dict) else None
    convention = x.get("convention") if isinstance(x, dict) else None
    if not isinstance(presentation, dict) or convention not in {"REDUCED", "UNREDUCED"}:
        return None
    vertices = presentation.get("vertices")
    facets = presentation.get("facets")
    if not isinstance(vertices, list) or not isinstance(facets, list):
        return None
    if any(not isinstance(name, str) for name in vertices):
        return None
    if any(
        not isinstance(facet, list) or any(not isinstance(name, str) for name in facet)
        for facet in facets
    ):
        return None
    return vertices, facets, convention


def _normalize_torsion(submitted_torsion: object) -> list[list[str]] | None:
    if not isinstance(submitted_torsion, list):
        return None
    normalized = []
    for row in submitted_torsion:
        if not isinstance(row, list):
            return None
        normalized_row = []
        for value in row:
            if type(value) is int:
                normalized_row.append(str(value))
            elif isinstance(value, str):
                normalized_row.append(value)
            else:
                return None
        normalized.append(normalized_row)
    return normalized


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict):
        return False
    parsed = _typed_presentation(x)
    if parsed is None:
        return False
    vertices, facets, convention = parsed
    free, torsion = _homology(vertices, facets, convention)
    submitted_free = result.get("free_ranks")
    if not isinstance(submitted_free, list):
        return False
    if any(type(value) is not int for value in submitted_free):
        return False
    normalized = _normalize_torsion(result.get("torsion"))
    if normalized is None:
        return False
    return submitted_free == free and normalized == torsion


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
