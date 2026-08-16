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


def _homology(vertices: list[str], facets: list[list[str]], convention: str):
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
    max_dim = max(faces, default=-1)
    if convention == "REDUCED":
        faces[-1] = [()]

    def boundary(dim: int) -> list[list[int]]:
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
                matrix[col_index[face]][col] += -1 if i % 2 else 1
        return matrix

    ranks: dict[int, int] = {}
    factors: dict[int, list[int]] = {}
    dims = list(range(-1 if convention == "REDUCED" else 0, max_dim + 1))
    for dim in dims:
        matrix = boundary(dim)
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
    for degree in range(0, top + 1):
        cell_count = len(faces.get(degree, []))
        betti = cell_count - ranks.get(degree, 0) - ranks.get(degree + 1, 0)
        free.append(betti)
        torsion.append(
            [str(value) for value in factors.get(degree + 1, []) if value > 1]
        )
    return free, torsion


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict):
        return False
    presentation = x.get("presentation")
    convention = x.get("convention")
    if not isinstance(presentation, dict) or convention not in {"REDUCED", "UNREDUCED"}:
        return False
    vertices = presentation.get("vertices")
    facets = presentation.get("facets")
    if not isinstance(vertices, list) or not isinstance(facets, list):
        return False
    if any(not isinstance(name, str) for name in vertices):
        return False
    if any(
        not isinstance(facet, list) or any(not isinstance(name, str) for name in facet)
        for facet in facets
    ):
        return False
    free, torsion = _homology(vertices, facets, convention)
    submitted_free = result.get("free_ranks")
    submitted_torsion = result.get("torsion")
    if not isinstance(submitted_free, list) or not isinstance(submitted_torsion, list):
        return False
    if any(type(value) is not int for value in submitted_free):
        return False
    normalized = []
    for row in submitted_torsion:
        if not isinstance(row, list):
            return False
        normalized_row = []
        for value in row:
            if type(value) is int:
                normalized_row.append(str(value))
            elif isinstance(value, str):
                normalized_row.append(value)
            else:
                return False
        normalized.append(normalized_row)
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
