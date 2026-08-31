"""Killable FLINT worker for invariant-form graph-lattice HNF."""

from __future__ import annotations

import json
import sys

from flint import fmpz_mat


def main() -> None:
    payload = json.load(sys.stdin)
    coefficient_count = payload["coefficient_count"]
    constraints = payload["constraints"]
    constraint_count = len(constraints)
    graph = fmpz_mat(
        [
            [
                *(
                    constraints[constraint][coordinate]
                    for constraint in range(constraint_count)
                ),
                *(int(coordinate == column) for column in range(coefficient_count)),
            ]
            for coordinate in range(coefficient_count)
        ]
    )
    graph_hnf = graph.hnf()
    primitive_kernel = [
        [
            int(graph_hnf[row, constraint_count + column])
            for column in range(coefficient_count)
        ]
        for row in range(coefficient_count)
        if all(graph_hnf[row, constraint] == 0 for constraint in range(constraint_count))
    ]
    json.dump(
        {
            "primitive_kernel": primitive_kernel,
            "constraint_rank": coefficient_count - len(primitive_kernel),
        },
        sys.stdout,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
