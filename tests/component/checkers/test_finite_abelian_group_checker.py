from __future__ import annotations

from tests.component.checkers.exact_domain_checker_support import _request

from jacobian_checkers.finite_abelian_groups import (
    check_finite_abelian_group_exact_factorization,
)


def test_finite_group_checker_rejects_normalized_duplicate_factor_entries() -> None:
    request = _request(
        "finite_abelian_group.exact_factorization.compute",
        "finite-abelian-group.exact-factorization.stdlib-replay",
        {"moduli": [4], "left": [[0], [4]], "right": [[0]]},
        {
            "semantics_version": "finite-abelian-group-factorization.v1",
            "moduli": [4],
            "normalized_left": [[0], [0]],
            "normalized_right": [[0]],
            "group_order": 4,
            "pair_count": 2,
            "distinct_sum_count": 1,
            "representation_histogram": [
                {"representation_count": 0, "element_count": 3},
                {"representation_count": 2, "element_count": 1},
            ],
            "is_exact_factorization": False,
            "first_missing": [1],
            "first_duplicate": {
                "element": [0],
                "left": [0],
                "right": [0],
                "other_left": [0],
                "other_right": [0],
            },
        },
    )

    decision = check_finite_abelian_group_exact_factorization(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_finite_group_checker_rejects_non_integer_candidate_scalars() -> None:
    request = _request(
        "finite_abelian_group.exact_factorization.compute",
        "finite-abelian-group.exact-factorization.stdlib-replay",
        {"moduli": [2], "left": [[0]], "right": [[0]]},
        {
            "semantics_version": "finite-abelian-group-factorization.v1",
            "moduli": [2],
            "normalized_left": [[0]],
            "normalized_right": [[0]],
            "group_order": 2,
            "pair_count": True,
            "distinct_sum_count": 1,
            "representation_histogram": [
                {"representation_count": 0, "element_count": 1},
                {"representation_count": 1, "element_count": 1},
            ],
            "is_exact_factorization": False,
            "first_missing": [1],
            "first_duplicate": None,
        },
    )

    decision = check_finite_abelian_group_exact_factorization(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
