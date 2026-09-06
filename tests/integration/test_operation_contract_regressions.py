"""Catalog-to-kernel regressions for composable operation contracts."""

import pytest
from tests.support.rationals import rational_payload as q

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices._models import IntegerLattice


def _truth_table(values: list[int]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational(num=str(v), den="1") for v in values)


def _lattice(ambient: int, basis: list[list[int]]) -> IntegerLattice:
    return IntegerLattice.model_validate(
        {
            "ambient_dimension": ambient,
            "basis": {"entries": [[str(v) for v in row] for row in basis]},
        }
    )


@pytest.mark.parametrize("n", [0, 10, 11, 12])
def test_walsh_conventions_have_exact_affine_relationship(n: int) -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    values = [x.bit_count() % 2 for x in range(1 << n)]
    catalog = Catalog.open()
    signs = invoke_operation(
        "boolean.fourier.walsh_transform.compute", {"truth_table": values}, catalog
    ).output
    raw = invoke_operation(
        "boolean.fourier_spectrum.compute",
        {"truth_table": [c.model_dump(mode="json") for c in _truth_table(values)]},
        catalog,
    ).output
    assert signs["variable_count"] == raw["variable_count"] == n
    assert [int(v) for v in signs["spectrum"]] == [
        ((1 << n) if i == 0 else 0) - 2 * int(v["num"])
        for i, v in enumerate(raw["spectrum"])
    ]


@pytest.mark.parametrize("count", [257, 688, 1100])
def test_forced_cover_beyond_recursive_depth(count: int) -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    items = [f"p{i:04}" for i in range(count)]
    payload = {
        "instance": {
            "primary_items": items,
            "secondary_items": [],
            "rows": [
                {"row_id": f"r{i:04}", "items": [item]} for i, item in enumerate(items)
            ],
        },
        "search_node_limit": count + 1,
    }
    result = invoke_operation(
        "combinatorics.generalized_exact_cover.find", payload, Catalog.open()
    ).output
    assert result["status"] == "FOUND"
    assert result["selected_row_ids"] == [f"r{i:04}" for i in range(count)]
    assert all(item["multiplicity"] == 1 for item in result["item_multiplicities"])
    payload["search_node_limit"] = count
    assert (
        invoke_operation(
            "combinatorics.generalized_exact_cover.find", payload, Catalog.open()
        ).output["status"]
        == "UNKNOWN"
    )


def test_exact_cover_prices_primary_mask_width_even_with_few_rows() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.dispatch import invoke_operation

    items = [f"p{i:04}" for i in range(4096)]
    instance = {
        "primary_items": items,
        "secondary_items": [],
        "rows": [{"row_id": "all", "items": items}],
    }
    catalog = Catalog.open()
    assert (
        invoke_operation(
            "combinatorics.generalized_exact_cover.find",
            {"instance": instance, "search_node_limit": 6250},
            catalog,
        ).output["status"]
        == "FOUND"
    )
    with pytest.raises(OperationDomainValidationError, match="item-scan"):
        invoke_operation(
            "combinatorics.generalized_exact_cover.find",
            {"instance": instance, "search_node_limit": 6251},
            catalog,
        )


def test_center_retains_parent_and_rejects_noncanonical_residue() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.finite_dim_algebras._models import CenterResult

    algebra = {"dimension": 1, "field_order": 3, "multiplication": [[[0]]]}
    output = invoke_operation(
        "algebra.center.compute", {"algebra": algebra}, Catalog.open()
    ).output
    assert output["algebra"] == algebra
    parsed = CenterResult.model_validate(output)
    assert parsed.algebra.field_order == 3
    output["center_basis"] = [[3]]
    with pytest.raises(ValueError, match="canonical residues"):
        CenterResult.model_validate(output)


@pytest.mark.parametrize("embedding", [[[3]], [[-2]]])
def test_sublattice_index_rejects_false_inclusion(embedding: list[list[int]]) -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    with pytest.raises(OperationDomainValidationError, match="E @ parent"):
        invoke_operation(
            "lattice.sublattice_index.compute",
            {
                "sublattice": _lattice(1, [[2]]).model_dump(mode="json"),
                "parent": _lattice(1, [[1]]).model_dump(mode="json"),
                "embedding": {"entries": [[str(x) for x in row] for row in embedding]},
            },
            Catalog.open(),
        )


def test_symbolic_characteristic_support_collects_raw_products() -> None:
    import sympy

    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    q = [[4, -1, -1, -1], [1, 4, -1, 1], [1, 1, 4, -1], [1, -1, 1, 4]]
    c = [
        [
            q[i][j - 4] if i < 4 <= j else q[j][i - 4] if j < 4 <= i else 0
            for j in range(8)
        ]
        for i in range(8)
    ]
    assert sympy.Matrix(q) * sympy.Matrix(q).T == 19 * sympy.eye(4)

    def entry(value: int, diagonal: bool) -> dict[str, object]:
        return {
            "variables": ["t"],
            "numerator": {
                "terms": [
                    {
                        "coefficient": {
                            "num": str(1 if diagonal else value),
                            "den": "1",
                        },
                        "exponents": [int(diagonal)],
                    }
                ]
                if diagonal or value
                else []
            },
            "denominator": {
                "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
            },
        }

    result = invoke_operation(
        "matrix.symbolic.characteristic_polynomial.compute",
        {
            "matrix": {
                "variables": ["t"],
                "entries": [
                    [entry(c[i][j], i == j) for j in range(8)] for i in range(8)
                ],
            }
        },
        Catalog.open(),
    ).output
    t, lam = sympy.symbols("t lambda")
    coefficients = [
        sum(
            sympy.Rational(term["coefficient"]["num"], term["coefficient"]["den"])
            * t ** term["exponents"][0]
            for term in coefficient["numerator"]["terms"]
        )
        for coefficient in result["coefficients_descending"]
    ]
    assert sympy.Poly.from_list(coefficients, lam) == sympy.Poly(
        ((lam - t) ** 2 - 19) ** 4, lam
    )


def test_disconnected_lp_returns_source_coordinate_optimum() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    n = 32
    program = {
        "variables": [f"x{i}" for i in range(n)],
        "objective": [q(1 + i % 2) for i in range(n)],
        "coefficients": [[q(int(i // 2 == j)) for i in range(n)] for j in range(16)],
        "rhs": [q(1)] * 16,
    }
    result = invoke_operation(
        "optimization.linear.rational_optimum.compute",
        {"program": program},
        Catalog.open(),
    ).output
    assert result["status"] == "OPTIMAL"
    assert result["primal_objective"] == q(16)
    assert result["primal_candidate"] == [q(1 - i % 2) for i in range(n)]
    assert result["dual_candidate"] == [q(1)] * 16
    assert result["dual_slacks"] == [q(i % 2) for i in range(n)]


def test_quotient_noncongruence_is_domain_rejection() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.dispatch import invoke_operation

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(
            "universal_algebra.quotient.compute",
            {
                "algebra": {
                    "carrier": ["0", "1", "2"],
                    "operations": [{"operation_id": "succ", "arity": 1}],
                    "tables": [[1, 2, 0]],
                },
                "partition": [[0, 1], [2]],
            },
            Catalog.open(),
        )
    assert (
        caught.value.errors()[0]["type"] == "universal_algebra.partition_not_congruence"
    )
