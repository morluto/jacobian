"""Cross-owner public-dispatch contracts for modular-form q-expansions."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_dispatch_returns_the_canonical_typed_q_expansion() -> None:
    result = invoke_operation(
        "modular_form.level_one.named_q_expansion.compute",
        {"form": "DELTA", "truncation_order": 4},
        Catalog.open(),
    )
    assert result.output is not None
    assert result.output["q_expansion"]["variable"] == "q"
    assert result.output["q_expansion"]["coefficients"] == [
        {"num": "0", "den": "1"},
        {"num": "1", "den": "1"},
        {"num": "-24", "den": "1"},
        {"num": "252", "den": "1"},
    ]


def test_dispatch_output_series_feeds_formal_series_power_unchanged() -> None:
    named = invoke_operation(
        "modular_form.level_one.named_q_expansion.compute",
        {"form": "E4", "truncation_order": 3},
        Catalog.open(),
    )
    assert named.output is not None
    squared = invoke_operation(
        "formal_series.rational.power.compute",
        {"series": named.output["q_expansion"], "exponent": 2},
        Catalog.open(),
    )
    assert squared.output is not None
    assert squared.output["result"]["coefficients"] == [
        {"num": "1", "den": "1"},
        {"num": "480", "den": "1"},
        {"num": "61920", "den": "1"},
    ]


def test_hecke_matrix_dispatch_returns_labeled_exact_basis_action() -> None:
    result = invoke_operation(
        "modular_form.hecke_matrix.compute",
        {
            "space": {
                "group": "GAMMA0",
                "level": 2,
                "weight": 4,
                "kind": "M",
                "character": "TRIVIAL",
                "coefficient_domain": "QQ",
            },
            "index": 3,
        },
        Catalog.open(),
    )
    assert result.output is not None
    assert result.output["row_labels"] == ["A2^2", "E4"]
    assert result.output["column_labels"] == ["A2^2", "E4"]
    assert result.output["entries"] == [
        [{"num": "28", "den": "1"}, {"num": "0", "den": "1"}],
        [{"num": "0", "den": "1"}, {"num": "28", "den": "1"}],
    ]
