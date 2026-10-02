"""Native/public centralizer admission, exact result decoding and composition."""

import json
from fractions import Fraction

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.canonical_forms import centralizer_basis
from jacobian.math.matrices.canonical_forms._models import CentralizerResult
from jacobian.math.matrices.values import rational_matrix_from_fractions


@pytest.mark.parametrize(
    ("size", "regime"),
    [(3, "jordan"), (16, "jordan"), (17, "scalar"), (32, "diagonal")],
)
def test_new_centralizer_regimes_roundtrip_and_consume_basis(
    size: int, regime: str
) -> None:
    source = rational_matrix_from_fractions(
        tuple(
            tuple(
                Fraction(
                    10**255
                    if regime == "jordan" and j == i + 1
                    else i
                    if regime == "diagonal" and i == j
                    else 7
                    if regime == "scalar" and i == j
                    else 0
                )
                for j in range(size)
            )
            for i in range(size)
        )
    )
    native = centralizer_basis(source)
    catalog = Catalog.open()
    output = invoke_operation(
        "matrix.centralizer.compute",
        {"matrix": source.model_dump(mode="json")},
        catalog,
    ).output
    restored = CentralizerResult.model_validate_json(json.dumps(output))
    assert restored == native
    assert restored.dimension == (size**2 if regime == "scalar" else size)
    # The producer's exact canonical matrix crosses JSON unchanged into a real
    # matrix consumer, even when the source order exceeds the old order16 cap.
    basis_wire = output["basis"][0]
    consumed = invoke_operation(
        "matrix.rank.compute", {"matrix": basis_wire}, catalog
    ).output
    assert consumed["rank"] == sum(
        value.num != 0 for row in restored.basis[0].entries for value in row
    )


@pytest.mark.parametrize(
    ("size", "regime", "code"),
    [
        (23, "scalar", "matrix.centralizer.output"),
        (17, "jordan", "matrix.centralizer.work"),
        (3, "height", "matrix.centralizer.source_height"),
    ],
)
def test_native_and_dispatch_keep_resource_refusals(
    size: int, regime: str, code: str
) -> None:
    source = rational_matrix_from_fractions(
        tuple(
            tuple(
                Fraction(
                    10**256
                    if regime == "height" and i == j
                    else int(i == j)
                    if regime == "scalar"
                    else int(j == i + 1)
                )
                for j in range(size)
            )
            for i in range(size)
        )
    )
    for run in (
        lambda: centralizer_basis(source),
        lambda: invoke_operation(
            "matrix.centralizer.compute",
            {"matrix": source.model_dump(mode="json")},
            Catalog.open(),
        ),
    ):
        with pytest.raises(OperationResourceAdmissionError) as exc:
            run()
        assert exc.value.errors()[0]["type"] == code
