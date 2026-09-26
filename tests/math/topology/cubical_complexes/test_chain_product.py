from collections import Counter

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes import (
    CubicalChainProductRequest,
    CubicalChainTerm,
    CubicalChainValue,
    chain_product,
    operations,
)
from jacobian.math.topology.cubical_complexes._models import CubicalCell
from jacobian.math.topology.cubical_complexes._tools import TOOLS


def _chain(ambient_dimension, degree, terms):
    return CubicalChainValue(
        ambient_dimension=ambient_dimension,
        degree=degree,
        terms=tuple(
            CubicalChainTerm(
                cell=CubicalCell(intervals=intervals), coefficient=coefficient
            )
            for intervals, coefficient in terms
        ),
    )


def _direct_boundary(intervals):
    """Independent cubical endpoint formula used only as a test oracle."""
    active_axes = [i for i, (lower, upper) in enumerate(intervals) if lower < upper]
    boundary_terms = []
    for position, axis in enumerate(active_axes):
        lower, upper = intervals[axis]
        orientation = 1 if position % 2 == 0 else -1
        for endpoint, coefficient in ((upper, orientation), (lower, -orientation)):
            face = list(intervals)
            face[axis] = (endpoint, endpoint)
            boundary_terms.append((tuple(face), coefficient))
    return boundary_terms


def _boundary_of_chain(chain):
    result = Counter()
    for term in chain.terms:
        for face, coefficient in _direct_boundary(term.cell.intervals):
            result[face] += term.coefficient * coefficient
    return Counter({cell: coefficient for cell, coefficient in result.items() if coefficient})


def _direct_product(left_terms, right_terms):
    left_terms = tuple(left_terms)
    right_terms = tuple(right_terms)
    result = Counter()
    for left_cell, left_coefficient in left_terms:
        for right_cell, right_coefficient in right_terms:
            result[left_cell + right_cell] += left_coefficient * right_coefficient
    return Counter({cell: coefficient for cell, coefficient in result.items() if coefficient})


@pytest.mark.parametrize(
    ("left_degree", "left_cell"),
    [
        (1, ((0, 1), (5, 5))),
        (2, ((0, 1), (5, 5), (-2, -1))),
    ],
)
def test_chain_product_obeys_graded_boundary_identity(left_degree, left_cell):
    left = _chain(2 if left_degree == 1 else 3, left_degree, [(left_cell, -2)])
    right_cell = ((3, 3), (-4, -3))
    right = _chain(2, 1, [(right_cell, 3)])
    product = chain_product(CubicalChainProductRequest(left=left, right=right))

    assert product.degree == left_degree + 1
    assert product.ambient_dimension == left.ambient_dimension + right.ambient_dimension
    assert [
        (term.cell.intervals, term.coefficient) for term in product.terms
    ] == [(left_cell + right_cell, -6)]

    lhs = _boundary_of_chain(product)
    left_boundary = _boundary_of_chain(left)
    right_boundary = _boundary_of_chain(right)
    rhs = _direct_product(
        ((cell, coefficient) for cell, coefficient in left_boundary.items()),
        ((term.cell.intervals, term.coefficient) for term in right.terms),
    )
    right_part = _direct_product(
        ((term.cell.intervals, term.coefficient) for term in left.terms),
        ((cell, coefficient) for cell, coefficient in right_boundary.items()),
    )
    sign = -1 if left_degree % 2 else 1
    rhs.update({cell: sign * coefficient for cell, coefficient in right_part.items()})
    rhs = Counter({cell: coefficient for cell, coefficient in rhs.items() if coefficient})
    assert lhs == rhs


def test_zero_chain_keeps_its_degree_and_product_ambient_context():
    zero = _chain(2, 1, [])
    point = _chain(1, 0, [(((7, 7),), 5)])

    result = chain_product(CubicalChainProductRequest(left=zero, right=point))

    assert result.ambient_dimension == 3
    assert result.degree == 1
    assert result.terms == ()


def test_chain_product_result_can_be_reused_with_a_large_exact_coefficient():
    coefficient = 10**127
    left = _chain(1, 1, [(((0, 1),), coefficient)])
    right = _chain(1, 0, [(((5, 5),), 1)])
    product = chain_product(CubicalChainProductRequest(left=left, right=right))

    reusable = chain_product(
        CubicalChainProductRequest(left=product, right=right)
    )
    assert reusable.terms[0].coefficient == coefficient
    payload = product.model_dump(mode="json")
    assert payload["terms"][0]["coefficient"] == str(coefficient)
    assert payload["terms"][0]["cell"]["intervals"] == [["0", "1"], ["5", "5"]]
    assert CubicalChainValue.model_validate_json(encode_strict_json(payload)) == product
    assert reusable.degree == 1
    assert reusable.ambient_dimension == 3


def test_chain_product_readmits_untrusted_nested_chain_values():
    malformed = CubicalChainValue.model_construct(
        ambient_dimension=1,
        degree=0,
        terms=(
            CubicalChainTerm.model_construct(
                cell=CubicalCell.model_construct(intervals=((0, 0), (1, 1))),
                coefficient=0,
            ),
        ),
    )
    request = CubicalChainProductRequest.model_construct(left=malformed, right=None)
    with pytest.raises(OperationDomainValidationError):
        chain_product(request)


def test_chain_product_preflights_term_growth_before_constructing_cells(monkeypatch):
    left = _chain(
        2,
        1,
        [(((index, index + 1), (0, 0)), 1) for index in range(64)],
    )
    right = _chain(
        2,
        1,
        [(((index + 100, index + 101), (0, 0)), 1) for index in range(33)],
    )
    request = CubicalChainProductRequest(left=left, right=right)
    monkeypatch.setattr(
        operations,
        "CubicalCell",
        lambda **_: pytest.fail("product cells must not be constructed before admission"),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        operations.chain_product(request)
    assert (
        error.value.errors()[0]["type"]
        == "cubical_complex.chain_product_term_budget"
    )


def test_chain_product_preflights_coefficient_growth_before_constructing_cells(
    monkeypatch,
):
    left = _chain(1, 1, [(((0, 1),), 10**100)])
    right = _chain(1, 1, [(((2, 3),), 10**100)])
    request = CubicalChainProductRequest(left=left, right=right)
    monkeypatch.setattr(
        operations,
        "CubicalCell",
        lambda **_: pytest.fail("product cells must not be constructed before admission"),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        operations.chain_product(request)
    assert (
        error.value.errors()[0]["type"]
        == "cubical_complex.chain_product_coefficient_budget"
    )


def test_chain_product_preflights_result_bytes_before_constructing_cells(monkeypatch):
    base = 10**63
    left = _chain(
        5,
        0,
        [
            (tuple((coordinate, coordinate) for coordinate in (base + i, 0, 0, 0, 0)), 1)
            for i in range(64)
        ],
    )
    right = _chain(
        5,
        0,
        [
            (tuple((coordinate, coordinate) for coordinate in (base + 100 + i, 0, 0, 0, 0)), 1)
            for i in range(32)
        ],
    )
    request = CubicalChainProductRequest(left=left, right=right)
    monkeypatch.setattr(
        operations,
        "CubicalCell",
        lambda **_: pytest.fail("product cells must not be constructed before admission"),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        operations.chain_product(request)
    assert (
        error.value.errors()[0]["type"]
        == "cubical_complex.chain_product_result_size"
    )


def test_chain_product_accepts_exact_term_bound_and_catalog_round_trips():
    left = _chain(
        2,
        1,
        [(((index, index + 1), (0, 0)), 1) for index in range(64)],
    )
    right = _chain(
        2,
        1,
        [(((index + 100, index + 101), (0, 0)), -1) for index in range(32)],
    )
    request = CubicalChainProductRequest(left=left, right=right)
    result = chain_product(request)
    assert len(result.terms) == 2048

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.cubical_chain.external_product.compute"
    )
    wire_request = request.model_dump(mode="json")
    typed_request = tool.request_type.model_validate_json(
        encode_strict_json(wire_request), strict=True
    )
    public_value = tool.run(typed_request)
    restored = CubicalChainValue.model_validate_json(encode_strict_json(public_value.model_dump(mode="json")))
    assert restored == result

    example_request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    example_result = tool.run(example_request)
    assert example_result.terms[0].cell.intervals == ((0, 1), (3, 4))
    assert example_result.terms[0].coefficient == -6
