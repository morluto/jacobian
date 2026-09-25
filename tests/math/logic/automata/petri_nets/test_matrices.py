"""Exact matrix projections of finite Petri nets."""

from jacobian.math.logic.automata import petri_nets
from jacobian.math.logic.automata.petri_nets._models import (
    PetriNetMatricesRequest,
)
from jacobian.math.logic.automata.petri_nets._tools import (
    compute_petri_net_matrices,
)
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def test_matrix_projection_preserves_axes_and_weighted_arc_data() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        place_ids=("stock", "finished"),
        transition_ids=("assemble", "inspect"),
        pre=((2, 0), (0, 1)),
        post=((0, 0), (3, 1)),
    )

    result = petri_nets.petri_net_matrices(net)

    assert result.net == net
    assert result.pre.entries == ((2, 0), (0, 1))
    assert result.post.entries == ((0, 0), (3, 1))
    assert result.incidence.entries == ((-2, 0), (3, 0))
    assert (result.pre.row_count, result.pre.column_count) == (2, 2)
    assert (result.post.row_count, result.post.column_count) == (2, 2)
    assert (result.incidence.row_count, result.incidence.column_count) == (2, 2)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_matrix_projection_preserves_degenerate_shape() -> None:
    net = PetriNet(
        place_count=0,
        transition_count=3,
        pre=(),
        post=(),
    )

    result = petri_nets.petri_net_matrices(net)

    for matrix in (result.pre, result.post, result.incidence):
        assert (matrix.row_count, matrix.column_count, matrix.entries) == (0, 3, ())
        assert matrix.model_validate_json(matrix.model_dump_json()) == matrix


def test_matrix_projection_is_published_in_manifest() -> None:
    from jacobian.math.logic.automata.petri_nets._tools import TOOLS

    operation = next(
        tool for tool in TOOLS if tool.operation_id == "petri_net.matrices.compute"
    )
    request = PetriNetMatricesRequest.model_validate(
        {
            "net": {
                "place_count": 1,
                "transition_count": 1,
                "pre": [[1]],
                "post": [[2]],
            }
        }
    )

    assert compute_petri_net_matrices(request).incidence.entries == ((1,),)
    assert operation.examples[0].input["net"]["pre"] == [[2], [0]]
