from itertools import permutations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge import (
    FiniteGroupConjugacyProfile,
    FiniteGroupConjugacyProfileRequest,
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyRequest,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    finite_group_gauge_holonomy,
    finite_group_holonomy_conjugacy_profile,
)
from jacobian.math.groups._table_models import (
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table


def _s3():
    elements = tuple(permutations(range(3)))

    def compose(first, second):
        return tuple(second[first[i]] for i in range(3))

    index = {element: i for i, element in enumerate(elements)}
    table = tuple(tuple(index[compose(a, b)] for b in elements) for a in elements)
    identity = index[(0, 1, 2)]
    group = construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=table, identity=identity)
    ).group
    return group, elements, index


def _loop(group, element):
    field = FiniteGroupGaugeField(
        lattice=GaugeLattice(
            vertices=("v",),
            edges=(GaugeEdge(edge_id="e", tail="v", head="v"),),
        ),
        group=group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="e", value=FiniteGroupTableElement(group=group, index=element)
            ),
        ),
    )
    path = OrientedGaugePath(steps=(GaugePathStep(edge_id="e", forward=True),))
    return finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=path)
    )


def test_conjugacy_profile_matches_independent_s3_orbit_and_roundtrips():
    group, elements, index = _s3()
    transposition = index[(1, 0, 2)]
    loop = _loop(group, transposition)
    profile = finite_group_holonomy_conjugacy_profile(
        FiniteGroupConjugacyProfileRequest(field=loop.field, path=loop.path)
    )

    def compose(first, second):
        return tuple(second[first[i]] for i in range(3))

    target = elements[transposition]
    expected = tuple(
        sorted(
            {
                index[compose(compose(g, target), tuple(g.index(i) for i in range(3)))]
                for g in elements
            }
        )
    )
    assert profile.conjugate_indices == expected
    assert profile.class_size == 3
    assert profile.class_representative_index == expected[0]
    other_transposition = index[(0, 2, 1)]
    conjugate_profile = finite_group_holonomy_conjugacy_profile(
        FiniteGroupConjugacyProfileRequest(
            field=_loop(group, other_transposition).field,
            path=_loop(group, other_transposition).path,
        )
    )
    assert conjugate_profile.conjugate_indices == profile.conjugate_indices
    assert (
        FiniteGroupConjugacyProfile.model_validate_json(profile.model_dump_json())
        == profile
    )


def test_identity_and_three_cycle_have_their_exact_conjugacy_sizes():
    group, _, index = _s3()
    identity = finite_group_holonomy_conjugacy_profile(
        FiniteGroupConjugacyProfileRequest(
            field=_loop(group, index[(0, 1, 2)]).field,
            path=_loop(group, index[(0, 1, 2)]).path,
        )
    )
    three_cycle = finite_group_holonomy_conjugacy_profile(
        FiniteGroupConjugacyProfileRequest(
            field=_loop(group, index[(1, 2, 0)]).field,
            path=_loop(group, index[(1, 2, 0)]).path,
        )
    )
    assert identity.class_size == 1
    assert three_cycle.class_size == 2


def test_profile_deserialization_is_structural():
    group, _, index = _s3()
    loop = _loop(group, index[(1, 0, 2)])
    exact = finite_group_holonomy_conjugacy_profile(
        FiniteGroupConjugacyProfileRequest(field=loop.field, path=loop.path)
    )
    with pytest.raises(ValueError, match="not canonical"):
        FiniteGroupConjugacyProfile.model_validate(
            {
                **exact.model_dump(),
                "conjugate_indices": [group.identity],
                "class_representative_index": 9,
                "class_size": -1,
            }
        )


def test_conjugacy_profile_rejects_open_path():
    group, _, index = _s3()
    loop = _loop(group, index[(1, 0, 2)])
    open_path = OrientedGaugePath(steps=(GaugePathStep(edge_id="e", forward=True),))
    field = FiniteGroupGaugeField(
        lattice=GaugeLattice(
            vertices=("v", "w"),
            edges=(GaugeEdge(edge_id="e", tail="v", head="w"),),
        ),
        group=group,
        edge_values=loop.field.edge_values,
    )
    with pytest.raises(OperationDomainValidationError, match="closed based loop"):
        finite_group_holonomy_conjugacy_profile(
            FiniteGroupConjugacyProfileRequest(field=field, path=open_path)
        )


def test_forged_request_nested_values_are_domain_rejected():
    with pytest.raises(OperationDomainValidationError, match="requires a finite-group"):
        finite_group_holonomy_conjugacy_profile(
            FiniteGroupConjugacyProfileRequest.model_construct(field=None, path=None)
        )


def test_conjugacy_profile_is_published_as_one_exact_operation():
    from jacobian.catalog.catalog import Catalog

    tool = Catalog.open().operation("lattice_gauge.holonomy.conjugacy_profile.compute")
    assert tool.result_type is FiniteGroupConjugacyProfile
