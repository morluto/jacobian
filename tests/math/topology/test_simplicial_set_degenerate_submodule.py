import pytest
from sympy import Matrix, zeros

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.values import CoefficientRing
from jacobian.math.topology.simplicial_sets import chains as chains_module
from jacobian.math.topology.simplicial_sets.degenerate_submodule import (
    DegenerateSubmoduleRequest,
    degenerate_submodule,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _matrix(rows: tuple[tuple[int, ...], ...], row_count: int, column_count: int):
    return Matrix(
        row_count,
        column_count,
        lambda row, column: rows[row][column],
    )


def _independent_degeneracy_matrix(source, degree: int):
    target_rank = len(source.sets[degree])
    columns = [
        target
        for degeneracy in source.degeneracy_maps[degree - 1]
        for target in degeneracy
    ]
    return Matrix(
        target_rank,
        len(columns),
        lambda row, column: int(columns[column] == row),
    )


def _independent_boundary(source, degree: int, prime: int | None = None):
    rows = len(source.sets[degree - 1])
    columns = len(source.sets[degree])

    def entry(row: int, column: int) -> int:
        value = sum(
            (1 if face_index % 2 == 0 else -1)
            for face_index, face in enumerate(source.face_maps[degree - 1])
            if face[column] == row
        )
        return value if prime is None else value % prime

    return Matrix(
        rows,
        columns,
        entry,
    )


@pytest.mark.parametrize(
    ("dimension", "max_degree"),
    [(0, 1), (1, 2), (2, 3), (1, 4)],
)
def test_degenerate_submodule_matches_independent_exact_span_and_chain_oracle(
    dimension: int, max_degree: int
):
    source = standard_simplex(dimension, max_degree)
    result = degenerate_submodule(DegenerateSubmoduleRequest(simplicial_set=source))

    assert result.degenerate_complex.basis_sizes == tuple(
        len(indices) for indices in result.degenerate_basis_indices
    )
    for degree in range(1, max_degree + 1):
        generated = _independent_degeneracy_matrix(source, degree)
        inclusion = _matrix(
            result.inclusion_matrices[degree],
            len(source.sets[degree]),
            len(result.degenerate_basis_indices[degree]),
        )
        assert inclusion.rank() == generated.rank()
        assert inclusion.row_join(generated).rank() == generated.rank()

    for degree in range(1, max_degree + 1):
        ambient_boundary = _independent_boundary(source, degree)
        source_inclusion = _matrix(
            result.inclusion_matrices[degree],
            len(source.sets[degree]),
            len(result.degenerate_basis_indices[degree]),
        )
        target_inclusion = _matrix(
            result.inclusion_matrices[degree - 1],
            len(source.sets[degree - 1]),
            len(result.degenerate_basis_indices[degree - 1]),
        )
        restricted_boundary = _matrix(
            result.degenerate_complex.differential_matrices[degree - 1],
            len(result.degenerate_basis_indices[degree - 1]),
            len(result.degenerate_basis_indices[degree]),
        )
        assert ambient_boundary * source_inclusion == (
            target_inclusion * restricted_boundary
        )
        if degree > 1:
            prior = _matrix(
                result.degenerate_complex.differential_matrices[degree - 2],
                len(result.degenerate_basis_indices[degree - 2]),
                len(result.degenerate_basis_indices[degree - 1]),
            )
            assert prior * restricted_boundary == zeros(
                len(result.degenerate_basis_indices[degree - 2]),
                len(result.degenerate_basis_indices[degree]),
            )

    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_delta_one_has_expected_degenerate_basis_and_restricted_boundaries():
    result = degenerate_submodule(
        DegenerateSubmoduleRequest(simplicial_set=standard_simplex(1, 2))
    )

    assert result.degenerate_basis_indices == ((), (0, 2), (0, 1, 2, 3))
    assert result.degenerate_complex.basis_sizes == (0, 2, 4)
    assert result.degenerate_complex.differential_matrices == (
        (),
        ((1, 1, 0, 0), (0, 0, 1, 1)),
    )


@pytest.mark.parametrize(
    ("coefficient_ring", "prime"),
    [
        (CoefficientRing.RATIONAL, None),
        (CoefficientRing.PRIME_FIELD, 2),
    ],
)
def test_degenerate_submodule_retains_requested_scalar_context(
    coefficient_ring: CoefficientRing, prime: int | None
):
    result = degenerate_submodule(
        DegenerateSubmoduleRequest(
            simplicial_set=standard_simplex(1, 2),
            coefficient_ring=coefficient_ring,
            prime=prime,
        )
    )

    assert result.unnormalized_chains.chain_complex.coefficient_ring is coefficient_ring
    assert result.degenerate_complex.coefficient_ring is coefficient_ring
    assert result.degenerate_complex.prime == prime
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_degenerate_submodule_handles_empty_carrier_and_zero_submodule():
    checked = from_tables(0, ((),), (), ())
    assert checked.simplicial_set is not None
    empty = checked.simplicial_set
    result = degenerate_submodule(DegenerateSubmoduleRequest(simplicial_set=empty))

    assert result.degenerate_basis_indices == ((),)
    assert result.inclusion_matrices == ((),)
    assert result.degenerate_complex.basis_sizes == (0,)
    assert result.degenerate_complex.differential_matrices == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_degenerate_submodule_wire_round_trip_is_stable_json():
    result = degenerate_submodule(
        DegenerateSubmoduleRequest(simplicial_set=standard_simplex(1, 2))
    )
    payload = result.model_dump_json()

    assert type(result).model_validate_json(payload) == result


def test_degenerate_submodule_checks_source_once_and_constructs_without_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    source = standard_simplex(1, 2)
    original_from_tables = chains_module.from_tables
    validations = 0

    def count_source_checks(*args, **kwargs):
        nonlocal validations
        validations += 1
        return original_from_tables(*args, **kwargs)

    def reject_public_chain_reentry(*args, **kwargs):
        pytest.fail("degenerate construction re-entered public chain admission")

    monkeypatch.setattr(chains_module, "from_tables", count_source_checks)
    monkeypatch.setattr(
        chains_module, "unnormalized_chains", reject_public_chain_reentry
    )

    result = degenerate_submodule(DegenerateSubmoduleRequest(simplicial_set=source))

    assert validations == 1
    assert result.unnormalized_chains.simplicial_set == source


def test_degenerate_submodule_admits_derived_output_before_chain_matrices(
    monkeypatch: pytest.MonkeyPatch,
):
    source = standard_simplex(1, 2)
    sizes = tuple(len(level) for level in source.sets)
    ambient_bound = chains_module._estimate_output_bytes(source, sizes)
    original_from_tables = chains_module.from_tables
    validations = 0
    matrix_construction_started = False

    def count_source_checks(*args, **kwargs):
        nonlocal validations
        validations += 1
        return original_from_tables(*args, **kwargs)

    def reject_chain_matrix_construction(*args, **kwargs):
        nonlocal matrix_construction_started
        matrix_construction_started = True
        pytest.fail("chain matrices were built before derived output admission")

    monkeypatch.setattr(chains_module, "from_tables", count_source_checks)
    monkeypatch.setattr(
        chains_module,
        "MAX_UNNORMALIZED_CHAIN_OUTPUT_BYTES",
        ambient_bound,
    )
    monkeypatch.setattr(
        chains_module,
        "_unnormalized_chains_from_checked_source",
        reject_chain_matrix_construction,
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        degenerate_submodule(DegenerateSubmoduleRequest(simplicial_set=source))

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.degenerate_submodule_output_budget_exceeded"
    )
    assert validations == 1
    assert not matrix_construction_started
