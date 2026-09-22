from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    construct,
    fiber,
    positive_grading,
)
from jacobian.math.affine_semigroups.semigroup_models import AffineFiberRequest
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS as AFFINE_TOOLS
from jacobian.math.combinatorics.algebraic.biword import (
    BiwordRSKPair,
    NonnegativeIntegerMatrix,
)
from jacobian.math.combinatorics.algebraic.biword_ops import (
    inverse_biword,
    inverse_matrix,
    matrix_biword,
)
from jacobian.math.combinatorics.matroids.delta.extra import BinarySymmetricMatrix
from jacobian.math.combinatorics.matroids.delta.extra_ops import dual, minor
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)


def _quadrant() -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=((1, 0), (0, 1)),
    )
    return construct(
        configuration,
        (CanonicalRational(num=1, den=1), CanonicalRational(num=1, den=1)),
    )


def test_positive_grading_is_complete_beyond_old_search_box() -> None:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=((10, -9), (-1, 1)),
    )
    result = positive_grading(configuration)
    assert result.positive is True
    assert tuple(value.as_fraction() for value in result.grading) == (
        Fraction(2),
        Fraction(19),
    )


def test_affine_fiber_rejects_target_before_recursion() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="candidate bound"):
        fiber(_quadrant(), (10**100, 0))


def test_affine_grading_accepts_negative_coordinate_witness() -> None:
    config = AffineConfiguration(
        row_labels=("x",), generator_labels=("a",), entries=((-4,),)
    )
    result = positive_grading(config)
    assert result.positive is True
    assert tuple(x.as_fraction() for x in result.grading) == (Fraction(-1, 4),)


def test_affine_catalog_preserves_resource_refusal() -> None:
    request = AffineFiberRequest(semigroup=_quadrant(), target=(10**100, 0))
    runner = cast(
        Callable[[AffineFiberRequest], object],
        next(
            tool.run
            for tool in AFFINE_TOOLS
            if tool.operation_id == "affine_semigroup.factorizations.compute"
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        runner(request)


def test_public_axis_schemas_match_parser_bounds() -> None:
    affine = {
        "row_labels": [str(i) for i in range(9)],
        "generator_labels": ["a"],
        "entries": [["1"] for _ in range(9)],
    }
    assert list(
        Draft202012Validator(AffineConfiguration.model_json_schema()).iter_errors(
            affine
        )
    )
    with pytest.raises(ValidationError):
        AffineConfiguration.model_validate_json(json.dumps(affine))

    binary = {
        "ground": [str(i) for i in range(13)],
        "entries": [[0] * 13 for _ in range(13)],
    }
    assert list(
        Draft202012Validator(BinarySymmetricMatrix.model_json_schema()).iter_errors(
            binary
        )
    )
    with pytest.raises(ValidationError):
        BinarySymmetricMatrix.model_validate_json(json.dumps(binary))

    matrix = {"row_labels": ["r"], "column_labels": ["c"], "entries": [["501"]]}
    assert list(
        Draft202012Validator(NonnegativeIntegerMatrix.model_json_schema()).iter_errors(
            matrix
        )
    )
    with pytest.raises(ValidationError):
        NonnegativeIntegerMatrix.model_validate_json(json.dumps(matrix))


def test_delta_minor_rejects_duplicate_native_axes_before_source_work() -> None:
    with pytest.raises(OperationDomainValidationError, match="sorted and distinct"):
        minor(_quadrant_delta(), delete=(0, 0))


def test_delta_resource_refusal_is_not_invalid_source() -> None:
    rows = tuple(
        sorted(tuple(i for i in range(14) if mask >> i & 1) for mask in range(1 << 14))
    )
    source = FiniteDeltaMatroid(ground=tuple(str(i) for i in range(14)), feasible=rows)
    with pytest.raises(OperationResourceAdmissionError):
        dual(source)


def test_forged_matrix_mass_is_rejected_before_expansion() -> None:
    forged = NonnegativeIntegerMatrix.model_construct(
        row_labels=("r",), column_labels=("c",), entries=((10**100,),)
    )
    with pytest.raises(OperationDomainValidationError, match="mass bounded"):
        matrix_biword(forged)


def _quadrant_delta() -> FiniteDeltaMatroid:
    rows = tuple(tuple(i for i in range(2) if mask >> i & 1) for mask in range(4))
    return FiniteDeltaMatroid(ground=("a", "b"), feasible=tuple(sorted(rows)))


def test_delta_minor_compacts_axes_and_uses_deletion_semantics() -> None:
    feasible = tuple(tuple(i for i in range(3) if mask >> i & 1) for mask in range(8))
    delta = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=tuple(sorted(feasible)))
    result = minor(delta, delete=(1,))
    assert result.ground == ("a", "c")
    assert result.feasible == ((), (0,), (0, 1), (1,))


def test_delta_extra_operations_reject_forged_non_delta_sources() -> None:
    forged = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=((), (0, 1, 2)))
    with pytest.raises(OperationDomainValidationError, match="not a delta-matroid"):
        dual(forged)


def test_binary_principal_minor_envelope_is_visible_and_preflighted() -> None:
    with pytest.raises(ValidationError, match="limited to 12"):
        BinarySymmetricMatrix(
            ground=tuple(str(i) for i in range(13)),
            entries=tuple(tuple(0 for _ in range(13)) for _ in range(13)),
        )
    forged = BinarySymmetricMatrix.model_construct(
        ground=tuple(str(i) for i in range(13)),
        entries=tuple(tuple(0 for _ in range(13)) for _ in range(13)),
    )
    from jacobian.math.combinatorics.matroids.delta.extra_ops import binary

    with pytest.raises(OperationDomainValidationError, match="limited to 12"):
        binary(forged)


def test_inverse_biword_rejects_forged_alphabet_pair() -> None:
    pair = BiwordRSKPair.model_construct(
        top_alphabet=(),
        bottom_alphabet=(),
        insertion_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        recording_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        shape=IntegerPartition.model_construct(parts=(1,)),
        source_kind="BIWORD",
        convention="ROW_INSERTION_RSK_V1",
    )
    with pytest.raises(OperationDomainValidationError, match="insertion entries"):
        inverse_biword(pair)


def test_inverse_rsk_reestablishes_semistandard_pair_membership() -> None:
    pair = BiwordRSKPair.model_construct(
        top_alphabet=("a", "b"),
        bottom_alphabet=("x", "y"),
        insertion_tableau=SemistandardYoungTableau.model_construct(rows=((1, 1),)),
        recording_tableau=SemistandardYoungTableau.model_construct(rows=((2, 1),)),
        shape=IntegerPartition.model_construct(parts=(2,)),
        source_kind="BIWORD",
        convention="ROW_INSERTION_RSK_V1",
    )
    with pytest.raises(OperationDomainValidationError, match="semistandard"):
        inverse_biword(pair)

    transported = BiwordRSKPair.model_validate_json(pair.model_dump_json())
    with pytest.raises(OperationDomainValidationError, match="semistandard"):
        inverse_biword(transported)

    matrix_pair = pair.model_copy(update={"source_kind": "MATRIX"})
    with pytest.raises(OperationDomainValidationError, match="semistandard"):
        inverse_matrix(matrix_pair, ("a", "b"), ("x", "y"))
