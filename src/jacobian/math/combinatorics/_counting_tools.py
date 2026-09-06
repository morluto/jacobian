"""Immutable declarations for elementary counting operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics import operations as native
from jacobian.math.combinatorics._counting_models import (
    IntegerListRequest,
    SparseCountingPairRequest,
)
from jacobian.math.combinatorics._models import (
    IntegerResult,
    NonnegativeIntegerRequest,
)


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=value)


def factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.factorial(request.n))


def double_factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.double_factorial(request.n))


def derangements(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.derangement_number(request.n))


def binomial(request: SparseCountingPairRequest) -> IntegerResult:
    return _integer_result(native.canonical_binomial(request.n, request.k))


def multinomial(request: IntegerListRequest) -> IntegerResult:
    values = request.values
    return _integer_result(native.multinomial(values))


def permutations(request: SparseCountingPairRequest) -> IntegerResult:
    return _integer_result(native.canonical_permutations(request.n, request.k))


def catalan(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.catalan_number(request.n))


def motzkin(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.motzkin_number(request.n))


def central_binomial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.central_binomial(request.n))


def compositions(request: SparseCountingPairRequest) -> IntegerResult:
    return _integer_result(native.canonical_compositions(request.n, request.k))


COUNTING_OPERATIONS = (
    MathTool(
        operation_id="combinatorics.compute.factorial",
        title="Compute factorial",
        description="Compute n factorial exactly.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=factorial,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="factorial_5", description="Compute 5 factorial.", input={"n": 5}
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.double_factorial",
        title="Compute double factorial",
        description="Compute n double factorial exactly.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=double_factorial,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="double_factorial_7",
                description="Compute 7 double factorial.",
                input={"n": 7},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.derangements",
        title="Count derangements",
        description="Count fixed-point-free permutations of n labeled objects.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=derangements,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="derangements_4",
                description="Count derangements of 4 objects.",
                input={"n": 4},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.binomial",
        title="Compute binomial coefficient",
        description="Compute the exact integer binomial coefficient n choose k, counting "
        "k-element subsets of an n-element set.",
        request_type=SparseCountingPairRequest,
        result_type=IntegerResult,
        run=binomial,
        tags=("combinatorics", "counting", "number-theory"),
        examples=(
            OperationExample(
                name="binomial_5_choose_2",
                description="Compute 5 choose 2.",
                input={"n": 5, "k": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.multinomial",
        title="Compute multinomial coefficient",
        description="Count arrangements with the supplied nonnegative part sizes.",
        request_type=IntegerListRequest,
        result_type=IntegerResult,
        run=multinomial,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="multinomial_2_1_2",
                description="Compute a multinomial coefficient for parts 2, 1, and 2.",
                input={"values": ["2", "1", "2"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.permutations",
        title="Count partial permutations",
        description="Count ordered selections of k objects from n.",
        request_type=SparseCountingPairRequest,
        result_type=IntegerResult,
        run=permutations,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="permutations_5_2",
                description="Count ordered selections of 2 from 5.",
                input={"n": 5, "k": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.catalan",
        title="Compute Catalan number",
        description="Compute the nth Catalan number.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=catalan,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="catalan_4",
                description="Compute the fourth Catalan number.",
                input={"n": 4},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.motzkin",
        title="Compute Motzkin number",
        description="Compute the nth Motzkin path count.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=motzkin,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="motzkin_5",
                description="Compute the fifth Motzkin number.",
                input={"n": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.central_binomial",
        title="Compute central binomial coefficient",
        description="Compute binomial(2n,n) exactly.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerResult,
        run=central_binomial,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="central_binomial_4",
                description="Compute the central binomial coefficient for n=4.",
                input={"n": 4},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.compute.compositions",
        title="Count positive compositions",
        description="Count ordered positive-part compositions of n into k parts.",
        request_type=SparseCountingPairRequest,
        result_type=IntegerResult,
        run=compositions,
        tags=("combinatorics", "counting"),
        examples=(
            OperationExample(
                name="compositions_5_2",
                description="Count positive compositions of 5 into 2 parts.",
                input={"n": 5, "k": 2},
            ),
        ),
    ),
)
