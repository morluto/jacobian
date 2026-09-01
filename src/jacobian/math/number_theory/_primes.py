"""Prime-owned exact number-theory operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._integer_models import (
    BooleanResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._prime_models import (
    PreviousPrimeRequest,
    PrimalityRequest,
    PrimorialRequest,
    PrimorialResult,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue
from jacobian.math.number_theory.operations import (
    euler_totient,
    is_prime,
    mobius,
    next_prime,
    nth_prime,
    previous_prime,
    prime_count,
    primorial,
)


def decide_prime(request: PrimalityRequest) -> BooleanResult:
    return is_prime(request.value)


def compute_next_prime(request: NonnegativeIntegerRequest) -> IntegerValue:
    return next_prime(request.n)


def compute_previous_prime(request: PreviousPrimeRequest) -> IntegerValue:
    return previous_prime(request.n)


def compute_prime_count(request: NonnegativeIntegerRequest) -> IntegerValue:
    return prime_count(request.n)


def compute_nth_prime(request: PositiveIntegerRequest) -> IntegerValue:
    return nth_prime(request.n)


def compute_primorial(request: PrimorialRequest) -> PrimorialResult:
    return primorial(request.n)


def compute_euler_totient(request: PositiveIntegerRequest) -> IntegerValue:
    return euler_totient(request.n)


def compute_mobius(request: PositiveIntegerRequest) -> IntegerValue:
    return mobius(request.n)


PRIME_OPERATIONS = (
    MathTool(
        operation_id="integer.decide.prime",
        title="Decide integer primality",
        description="Decide whether one integer is prime.",
        request_type=PrimalityRequest,
        result_type=BooleanResult,
        run=decide_prime,
        tags=("number-theory", "predicate"),
        examples=(
            OperationExample(
                name="prime_17",
                description="Check whether 17 is prime.",
                input={"value": "17"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.next_prime",
        title="Compute next prime",
        description="Compute the least prime strictly greater than n.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerValue,
        run=compute_next_prime,
        tags=("number-theory", "prime"),
        examples=(
            OperationExample(
                name="next_prime_14",
                description="Find the next prime after 14.",
                input={"n": 14},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.previous_prime",
        title="Compute previous prime",
        description="Compute the greatest prime strictly below n.",
        request_type=PreviousPrimeRequest,
        result_type=IntegerValue,
        run=compute_previous_prime,
        tags=("number-theory", "prime"),
        examples=(
            OperationExample(
                name="previous_prime_14",
                description="Find the previous prime before 14.",
                input={"n": 14},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.prime_count",
        title="Count primes through n",
        description="Count primes not exceeding one nonnegative integer.",
        request_type=NonnegativeIntegerRequest,
        result_type=IntegerValue,
        run=compute_prime_count,
        tags=("number-theory", "prime"),
        examples=(
            OperationExample(
                name="prime_count_20",
                description="Count primes through 20.",
                input={"n": 20},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.nth_prime",
        title="Compute nth prime",
        description="Compute the nth prime using one-based indexing.",
        request_type=PositiveIntegerRequest,
        result_type=IntegerValue,
        run=compute_nth_prime,
        tags=("number-theory", "prime"),
        examples=(
            OperationExample(
                name="nth_prime_6",
                description="Compute the sixth prime.",
                input={"n": 6},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.primorial",
        title="Compute primorial",
        description="Compute the product of the first n primes.",
        request_type=PrimorialRequest,
        result_type=PrimorialResult,
        run=compute_primorial,
        tags=("number-theory", "prime"),
        examples=(
            OperationExample(
                name="primorial_5",
                description="Compute the product of the first five primes.",
                input={"n": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.euler_totient",
        title="Compute Euler totient",
        description="Count residues coprime to one positive integer.",
        request_type=PositiveIntegerRequest,
        result_type=IntegerValue,
        run=compute_euler_totient,
        tags=("number-theory", "arithmetic-function"),
        examples=(
            OperationExample(
                name="totient_12",
                description="Count residues coprime to 12.",
                input={"n": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.mobius",
        title="Compute Mobius value",
        description="Compute the Mobius arithmetic function of one positive integer.",
        request_type=PositiveIntegerRequest,
        result_type=IntegerValue,
        run=compute_mobius,
        tags=("number-theory", "arithmetic-function"),
        examples=(
            OperationExample(
                name="mobius_30",
                description="Compute the Mobius value of 30.",
                input={"n": 30},
            ),
        ),
    ),
)
