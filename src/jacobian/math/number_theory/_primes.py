"""Prime-owned exact number-theory operations."""

from jacobian.catalog._examples import example
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
from jacobian.math.number_theory._support import (
    number_theory_operation,
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
    number_theory_operation(
        "integer.decide.prime",
        "Decide integer primality",
        "Decide whether one integer is prime.",
        PrimalityRequest,
        BooleanResult,
        decide_prime,
        "number-theory",
        "predicate",
        examples=(example("prime_17", "Check whether 17 is prime.", {"value": "17"}),),
    ),
    number_theory_operation(
        "integer.compute.next_prime",
        "Compute next prime",
        "Compute the least prime strictly greater than n.",
        NonnegativeIntegerRequest,
        IntegerValue,
        compute_next_prime,
        "number-theory",
        "prime",
        examples=(
            example("next_prime_14", "Find the next prime after 14.", {"n": 14}),
        ),
    ),
    number_theory_operation(
        "integer.compute.previous_prime",
        "Compute previous prime",
        "Compute the greatest prime strictly below n.",
        PreviousPrimeRequest,
        IntegerValue,
        compute_previous_prime,
        "number-theory",
        "prime",
        examples=(
            example(
                "previous_prime_14", "Find the previous prime before 14.", {"n": 14}
            ),
        ),
    ),
    number_theory_operation(
        "integer.compute.prime_count",
        "Count primes through n",
        "Count primes not exceeding one nonnegative integer.",
        NonnegativeIntegerRequest,
        IntegerValue,
        compute_prime_count,
        "number-theory",
        "prime",
        examples=(example("prime_count_20", "Count primes through 20.", {"n": 20}),),
    ),
    number_theory_operation(
        "integer.compute.nth_prime",
        "Compute nth prime",
        "Compute the nth prime using one-based indexing.",
        PositiveIntegerRequest,
        IntegerValue,
        compute_nth_prime,
        "number-theory",
        "prime",
        examples=(example("nth_prime_6", "Compute the sixth prime.", {"n": 6}),),
    ),
    number_theory_operation(
        "integer.compute.primorial",
        "Compute primorial",
        "Compute the product of the first n primes.",
        PrimorialRequest,
        PrimorialResult,
        compute_primorial,
        "number-theory",
        "prime",
        examples=(
            example(
                "primorial_5", "Compute the product of the first five primes.", {"n": 5}
            ),
        ),
    ),
    number_theory_operation(
        "integer.compute.euler_totient",
        "Compute Euler totient",
        "Count residues coprime to one positive integer.",
        PositiveIntegerRequest,
        IntegerValue,
        compute_euler_totient,
        "number-theory",
        "arithmetic-function",
        examples=(example("totient_12", "Count residues coprime to 12.", {"n": 12}),),
    ),
    number_theory_operation(
        "integer.compute.mobius",
        "Compute Mobius value",
        "Compute the Mobius arithmetic function of one positive integer.",
        PositiveIntegerRequest,
        IntegerValue,
        compute_mobius,
        "number-theory",
        "arithmetic-function",
        examples=(example("mobius_30", "Compute the Mobius value of 30.", {"n": 30}),),
    ),
)
