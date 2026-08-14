"""Prime-owned exact number-theory operations."""

from jacobian.contracts.number_theory import (
    BooleanResult,
    IntegerValueRequest,
    IntegerValueResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.number_theory._support import (
    number_theory_operation,
)
from jacobian.domains.number_theory.prime_operations import (
    compute_euler_totient,
    compute_mobius,
    compute_next_prime,
    compute_nth_prime,
    compute_previous_prime,
    compute_prime_count,
    compute_primorial,
    decide_prime,
)

PRIME_OPERATIONS = (
    number_theory_operation(
        "integer.decide.prime",
        "Decide integer primality",
        "Decide whether one integer is prime.",
        IntegerValueRequest,
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
        IntegerValueResult,
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
        NonnegativeIntegerRequest,
        IntegerValueResult,
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
        IntegerValueResult,
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
        IntegerValueResult,
        compute_nth_prime,
        "number-theory",
        "prime",
        examples=(example("nth_prime_6", "Compute the sixth prime.", {"n": 6}),),
    ),
    number_theory_operation(
        "integer.compute.primorial",
        "Compute primorial",
        "Compute the product of the first n primes.",
        NonnegativeIntegerRequest,
        IntegerValueResult,
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
        IntegerValueResult,
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
        IntegerValueResult,
        compute_mobius,
        "number-theory",
        "arithmetic-function",
        examples=(example("mobius_30", "Compute the Mobius value of 30.", {"n": 30}),),
    ),
)
