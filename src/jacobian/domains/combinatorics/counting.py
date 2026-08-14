"""Counting-owned exact combinatorics operations."""

from jacobian.contracts.combinatorics import (
    BinomialRequest,
    IntegerListRequest,
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.combinatorics._support import (
    combinatorics_operation,
)
from jacobian.domains.combinatorics.operations import (
    binomial,
    catalan,
    central_binomial,
    compositions,
    derangements,
    double_factorial,
    factorial,
    motzkin,
    multinomial,
    permutations,
)

COUNTING_OPERATIONS = (
    combinatorics_operation(
        "combinatorics.compute.factorial",
        "Compute factorial",
        "Compute n factorial exactly.",
        NonnegativeIntegerRequest,
        IntegerResult,
        factorial,
        "combinatorics",
        "counting",
        examples=(example("factorial_5", "Compute 5 factorial.", {"n": 5}),),
    ),
    combinatorics_operation(
        "combinatorics.compute.double_factorial",
        "Compute double factorial",
        "Compute n double factorial exactly.",
        NonnegativeIntegerRequest,
        IntegerResult,
        double_factorial,
        "combinatorics",
        "counting",
        examples=(
            example("double_factorial_7", "Compute 7 double factorial.", {"n": 7}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.derangements",
        "Count derangements",
        "Count fixed-point-free permutations of n labeled objects.",
        NonnegativeIntegerRequest,
        IntegerResult,
        derangements,
        "combinatorics",
        "counting",
        examples=(
            example("derangements_4", "Count derangements of 4 objects.", {"n": 4}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.binomial",
        "Compute binomial coefficient",
        "Compute the exact integer binomial coefficient n choose k, counting "
        "k-element subsets of an n-element set.",
        BinomialRequest,
        IntegerResult,
        binomial,
        "combinatorics",
        "counting",
        "number-theory",
        examples=(
            example("binomial_5_choose_2", "Compute 5 choose 2.", {"n": 5, "k": 2}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.multinomial",
        "Compute multinomial coefficient",
        "Count arrangements with the supplied nonnegative part sizes.",
        IntegerListRequest,
        IntegerResult,
        multinomial,
        "combinatorics",
        "counting",
        examples=(
            example(
                "multinomial_2_1_2",
                "Compute a multinomial coefficient for parts 2, 1, and 2.",
                {"values": ["2", "1", "2"]},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.permutations",
        "Count partial permutations",
        "Count ordered selections of k objects from n.",
        NonnegativePairRequest,
        IntegerResult,
        permutations,
        "combinatorics",
        "counting",
        examples=(
            example(
                "permutations_5_2",
                "Count ordered selections of 2 from 5.",
                {"n": 5, "k": 2},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.catalan",
        "Compute Catalan number",
        "Compute the nth Catalan number.",
        NonnegativeIntegerRequest,
        IntegerResult,
        catalan,
        "combinatorics",
        "counting",
        examples=(
            example("catalan_4", "Compute the fourth Catalan number.", {"n": 4}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.motzkin",
        "Compute Motzkin number",
        "Compute the nth Motzkin path count.",
        NonnegativeIntegerRequest,
        IntegerResult,
        motzkin,
        "combinatorics",
        "counting",
        examples=(example("motzkin_5", "Compute the fifth Motzkin number.", {"n": 5}),),
    ),
    combinatorics_operation(
        "combinatorics.compute.central_binomial",
        "Compute central binomial coefficient",
        "Compute binomial(2n,n) exactly.",
        NonnegativeIntegerRequest,
        IntegerResult,
        central_binomial,
        "combinatorics",
        "counting",
        examples=(
            example(
                "central_binomial_4",
                "Compute the central binomial coefficient for n=4.",
                {"n": 4},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.compositions",
        "Count positive compositions",
        "Count ordered positive-part compositions of n into k parts.",
        NonnegativePairRequest,
        IntegerResult,
        compositions,
        "combinatorics",
        "counting",
        examples=(
            example(
                "compositions_5_2",
                "Count positive compositions of 5 into 2 parts.",
                {"n": 5, "k": 2},
            ),
        ),
    ),
)
