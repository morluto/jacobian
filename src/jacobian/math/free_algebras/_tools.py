"""Public declaration for exact free associative algebra multiplication."""

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdealDegreeComponentRequest,
    FreeAlgebraIdealDegreeComponentResult,
    FreeAlgebraIdealMembershipRequest,
    FreeAlgebraIdealMembershipResult,
    FreeAlgebraIdealPrefixRequest,
    FreeAlgebraIdealPrefixResult,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialAddRequest,
    FreeAlgebraPolynomialProductRequest,
    FreeAlgebraPolynomialProductResult,
    FreeAlgebraPolynomialSubstitutionRequest,
    FreeAlgebraQuotientProfileRequest,
    FreeAlgebraQuotientProfileResult,
    FreeAlgebraWordCompareResult,
    FreeAlgebraWordFactorsResult,
    FreeAlgebraWordOverlapResult,
    FreeAlgebraWordPairRequest,
    FreeAlgebraWordPairResult,
    FreeAlgebraWordPowerRequest,
    FreeAlgebraWordPowerResult,
    FreeAlgebraWordPrefixesResult,
    FreeAlgebraWordRequest,
    FreeAlgebraWordReverseResult,
    FreeAlgebraWordSubstitutionRequest,
    FreeAlgebraWordSubstitutionResult,
    FreeAlgebraWordSuffixesResult,
    GroebnerShirshovRequest,
    GroebnerShirshovResult,
    TruncatedFreeAlgebraQuotient,
    TruncatedFreeAlgebraQuotientRequest,
)
from jacobian.math.free_algebras.operations import (
    add,
    compare_words,
    concatenate_words,
    groebner_shirshov_through_degree,
    ideal_degree_component,
    ideal_generated_prefix,
    ideal_membership,
    multiply,
    power_word,
    quotient_normal_word_profile,
    reverse_word,
    substitute_polynomial,
    substitute_word,
    truncated_quotient_algebra,
    word_factors,
    word_overlaps,
    word_prefixes,
    word_suffixes,
)


def _run_multiply(
    request: FreeAlgebraPolynomialProductRequest,
) -> FreeAlgebraPolynomialProductResult:
    return multiply(request.left, request.right)


def _run_add(request: FreeAlgebraPolynomialAddRequest) -> FreeAlgebraPolynomial:
    return add(request.left, request.right)


def _run_word_concatenate(
    request: FreeAlgebraWordPairRequest,
) -> FreeAlgebraWordPairResult:
    return concatenate_words(request.left, request.right)


def _run_word_power(request: FreeAlgebraWordPowerRequest) -> FreeAlgebraWordPowerResult:
    return power_word(request.word, request.exponent)


def _run_word_reverse(request: FreeAlgebraWordRequest) -> FreeAlgebraWordReverseResult:
    return reverse_word(request.word)


def _run_word_prefixes(
    request: FreeAlgebraWordRequest,
) -> FreeAlgebraWordPrefixesResult:
    return word_prefixes(request.word)


def _run_word_suffixes(
    request: FreeAlgebraWordRequest,
) -> FreeAlgebraWordSuffixesResult:
    return word_suffixes(request.word)


def _run_word_compare(
    request: FreeAlgebraWordPairRequest,
) -> FreeAlgebraWordCompareResult:
    return compare_words(request.left, request.right)


def _run_word_factors(request: FreeAlgebraWordRequest) -> FreeAlgebraWordFactorsResult:
    return word_factors(request.word)


def _run_word_overlaps(
    request: FreeAlgebraWordPairRequest,
) -> FreeAlgebraWordOverlapResult:
    return word_overlaps(request.left, request.right)


def _run_word_substitute(
    request: FreeAlgebraWordSubstitutionRequest,
) -> FreeAlgebraWordSubstitutionResult:
    return substitute_word(request.substitution, request.word)


def _run_polynomial_substitute(
    request: FreeAlgebraPolynomialSubstitutionRequest,
) -> FreeAlgebraPolynomial:
    return substitute_polynomial(request.substitution, request.polynomial)


_EXAMPLE_ALPHABET = ["x", "y"]

# (y + x)(x - y) = yx - yy + xx - xy, with xy and yx distinct.
_EXAMPLE_LEFT = {
    "alphabet": _EXAMPLE_ALPHABET,
    "terms": [
        {"coefficient": {"num": "1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}
_EXAMPLE_RIGHT = {
    "alphabet": _EXAMPLE_ALPHABET,
    "terms": [
        {"coefficient": {"num": "-1", "den": "1"}, "word": ["y"]},
        {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
    ],
}
_ADD_EXAMPLE = {
    "left": {
        "alphabet": _EXAMPLE_ALPHABET,
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "word": ["y"]},
            {"coefficient": {"num": "2", "den": "1"}, "word": ["x"]},
        ],
    },
    "right": {
        "alphabet": _EXAMPLE_ALPHABET,
        "terms": [
            {"coefficient": {"num": "3", "den": "2"}, "word": ["x", "y"]},
            {"coefficient": {"num": "-2", "den": "1"}, "word": ["x"]},
        ],
    },
}


def _run_left_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "left":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.left_side",
            message="the left-ideal operation requires side='left'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


def _run_right_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "right":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.right_side",
            message="the right-ideal operation requires side='right'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


def _run_two_sided_prefix(
    request: FreeAlgebraIdealPrefixRequest,
) -> FreeAlgebraIdealPrefixResult:
    if request.ideal.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.two_sided_side",
            message="the two-sided ideal operation requires side='two-sided'",
        )
    return ideal_generated_prefix(request.ideal, request.degree)


def _run_degree_component(
    request: FreeAlgebraIdealDegreeComponentRequest,
) -> FreeAlgebraIdealDegreeComponentResult:
    return ideal_degree_component(request.ideal, request.degree)


def _run_membership(
    request: FreeAlgebraIdealMembershipRequest,
) -> FreeAlgebraIdealMembershipResult:
    return ideal_membership(request.ideal, request.polynomial)


def _run_truncated_quotient(
    request: TruncatedFreeAlgebraQuotientRequest,
) -> TruncatedFreeAlgebraQuotient:
    return truncated_quotient_algebra(request.ideal, request.degree)


TOOLS = (
    MathTool(
        operation_id="free_algebra.polynomial.multiply.compute",
        title="Multiply sparse noncommutative polynomials exactly",
        description=(
            "Compute the exact distributive concatenation product of two "
            "sparse QQ-linear polynomials over one shared finite ordered "
            "generator alphabet: (sum a_u u)(sum b_v v) = sum a_u b_v (uv). "
            "Like product words are collected and zero sums dropped, so the "
            "result is a canonical sparse free-algebra polynomial in "
            "descending degree-lexicographic order. The free algebra is "
            "noncommutative, so x*y and y*x stay distinct for distinct "
            "generators. Admission bounds each alphabet to 26 distinct "
            "letters, operand words to 32 letters, operand terms to 64, "
            "product term pairs and result terms to 4096, and predicted "
            "coefficient growth to 64 digits before expansion; the bounded "
            "term-pair and collected-like-word ledger is returned."
        ),
        request_type=FreeAlgebraPolynomialProductRequest,
        result_type=FreeAlgebraPolynomialProductResult,
        run=_run_multiply,
        tags=("algebra", "free-algebra", "noncommutative", "polynomial", "exact"),
        discovery_terms=(
            "free associative algebra",
            "noncommutative polynomial",
            "word multiplication",
            "distributive product",
            "sparse nc polynomial",
        ),
        examples=(
            OperationExample(
                name="xy_yx_distinct",
                description=(
                    "Multiply (y + x) by (x - y) over the alphabet [x, y] and "
                    "return the canonical product with its term-pair ledger; "
                    "the distinct words x*y and y*x must not be collected."
                ),
                input={"left": _EXAMPLE_LEFT, "right": _EXAMPLE_RIGHT},
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.polynomial.add.compute",
        title="Add sparse noncommutative polynomials exactly",
        description=(
            "Add two sparse QQ-linear polynomials bound to the identical "
            "ordered generator alphabet. Equal words have their exact "
            "rational coefficients combined, zero sums are omitted, and the "
            "result is returned as the canonical degree-lexicographically "
            "ordered polynomial; distinct words such as xy and yx stay "
            "distinct. Each operand is limited to 64 terms of word length at "
            "most 32. Admission bounds the 128-term worst-case support, "
            "coefficient growth, and a 2 MB serialized result before "
            "aggregation."
        ),
        request_type=FreeAlgebraPolynomialAddRequest,
        result_type=FreeAlgebraPolynomial,
        run=_run_add,
        tags=("algebra", "free-algebra", "noncommutative", "polynomial", "exact"),
        discovery_terms=(
            "free associative algebra addition",
            "noncommutative polynomial sum",
            "sparse nc polynomial addition",
        ),
        examples=(
            OperationExample(
                name="collect_and_cancel_like_words",
                description=(
                    "Add 2x+y and -2x+3/2 xy. The x terms cancel; xy and y "
                    "remain distinct words over the same ordered alphabet."
                ),
                input=_ADD_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.concatenate.compute",
        title="Concatenate free-algebra words",
        description="Concatenate two source words over one ordered alphabet. Output length is preflighted to 64; source ranges and exact degree addition are returned.",
        request_type=FreeAlgebraWordPairRequest,
        result_type=FreeAlgebraWordPairResult,
        run=_run_word_concatenate,
        tags=("free-algebra", "word", "concatenation", "exact"),
        discovery_terms=(
            "free word product",
            "word concatenation",
            "free monoid operation",
        ),
        examples=(
            OperationExample(
                name="unit_and_word",
                description="Concatenate the empty word with xy; both words must use the same ordered alphabet.",
                input={
                    "left": {"alphabet": ["x", "y"], "letters": []},
                    "right": {"alphabet": ["x", "y"], "letters": ["x", "y"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.power.compute",
        title="Raise a free word to a bounded power",
        description="Return a nonnegative concatenation power with exponent at most 64 and output length at most 64. Exponent zero returns the empty word, including for a nonempty base.",
        request_type=FreeAlgebraWordPowerRequest,
        result_type=FreeAlgebraWordPowerResult,
        run=_run_word_power,
        tags=("free-word", "power", "exact"),
        discovery_terms=(
            "free word power",
            "word concatenation power",
            "free monoid exponent",
        ),
        examples=(
            OperationExample(
                name="zero_power",
                description="Raise xy to exponent zero, producing the empty word; the exponent is a nonnegative integer.",
                input={
                    "word": {"alphabet": ["x", "y"], "letters": ["x", "y"]},
                    "exponent": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.reverse.compute",
        title="Reverse a free-algebra word",
        description="Reverse the ordered generator sequence while preserving its alphabet. Reversal is involutive and reverses product order.",
        request_type=FreeAlgebraWordRequest,
        result_type=FreeAlgebraWordReverseResult,
        run=_run_word_reverse,
        tags=("free-algebra", "word", "reverse", "exact"),
        discovery_terms=("reverse free word", "word reversal", "anti-automorphism"),
        examples=(
            OperationExample(
                name="reverse_xy",
                description="Reverse xy to yx; every letter must belong to the declared alphabet.",
                input={"word": {"alphabet": ["x", "y"], "letters": ["x", "y"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.prefixes.compute",
        title="Return all prefixes with reconstruction suffixes",
        description="Return every prefix of a source word, including empty and whole, with its complementary suffix so each split reconstructs the source word.",
        request_type=FreeAlgebraWordRequest,
        result_type=FreeAlgebraWordPrefixesResult,
        run=_run_word_prefixes,
        tags=("free-word", "prefixes", "exact"),
        discovery_terms=(
            "free word prefixes",
            "word prefix family",
            "prefix decomposition",
        ),
        examples=(
            OperationExample(
                name="prefixes_of_xy",
                description="Return the empty, x, and xy prefixes with complementary suffixes; source positions are all split boundaries.",
                input={"word": {"alphabet": ["x", "y"], "letters": ["x", "y"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.suffixes.compute",
        title="Return all suffixes with reconstruction prefixes",
        description="Return every suffix of a source word, including empty and whole, with its complementary prefix so each split reconstructs the source word.",
        request_type=FreeAlgebraWordRequest,
        result_type=FreeAlgebraWordSuffixesResult,
        run=_run_word_suffixes,
        tags=("free-word", "suffixes", "exact"),
        discovery_terms=(
            "free word suffixes",
            "word suffix family",
            "suffix decomposition",
        ),
        examples=(
            OperationExample(
                name="suffixes_of_xy",
                description="Return the xy, y, and empty suffixes with complementary prefixes; source positions are all split boundaries.",
                input={"word": {"alphabet": ["x", "y"], "letters": ["x", "y"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.order.compare",
        title="Compare free-algebra words",
        description="Compare words over the same ordered alphabet in degree-lexicographic order. Return degree comparison, first differing position and generator ranks, and the total comparison; host string ordering is not used.",
        request_type=FreeAlgebraWordPairRequest,
        result_type=FreeAlgebraWordCompareResult,
        run=_run_word_compare,
        tags=("free-word", "order", "degree-lexicographic", "exact"),
        discovery_terms=(
            "free word order",
            "degree lex word comparison",
            "monomial order compare",
        ),
        examples=(
            OperationExample(
                name="degree_before_lex",
                description="Compare y with xx, which is greater by degree; both words must use the same ordered alphabet.",
                input={
                    "left": {"alphabet": ["x", "y"], "letters": ["y"]},
                    "right": {"alphabet": ["x", "y"], "letters": ["x", "x"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.factors.compute",
        title="Enumerate contiguous free-word factors",
        description="Return each distinct contiguous factor once with every source start position, including the empty factor at every boundary. Preflight at most 561 occurrences and bounded aggregate factor-letter cells before enumeration.",
        request_type=FreeAlgebraWordRequest,
        result_type=FreeAlgebraWordFactorsResult,
        run=_run_word_factors,
        tags=("free-algebra", "word", "factors", "exact"),
        discovery_terms=(
            "free word factors",
            "contiguous subwords",
            "word prefixes suffixes factors",
            "distinct free word factors with positions",
        ),
        examples=(
            OperationExample(
                name="factors_of_xy",
                description="Group distinct factors of xy with all source start positions; generator letters must use the declared source alphabet.",
                input={"word": {"alphabet": ["x", "y"], "letters": ["x", "y"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.overlaps.compute",
        title="Find free-word overlap and inclusion ambiguities",
        description="Return every compatible nontrivial alignment of the two source words, including proper factor inclusions. Each witness binds a common word, both offsets, and both occurrence contexts. Candidate alignments and aggregate witness letter cells are admitted before enumeration.",
        request_type=FreeAlgebraWordPairRequest,
        result_type=FreeAlgebraWordOverlapResult,
        run=_run_word_overlaps,
        tags=("free-algebra", "word", "overlap", "exact"),
        discovery_terms=(
            "word overlap",
            "suffix prefix overlap",
            "critical word overlap",
        ),
        examples=(
            OperationExample(
                name="xy_yx_overlap",
                description="Find the inclusion of a inside aba with common word and occurrence contexts; both words must use the same ordered alphabet.",
                input={
                    "left": {"alphabet": ["a", "b"], "letters": ["a"]},
                    "right": {"alphabet": ["a", "b"], "letters": ["a", "b", "a"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_word.substitute.compute",
        title="Apply a free-monoid word substitution",
        description="Apply an explicit generator-to-word map to a source word. Source and target alphabets are bound separately; output length is preflighted to 64 and one target interval is returned for every source occurrence, including empty images.",
        request_type=FreeAlgebraWordSubstitutionRequest,
        result_type=FreeAlgebraWordSubstitutionResult,
        run=_run_word_substitute,
        tags=("free-algebra", "word", "substitution", "exact"),
        discovery_terms=(
            "free monoid homomorphism apply",
            "word substitution",
            "generator image substitution",
        ),
        examples=(
            OperationExample(
                name="substitute_xy",
                description="Apply x↦xy and y↦y to xy; the input word must use the substitution source alphabet.",
                input={
                    "substitution": {
                        "source_alphabet": ["x", "y"],
                        "target_alphabet": ["x", "y"],
                        "images": [
                            {"alphabet": ["x", "y"], "letters": ["x", "y"]},
                            {"alphabet": ["x", "y"], "letters": ["y"]},
                        ],
                    },
                    "word": {"alphabet": ["x", "y"], "letters": ["x", "y"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.polynomial.substitute.compute",
        title="Apply an exact free-algebra homomorphism to a polynomial",
        description=(
            "Extend a specified generator-to-polynomial map uniquely to a "
            "unital QQ-algebra homomorphism between free associative algebras. "
            "Source and target ordered alphabets are explicit, and a generator "
            "may map to zero. Expansion count, word length, exact coefficient "
            "growth, work, support, and serialized output are admitted before "
            "word-by-word expansion."
        ),
        request_type=FreeAlgebraPolynomialSubstitutionRequest,
        result_type=FreeAlgebraPolynomial,
        run=_run_polynomial_substitute,
        tags=("free-algebra", "polynomial", "homomorphism", "substitution", "exact"),
        discovery_terms=(
            "free associative algebra homomorphism",
            "noncommutative polynomial substitution",
            "generator-to-polynomial map",
            "free algebra morphism apply",
        ),
        examples=(
            OperationExample(
                name="commutative_collapse_with_zero_image",
                description=(
                    "Map x to a+b and y to zero in QQ<x,y>, then evaluate "
                    "xy+1; the image is the unit because the y-containing "
                    "monomial vanishes."
                ),
                input={
                    "substitution": {
                        "source_alphabet": ["x", "y"],
                        "target_alphabet": ["a", "b"],
                        "images": [
                            {
                                "alphabet": ["a", "b"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["b"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["a"],
                                    },
                                ],
                            },
                            {"alphabet": ["a", "b"], "terms": []},
                        ],
                    },
                    "polynomial": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "word": ["x", "y"],
                            },
                            {"coefficient": {"num": "1", "den": "1"}, "word": []},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.left_ideal.generated_prefix.compute",
        title="Generate a bounded left free-algebra ideal prefix",
        description="Generate exact left multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side left.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_left_prefix,
        tags=("free-algebra", "left-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="left_prefix",
                description="Generate degree-two left multiples of x; the ideal side must be left.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "left",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.right_ideal.generated_prefix.compute",
        title="Generate a bounded right free-algebra ideal prefix",
        description="Generate exact right multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side right.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_right_prefix,
        tags=("free-algebra", "right-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="right_prefix",
                description="Generate degree-two right multiples of x; the ideal side must be right.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "right",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.generated_prefix.compute",
        title="Generate a bounded two-sided free-algebra ideal prefix",
        description="Generate exact two-sided multiples through one finite degree; the ideal is bound to one ordered free alphabet and must have side two-sided.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=_run_two_sided_prefix,
        tags=("free-algebra", "two-sided-ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="two_sided_prefix",
                description="Generate degree-two two-sided multiples of x; the ideal side must be two-sided.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.ideal.generated_prefix.compute",
        title="Generate a bounded sided free-algebra ideal prefix",
        description="Generate the exact homogeneous degree prefix of a left, right, or two-sided ideal over one ordered free alphabet; sidedness remains part of the ideal value.",
        request_type=FreeAlgebraIdealPrefixRequest,
        result_type=FreeAlgebraIdealPrefixResult,
        run=lambda request: ideal_generated_prefix(request.ideal, request.degree),
        tags=("free-algebra", "ideal", "noncommutative", "exact"),
        examples=(
            OperationExample(
                name="left_right_distinction",
                description="Generate degree-two multiples of the generator x on the left; sidedness and the ordered alphabet are explicit.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x"],
                                    }
                                ],
                            }
                        ],
                        "side": "left",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.degree_component.compute",
        title="Compute an exact homogeneous two-sided ideal component",
        description=(
            "Return the canonical rational row-reduced basis of I_n in the "
            "degree-n word space for a finitely generated homogeneous two-sided "
            "ideal I in QQ<X>. This direct finite linear-algebra computation "
            "does not depend on Groebner-Shirshov completion. It admits at most "
            "128 ambient words, 256 context multiples, 32768 matrix cells, "
            "64-digit exact basis coefficients, and a 2 MB serialized result."
        ),
        request_type=FreeAlgebraIdealDegreeComponentRequest,
        result_type=FreeAlgebraIdealDegreeComponentResult,
        run=_run_degree_component,
        tags=("free-algebra", "two-sided-ideal", "homogeneous", "exact"),
        discovery_terms=(
            "free associative ideal degree component",
            "homogeneous two-sided ideal basis",
            "noncommutative ideal component",
            "free algebra exact membership degree",
        ),
        examples=(
            OperationExample(
                name="commutator_ideal_degree_three",
                description=(
                    "Compute the degree-three component generated by xy-yx "
                    "over QQ<x,y>; the basis is the row space of all degree-"
                    "three left and right context multiples."
                ),
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.membership.decide",
        title="Decide bounded two-sided ideal membership",
        description=(
            "Decide membership in a finitely generated homogeneous two-sided "
            "ideal of QQ<X> by completing a degree-lexicographic "
            "Groebner-Shirshov basis through the candidate's largest degree "
            "and reducing the candidate. MEMBER and NOT_MEMBER are returned "
            "only after completion and exact normal-form reduction. If the "
            "bounded completion or reduction envelope is exceeded, return "
            "UNKNOWN with no mathematical conclusion. Nonhomogeneous ideal "
            "generators are rejected because a finite degree cutoff would not "
            "certify this contract."
        ),
        request_type=FreeAlgebraIdealMembershipRequest,
        result_type=FreeAlgebraIdealMembershipResult,
        run=_run_membership,
        tags=("free-algebra", "two-sided-ideal", "membership", "exact"),
        discovery_terms=(
            "noncommutative polynomial ideal membership",
            "free associative algebra membership",
            "Groebner Shirshov ideal membership",
            "two-sided ideal normal form",
        ),
        examples=(
            OperationExample(
                name="commutator_membership",
                description="Decide whether xy-yx lies in the two-sided ideal it generates.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "polynomial": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "-1", "den": "1"},
                                "word": ["y", "x"],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "word": ["x", "y"],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_ideal.groebner_shirshov_through_degree.compute",
        title="Complete two-sided Groebner-Shirshov compositions through a degree",
        description=(
            "For homogeneous two-sided ideal generators, reduce every ordered "
            "overlap and every occurrence of each inclusion composition through "
            "the requested degree to a fixed point. Return only basis elements "
            "in that degree prefix; this is not a global completion claim."
        ),
        request_type=GroebnerShirshovRequest,
        result_type=GroebnerShirshovResult,
        run=lambda request: groebner_shirshov_through_degree(
            request.ideal, request.degree
        ),
        tags=("free-algebra", "groebner-shirshov", "two-sided", "exact"),
        examples=(
            OperationExample(
                name="commutator_generator",
                description="Complete the one-generator two-sided ideal through degree three; every overlap composition through the bound must reduce to zero.",
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_quotient.truncated_algebra.compute",
        title="Build a finite degree-truncated free-algebra quotient",
        description=(
            "Construct the exact associative algebra QQ<X>/(I + F_{>D}) "
            "from a homogeneous two-sided ideal I. The result is the "
            "canonical normal-word basis through degree D, an exact reduced "
            "multiplication table over QQ, and the quotient unit. Products "
            "whose total degree exceeds D are zero by the explicit truncation. "
            "This is a genuine finite-dimensional quotient and does not claim "
            "that the untruncated presented algebra is finite-dimensional. "
            "Normal-word, table-term, reduction-work, and output bounds are "
            "checked before multiplication-table construction."
        ),
        request_type=TruncatedFreeAlgebraQuotientRequest,
        result_type=TruncatedFreeAlgebraQuotient,
        run=_run_truncated_quotient,
        tags=(
            "free-algebra",
            "quotient",
            "truncated-algebra",
            "structure-constants",
            "exact",
        ),
        discovery_terms=(
            "truncated free associative algebra quotient",
            "finite-dimensional quotient of a free algebra through degree D",
            "noncommutative quotient multiplication table",
            "quotient algebra structure constants from relations",
        ),
        examples=(
            OperationExample(
                name="commutative_polynomial_quotient_truncated_at_degree_two",
                description=(
                    "The commutator quotient has the six normal words through "
                    "degree two, and products above degree two vanish."
                ),
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="free_algebra.two_sided_quotient.normal_word_profile.compute",
        title="Compute a bounded free-algebra quotient normal-word profile",
        description=(
            "For a homogeneous two-sided ideal in QQ<X>, compute a complete "
            "Groebner-Shirshov basis through degree D, then return the canonical "
            "irreducible-word basis and Hilbert-function value in every degree "
            "from zero through D. This finite graded prefix does not claim that "
            "the quotient is finite-dimensional."
        ),
        request_type=FreeAlgebraQuotientProfileRequest,
        result_type=FreeAlgebraQuotientProfileResult,
        run=lambda request: quotient_normal_word_profile(request.ideal, request.degree),
        tags=("free-algebra", "quotient", "normal-words", "hilbert-function", "exact"),
        examples=(
            OperationExample(
                name="commutative_prefix",
                description=(
                    "The commutator quotient of QQ<x,y> has one normal word in "
                    "degree zero, two in degree one, and n+1 in each degree n "
                    "through the requested bound."
                ),
                input={
                    "ideal": {
                        "alphabet": ["x", "y"],
                        "generators": [
                            {
                                "alphabet": ["x", "y"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "word": ["y", "x"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["x", "y"],
                                    },
                                ],
                            }
                        ],
                        "side": "two-sided",
                    },
                    "degree": 3,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
