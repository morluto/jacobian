"""Code linear operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_linear._models import (
    CodeEqualRequest,
    CodeEqualResult,
    CodewordCheckRequest,
    CodewordCheckResult,
    DualCodeRequest,
    DualCodeResult,
    FromGeneratorResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    ParityCheckRequest,
    ParityCheckResult,
    PunctureRequest,
    PunctureResult,
    ReceivedWordProfileRequest,
    ReceivedWordProfileResult,
    ShortenRequest,
    ShortenResult,
    SyndromeRequest,
    SyndromeResult,
)
from jacobian.math.code_linear._operations import (
    compute_code_equal,
    compute_codeword_check,
    compute_dual_code,
    compute_from_generator,
    compute_macwilliams_transform,
    compute_parity_check,
    compute_puncture,
    compute_received_word_profile,
    compute_shorten,
    compute_syndrome,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "code.linear.received_word_profile.compute",
        "Compute a linear code's received-word distance profile",
        "Enumerate every word of a bounded full-rank prime-field linear "
        "encoder and return its complete Hamming-distance histogram from one "
        "received word. The dense histogram is the coset weight distribution; "
        "optional exact integer thresholds can count, return the first, or "
        "return every replayable message/codeword witness. Message and codeword "
        "coordinates follow the encoder's explicit ordered axes.",
        ReceivedWordProfileRequest,
        ReceivedWordProfileResult,
        compute_received_word_profile,
        "code",
        "linear",
        "coset-weight-distribution",
        "hamming-distance",
        "exact",
        version="2",
        examples=(
            example(
                "outside_binary_repetition_code",
                "Profile received word 10 against the length-two binary "
                "repetition code; generator rows must be full rank and every "
                "word entry must be a canonical GF(2) residue.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                    "received_word": [1, 0],
                },
            ),
            example(
                "first_strict_agreement_witness",
                "Return the first codeword agreeing in more than zero "
                "coordinates with 10; FIRST requires an exact threshold and "
                "messages are ordered lexicographically on message_axis.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                    "received_word": [1, 0],
                    "threshold": {
                        "metric": "AGREEMENT",
                        "comparison": "GT",
                        "value": 0,
                    },
                    "witness_mode": "FIRST",
                },
            ),
        ),
    ),
    _op(
        "code.linear.from_generator.compute",
        "Canonicalize a linear code from a generator matrix",
        "Reduce a generator matrix over a bounded prime field to canonical "
        "full-row-rank form, preserve its ordered coordinate-axis labels, and "
        "return a linear encoder. Canonical basis rows receive deterministic "
        "message-axis labels m0, m1, and so on.",
        GeneratorMatrixRequest,
        FromGeneratorResult,
        compute_from_generator,
        "code",
        "linear",
        "exact",
        version="2",
        examples=(
            example(
                "binary_repetition",
                "Canonicalize the binary repetition code of length two; the "
                "coordinate axis must label the generator columns in order.",
                {
                    "field_order": 2,
                    "generator_matrix": [[1, 1]],
                    "coordinate_axis": ["x0", "x1"],
                },
            ),
        ),
    ),
    _op(
        "code.linear.dual.compute",
        "Compute the dual code of a linear code",
        "Compute the exact dual code C^perp of a canonical full-rank linear "
        "encoder as a linear encoder that preserves the primal coordinate "
        "axis. Canonical dual basis rows receive deterministic message-axis "
        "labels m0, m1, and so on.",
        DualCodeRequest,
        DualCodeResult,
        compute_dual_code,
        "code",
        "dual",
        "exact",
        version="3",
        examples=(
            example(
                "binary_repetition",
                "Compute the dual of the binary repetition code of length "
                "two; supply any producer's canonical full-rank encoder "
                "unchanged.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "code.linear.parity_check.compute",
        "Compute a parity-check matrix for a linear code",
        "Return one canonical parity-check matrix for the code of a canonical "
        "full-rank linear encoder, with dimension and rank relation "
        "k = n - rank(H). The matrix columns keep the encoder's ordered "
        "coordinate axis, and the length-zero encoder yields a zero-column "
        "matrix.",
        ParityCheckRequest,
        ParityCheckResult,
        compute_parity_check,
        "code",
        "parity-check",
        "exact",
        version="3",
        examples=(
            example(
                "binary_repetition",
                "Parity-check of the binary repetition code of length two; "
                "supply any producer's canonical full-rank encoder unchanged.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "code.linear.codeword.check",
        "Check whether a word is a codeword",
        "Check whether a word lies in the row space of a canonical full-rank "
        "linear encoder's generator and return membership, Hamming weight, "
        "and syndrome. The word entries follow the encoder's ordered "
        "coordinate axis.",
        CodewordCheckRequest,
        CodewordCheckResult,
        compute_codeword_check,
        "code",
        "codeword",
        "exact",
        version="2",
        examples=(
            example(
                "member_word",
                "Check [1,1] against the canonical binary repetition encoder "
                "of length two.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                    "word": [1, 1],
                },
            ),
        ),
    ),
    _op(
        "code.linear.syndrome.compute",
        "Compute the syndrome of a word under a parity-check",
        "Return the exact syndrome Hw^T over the prime field, and whether "
        "the word is a member of the code. The word must present the same "
        "ordered coordinate axis as the parity-check columns.",
        SyndromeRequest,
        SyndromeResult,
        compute_syndrome,
        "code",
        "syndrome",
        "exact",
        version="2",
        examples=(
            example(
                "binary_repetition_syndrome",
                "Syndrome of [1,0] under parity-check [1,1] over F_2; the "
                "word presents the parity-check's ordered column axis.",
                {
                    "parity_check": {
                        "field_order": 2,
                        "coordinate_axis": ["x0", "x1"],
                        "rows": [[1, 1]],
                    },
                    "coordinate_axis": ["x0", "x1"],
                    "word": [1, 0],
                },
            ),
        ),
    ),
    _op(
        "code.linear.equal.decide",
        "Decide whether two linear encoders define the same code",
        "Check exact mutual row-space containment between two canonical "
        "full-rank linear encoders on one shared ordered coordinate axis; "
        "return equality or a concrete codeword witnessing the difference.",
        CodeEqualRequest,
        CodeEqualResult,
        compute_code_equal,
        "code",
        "equality",
        "exact",
        version="2",
        examples=(
            example(
                "equal_row_equivalent",
                "Two row-equivalent encoders of the full binary space of "
                "length two define the same code.",
                {
                    "encoder_a": {
                        "field_order": 2,
                        "message_axis": ["m0", "m1"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 0], [0, 1]],
                    },
                    "encoder_b": {
                        "field_order": 2,
                        "message_axis": ["m0", "m1"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1], [0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "code.linear.macwilliams_transform.compute",
        "MacWilliams transform of a weight distribution",
        "Apply the q-ary MacWilliams identity to compute the dual code weight "
        "distribution from the primal weight enumerator.",
        MacWilliamsRequest,
        MacWilliamsResult,
        compute_macwilliams_transform,
        "code",
        "macwilliams",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "MacWilliams transform of the binary length-2 repetition code.",
                {
                    "field_order": 2,
                    "code_cardinality": 2,
                    "length": 2,
                    "weights": [1, 0, 1],
                },
            ),
        ),
    ),
    _op(
        "code.linear.puncture.compute",
        "Puncture a linear code at one coordinate",
        "Delete one indexed coordinate and its axis label from a canonical "
        "linear encoder, then return the punctured code as a canonical linear "
        "encoder whose basis rows receive deterministic message-axis labels "
        "m0, m1, and so on. The source encoder must be full rank, and the "
        "coordinate must index its ordered coordinate axis.",
        PunctureRequest,
        PunctureResult,
        compute_puncture,
        "code",
        "puncture",
        "exact",
        version="2",
        examples=(
            example(
                "binary_repetition",
                "Puncture the binary length-2 repetition code at coordinate 0; "
                "supply any producer's canonical full-rank encoder unchanged, "
                "with the coordinate indexing its ordered axis.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                    "coordinate": 0,
                },
            ),
        ),
    ),
    _op(
        "code.linear.shorten.compute",
        "Shorten a linear code at one coordinate",
        "Shorten a canonical linear encoder by fixing one coordinate to zero "
        "and then puncturing it, deleting that coordinate's axis label and "
        "returning a canonical linear encoder. Canonical basis rows receive "
        "deterministic message-axis labels m0, m1, and so on. The source "
        "encoder must be full rank, and the coordinate must index its ordered "
        "coordinate axis.",
        ShortenRequest,
        ShortenResult,
        compute_shorten,
        "code",
        "shorten",
        "exact",
        version="2",
        examples=(
            example(
                "binary_repetition",
                "Shorten the binary length-3 repetition code at coordinate 0; "
                "supply any producer's canonical full-rank encoder unchanged, "
                "with the coordinate indexing its ordered axis.",
                {
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1", "x2"],
                        "generator_matrix": [[1, 1, 1]],
                    },
                    "coordinate": 0,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
