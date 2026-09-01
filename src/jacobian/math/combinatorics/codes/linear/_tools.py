"""Code linear operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.codes.linear._models import (
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
from jacobian.math.combinatorics.codes.linear.operations import (
    code_equal,
    codeword_check,
    dual_code,
    from_generator,
    macwilliams_transform,
    parity_check,
    puncture,
    received_word_profile,
    shorten,
    syndrome,
)


def compute_received_word_profile(
    request: ReceivedWordProfileRequest,
) -> ReceivedWordProfileResult:
    return received_word_profile(
        request.encoder,
        request.received_word,
        request.threshold,
        request.witness_mode,
    )


def compute_from_generator(request: GeneratorMatrixRequest) -> FromGeneratorResult:
    return from_generator(
        request.field_order,
        request.generator_matrix,
        request.coordinate_axis,
    )


def compute_dual_code(request: DualCodeRequest) -> DualCodeResult:
    return dual_code(request.encoder)


def compute_parity_check(request: ParityCheckRequest) -> ParityCheckResult:
    return parity_check(request.encoder)


def compute_codeword_check(request: CodewordCheckRequest) -> CodewordCheckResult:
    return codeword_check(request.encoder, request.word)


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    return syndrome(request.parity_check, request.coordinate_axis, request.word)


def compute_code_equal(request: CodeEqualRequest) -> CodeEqualResult:
    return code_equal(request.encoder_a, request.encoder_b)


def compute_macwilliams_transform(request: MacWilliamsRequest) -> MacWilliamsResult:
    return macwilliams_transform(
        request.field_order,
        request.code_cardinality,
        request.length,
        request.weights,
    )


def compute_puncture(request: PunctureRequest) -> PunctureResult:
    return puncture(request.encoder, request.coordinate)


def compute_shorten(request: ShortenRequest) -> ShortenResult:
    return shorten(request.encoder, request.coordinate)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="code.linear.received_word_profile.compute",
        title="Compute a linear code's received-word distance profile",
        description="Enumerate every word of a bounded full-rank prime-field linear "
        "encoder and return its complete Hamming-distance histogram from one "
        "received word. The dense histogram is the coset weight distribution; "
        "optional exact integer thresholds can count, return the first, or "
        "return every replayable message/codeword witness. Message and codeword "
        "coordinates follow the encoder's explicit ordered axes.",
        request_type=ReceivedWordProfileRequest,
        result_type=ReceivedWordProfileResult,
        run=compute_received_word_profile,
        tags=(
            "code",
            "linear",
            "coset-weight-distribution",
            "hamming-distance",
            "exact",
        ),
        examples=(
            OperationExample(
                name="outside_binary_repetition_code",
                description="Profile received word 10 against the length-two binary "
                "repetition code; generator rows must be full rank and every "
                "word entry must be a canonical GF(2) residue.",
                input={
                    "encoder": {
                        "field_order": 2,
                        "message_axis": ["m0"],
                        "coordinate_axis": ["x0", "x1"],
                        "generator_matrix": [[1, 1]],
                    },
                    "received_word": [1, 0],
                },
            ),
            OperationExample(
                name="first_strict_agreement_witness",
                description="Return the first codeword agreeing in more than zero "
                "coordinates with 10; FIRST requires an exact threshold and "
                "messages are ordered lexicographically on message_axis.",
                input={
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
    MathTool(
        operation_id="code.linear.from_generator.compute",
        title="Canonicalize a linear code from a generator matrix",
        description="Reduce a generator matrix over a bounded prime field to canonical "
        "full-row-rank form, preserve its ordered coordinate-axis labels, and "
        "return a linear encoder. Canonical basis rows receive deterministic "
        "message-axis labels m0, m1, and so on.",
        request_type=GeneratorMatrixRequest,
        result_type=FromGeneratorResult,
        run=compute_from_generator,
        tags=("code", "linear", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="Canonicalize the binary repetition code of length two; the "
                "coordinate axis must label the generator columns in order.",
                input={
                    "field_order": 2,
                    "generator_matrix": [[1, 1]],
                    "coordinate_axis": ["x0", "x1"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.linear.dual.compute",
        title="Compute the dual code of a linear code",
        description="Compute the exact dual code C^perp of a canonical full-rank linear "
        "encoder as a linear encoder that preserves the primal coordinate "
        "axis. Canonical dual basis rows receive deterministic message-axis "
        "labels m0, m1, and so on.",
        request_type=DualCodeRequest,
        result_type=DualCodeResult,
        run=compute_dual_code,
        tags=("code", "dual", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="Compute the dual of the binary repetition code of length "
                "two; supply any producer's canonical full-rank encoder "
                "unchanged.",
                input={
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
    MathTool(
        operation_id="code.linear.parity_check.compute",
        title="Compute a parity-check matrix for a linear code",
        description="Return one canonical parity-check matrix for the code of a canonical "
        "full-rank linear encoder, with dimension and rank relation "
        "k = n - rank(H). The matrix columns keep the encoder's ordered "
        "coordinate axis, and the length-zero encoder yields a zero-column "
        "matrix.",
        request_type=ParityCheckRequest,
        result_type=ParityCheckResult,
        run=compute_parity_check,
        tags=("code", "parity-check", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="Parity-check of the binary repetition code of length two; "
                "supply any producer's canonical full-rank encoder unchanged.",
                input={
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
    MathTool(
        operation_id="code.linear.codeword.check",
        title="Check whether a word is a codeword",
        description="Check whether a word lies in the row space of a canonical full-rank "
        "linear encoder's generator and return membership, Hamming weight, "
        "and syndrome. The word entries follow the encoder's ordered "
        "coordinate axis.",
        request_type=CodewordCheckRequest,
        result_type=CodewordCheckResult,
        run=compute_codeword_check,
        tags=("code", "codeword", "exact"),
        examples=(
            OperationExample(
                name="member_word",
                description="Check [1,1] against the canonical binary repetition encoder "
                "of length two.",
                input={
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
    MathTool(
        operation_id="code.linear.syndrome.compute",
        title="Compute the syndrome of a word under a parity-check",
        description="Return the exact syndrome Hw^T over the prime field, and whether "
        "the word is a member of the code. The word must present the same "
        "ordered coordinate axis as the parity-check columns.",
        request_type=SyndromeRequest,
        result_type=SyndromeResult,
        run=compute_syndrome,
        tags=("code", "syndrome", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition_syndrome",
                description="Syndrome of [1,0] under parity-check [1,1] over F_2; the "
                "word presents the parity-check's ordered column axis.",
                input={
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
    MathTool(
        operation_id="code.linear.equal.decide",
        title="Decide whether two linear encoders define the same code",
        description="Check exact mutual row-space containment between two canonical "
        "full-rank linear encoders on one shared ordered coordinate axis; "
        "return equality or a concrete codeword witnessing the difference.",
        request_type=CodeEqualRequest,
        result_type=CodeEqualResult,
        run=compute_code_equal,
        tags=("code", "equality", "exact"),
        examples=(
            OperationExample(
                name="equal_row_equivalent",
                description="Two row-equivalent encoders of the full binary space of "
                "length two define the same code.",
                input={
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
    MathTool(
        operation_id="code.linear.macwilliams_transform.compute",
        title="MacWilliams transform of a weight distribution",
        description="Apply the q-ary MacWilliams identity to compute the dual code weight "
        "distribution from the primal weight enumerator.",
        request_type=MacWilliamsRequest,
        result_type=MacWilliamsResult,
        run=compute_macwilliams_transform,
        tags=("code", "macwilliams", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="MacWilliams transform of the binary length-2 repetition code.",
                input={
                    "field_order": 2,
                    "code_cardinality": 2,
                    "length": 2,
                    "weights": [1, 0, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="code.linear.puncture.compute",
        title="Puncture a linear code at one coordinate",
        description="Delete one indexed coordinate and its axis label from a canonical "
        "linear encoder, then return the punctured code as a canonical linear "
        "encoder whose basis rows receive deterministic message-axis labels "
        "m0, m1, and so on. The source encoder must be full rank, and the "
        "coordinate must index its ordered coordinate axis.",
        request_type=PunctureRequest,
        result_type=PunctureResult,
        run=compute_puncture,
        tags=("code", "puncture", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="Puncture the binary length-2 repetition code at coordinate 0; "
                "supply any producer's canonical full-rank encoder unchanged, "
                "with the coordinate indexing its ordered axis.",
                input={
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
    MathTool(
        operation_id="code.linear.shorten.compute",
        title="Shorten a linear code at one coordinate",
        description="Shorten a canonical linear encoder by fixing one coordinate to zero "
        "and then puncturing it, deleting that coordinate's axis label and "
        "returning a canonical linear encoder. Canonical basis rows receive "
        "deterministic message-axis labels m0, m1, and so on. The source "
        "encoder must be full rank, and the coordinate must index its ordered "
        "coordinate axis.",
        request_type=ShortenRequest,
        result_type=ShortenResult,
        run=compute_shorten,
        tags=("code", "shorten", "exact"),
        examples=(
            OperationExample(
                name="binary_repetition",
                description="Shorten the binary length-3 repetition code at coordinate 0; "
                "supply any producer's canonical full-rank encoder unchanged, "
                "with the coordinate indexing its ordered axis.",
                input={
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
