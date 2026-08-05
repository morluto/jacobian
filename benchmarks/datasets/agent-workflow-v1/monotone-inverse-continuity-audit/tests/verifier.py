import codecs
import json
import re
from collections import deque
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

E = Path("/tests")
W = Path("/app")
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
EVIDENCE_CHUNK_BYTES = 64 * 1024
RESULT_PREFIX = "RESULT_JSON:"
PROSE_WINDOW_CHARS = 512
LINE_BREAKS = re.compile(r"([\n\r\v\f\x1c-\x1e\x85\u2028\u2029])")


def _fraction(value):
    if not isinstance(value, str):
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def _valid_countermodel(result, source):
    keys = {
        "left_slope",
        "right_slope",
        "offset",
        "jump",
        "left_endpoint_value",
        "left_limit",
        "right_breakpoint_value",
        "right_endpoint_value",
        "gap_witness",
    }
    if not isinstance(result, dict) or set(result) != keys:
        return False
    try:
        value = {key: _fraction(item) for key, item in result.items()}
        bounds = source["parameter_bounds"]
        left = _fraction(source["interval"]["left"])
        right = _fraction(source["interval"]["right"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    for key in ("left_slope", "right_slope", "jump", "offset"):
        try:
            if (
                not _fraction(bounds[key]["minimum"])
                <= value[key]
                <= _fraction(bounds[key]["maximum"])
            ):
                return False
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
    m_left = value["left_slope"]
    m_right = value["right_slope"]
    offset = value["offset"]
    jump = value["jump"]
    left_limit = offset
    right_zero = offset + jump
    witness = value["gap_witness"]
    return bool(
        m_left > 0
        and m_right > 0
        and jump > 0
        and value["left_endpoint_value"] == m_left * left + offset
        and value["left_limit"] == left_limit
        and value["right_breakpoint_value"] == right_zero
        and value["right_endpoint_value"] == m_right * right + right_zero
        and left_limit < witness < right_zero
        and value["left_endpoint_value"] < witness < value["right_endpoint_value"]
    )


class _ProseScanner:
    """Recognize the published prose obligations in linear time and bounded space."""

    def __init__(self):
        self._tail = ""
        self._position = 0
        self._previous_was_space = False
        self._branch_positions = deque()
        self._pending_contradictions = deque()
        self._sentence_has_inverse = False
        self._sentence_has_contradiction = False
        self.contradicts_inverse_failure = False
        self.branch_subject = False
        self.strictly_increasing = False
        self.positive_slopes = False
        self.positive_jump_term = False
        self.jump_term = False
        self.image_term = False
        self.omitted_term = False
        self.interval_term = False
        self.inverse_term = False
        self.failure_term = False

    def feed(self, text):
        normalized = re.sub(r"\s+", " ", text.casefold())
        if self._previous_was_space and normalized.startswith(" "):
            normalized = normalized[1:]
        if not normalized:
            return
        self._previous_was_space = normalized.endswith(" ")

        previous_position = self._position
        combined = self._tail + normalized
        combined_start = previous_position - len(self._tail)
        self._scan_obligations(combined)

        events = []
        for kind, pattern in (
            ("inverse", re.compile(r"\binverse\b")),
            ("branch", re.compile(r"\b(?:branch(?:es)?|pieces?)\b")),
            (
                "contradiction",
                re.compile(
                    r"\b(?:(?:(?:do(?:es)?|did|will|would|can|could|should|must)"
                    r"\s+not|cannot)\s+fail(?:s|ed|ing)?|"
                    r"(?:doesn|don|didn|won|wouldn|can|couldn|shouldn|mustn)"
                    r"['\u2019]?t\s+fail(?:s|ed|ing)?|succeeds?|works?)\b"
                ),
            ),
            ("sentence", re.compile(r"[.!?;]")),
        ):
            for match in pattern.finditer(combined):
                start = combined_start + match.start()
                end = combined_start + match.end()
                if end > previous_position:
                    events.append((start, end, kind))
        for start, end, kind in sorted(events):
            if kind == "sentence":
                self._finish_sentence()
            elif kind == "inverse":
                self._sentence_has_inverse = True
            elif kind == "branch":
                self._record_branch(start, end)
            else:
                self._record_contradiction(start, end)

        self._position += len(normalized)
        self._expire_local_state(self._position)
        self._tail = combined[-PROSE_WINDOW_CHARS:]

    def finish(self):
        self._expire_local_state(float("inf"))
        self._finish_sentence()

    def matches_obligations(self):
        return bool(
            self.branch_subject
            and self.strictly_increasing
            and self.positive_slopes
            and self.positive_jump_term
            and self.jump_term
            and self.image_term
            and self.omitted_term
            and self.interval_term
            and self.inverse_term
            and self.failure_term
            and not self.contradicts_inverse_failure
        )

    def _scan_obligations(self, text):
        self.branch_subject |= bool(
            re.search(
                r"\b(?:(?:both|each)\s+(?:affine\s+)?(?:branch(?:es)?|pieces?)|"
                r"(?:the\s+)?left\s+and\s+right\s+(?:affine\s+)?"
                r"(?:branch(?:es)?|pieces?))\b",
                text,
            )
        )
        self.strictly_increasing |= bool(
            re.search(r"\bstrictly\s+(?:increas(?:e|es|ing|ed)|monotone)\b", text)
        )
        self.positive_slopes |= bool(
            re.search(
                r"\b(?:slopes?\b[^.!?;\n]{0,32}\b"
                r"(?:are|is|remain|remains|stay|stays)\s+(?:strictly\s+)?positive|"
                r"(?:their|the|both)\s+(?:strictly\s+)?positive\s+slopes?)\b",
                text,
            )
        )
        self.positive_jump_term |= bool(
            re.search(r"\b(?:positive|upward|strictly positive|nonzero)\b", text)
        )
        self.jump_term |= bool(re.search(r"\b(?:jump|discontinu(?:ity|ous))\b", text))
        self.image_term |= "image" in text
        self.omitted_term |= bool(
            re.search(
                r"\b(?:omit(?:s|ted)?|gap|missing|exclude(?:s|d)?|no preimage)\b",
                text,
            )
        )
        self.interval_term |= bool(
            re.search(r"\b(?:between|endpoint|full interval|no preimage)\b", text)
        )
        self.inverse_term |= bool(re.search(r"\b(?:inverse|preimage)\b", text))
        self.failure_term |= bool(
            re.search(r"\b(?:no|not|fail(?:s|ure)?|without|omits?)\b", text)
        )

    def _record_branch(self, start, end):
        self._branch_positions.append((start, end))
        for pending in self._pending_contradictions:
            contradiction_start, contradiction_end, qualified = pending
            if (
                not qualified
                and start >= contradiction_start - 48
                and end <= contradiction_end + 48
            ):
                pending[2] = True

    def _record_contradiction(self, start, end):
        qualified = any(
            branch_start >= start - 48 and branch_end <= end + 48
            for branch_start, branch_end in self._branch_positions
        )
        self._pending_contradictions.append([start, end, qualified])

    def _expire_local_state(self, position):
        # Keep the overlap's branch events until any token spanning the next
        # chunk boundary has been recognized; exact 48-character qualification
        # is applied when the contradiction is recorded.
        while (
            self._branch_positions
            and self._branch_positions[0][1] < position - PROSE_WINDOW_CHARS
        ):
            self._branch_positions.popleft()
        while (
            self._pending_contradictions
            and self._pending_contradictions[0][1] + 48 < position
        ):
            _, _, qualified = self._pending_contradictions.popleft()
            self._sentence_has_contradiction |= not qualified

    def _finish_sentence(self):
        while self._pending_contradictions:
            _, _, qualified = self._pending_contradictions.popleft()
            self._sentence_has_contradiction |= not qualified
        if self._sentence_has_inverse and self._sentence_has_contradiction:
            self.contradicts_inverse_failure = True
        self._sentence_has_inverse = False
        self._sentence_has_contradiction = False
        self._branch_positions.clear()


class _MarkerScanner:
    """Collect one semantic JSON object without retaining unbounded whitespace."""

    def __init__(self, expected):
        try:
            expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError, MemoryError):
            expected_json = ""
        # A matching object can spell every expected ASCII character as a six-byte
        # JSON escape. Whitespace outside strings is discarded as it streams.
        self._compact_limit = 6 * len(expected_json) + 2
        self._expected = expected
        self._characters = []
        self._in_string = False
        self._escaped = False
        self._overflowed = not expected_json

    def feed(self, text):
        if self._overflowed:
            return
        for character in text:
            if not self._in_string and character.isspace():
                continue
            if len(self._characters) >= self._compact_limit:
                self._overflowed = True
                return
            self._characters.append(character)
            if not self._in_string:
                if character == '"':
                    self._in_string = True
                continue
            if self._escaped:
                self._escaped = False
            elif character == "\\":
                self._escaped = True
            elif character == '"':
                self._in_string = False

    def matches(self):
        if self._overflowed or self._in_string or self._escaped:
            return False
        try:
            parsed = json.loads("".join(self._characters))
        except (ValueError, RecursionError, MemoryError):
            return False
        return isinstance(parsed, dict) and parsed == self._expected


class _EvidenceStreamScanner:
    """Consume evidence lines while retaining only bounded scanner state."""

    def __init__(self, result):
        self._result = result
        self._prose = _ProseScanner()
        self._stored_marker = None
        self._marker_count = 0
        self._marker = None
        self._prefix = ""
        self._mode = "prefix"

    def consume(self, text):
        for index, piece in enumerate(LINE_BREAKS.split(text)):
            if index % 2:
                self._finish_line()
            elif piece:
                self._consume_line(piece)

    def finish(self):
        self._finish_line()
        self._prose.finish()
        return bool(
            self._marker_count == 1
            and self._stored_marker is not None
            and self._stored_marker.matches()
            and self._prose.matches_obligations()
        )

    def _consume_line(self, text):
        if self._mode == "marker":
            self._marker.feed(text)
            return
        if self._mode == "prose":
            self._prose.feed(text)
            return
        needed = len(RESULT_PREFIX) - len(self._prefix)
        candidate = text[:needed]
        expected = RESULT_PREFIX[len(self._prefix) : len(self._prefix) + len(candidate)]
        if candidate != expected:
            self._prose.feed(self._prefix + text)
            self._prefix = ""
            self._mode = "prose"
            return
        self._prefix += candidate
        remainder = text[len(candidate) :]
        if self._prefix == RESULT_PREFIX:
            self._marker = _MarkerScanner(self._result)
            self._mode = "marker"
            self._marker.feed(remainder)

    def _finish_line(self):
        if self._mode == "marker":
            self._marker_count += 1
            if self._marker_count == 1:
                self._stored_marker = self._marker
        elif self._mode == "prefix":
            self._prose.feed(self._prefix)
        self._prose.feed(" ")
        self._marker = None
        self._prefix = ""
        self._mode = "prefix"


def _stream_evidence_matches_result(target, result):
    scanner = _EvidenceStreamScanner(result)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

    try:
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(EVIDENCE_CHUNK_BYTES), b""):
                scanner.consume(decoder.decode(block))
            scanner.consume(decoder.decode(b"", final=True))
    except (OSError, UnicodeError, MemoryError):
        return False
    return scanner.finish()


def _evidence_matches_result(submission):
    """Bind evidence content to the submitted countermodel.

    The public instruction requires a concise derivation in answer.txt. A
    digest-bound file of unrelated bytes, or a marker-only file, must not score
    as valid evidence. The marker binds the structured values, while the prose
    must state the published monotonicity, jump, omitted-image, and inverse
    obligations.
    """

    if not isinstance(submission, dict):
        return False
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    return _stream_evidence_matches_result(target, submission.get("result"))


def main():
    submission = load_submission()
    with (E / "input.json").open(encoding="utf-8") as stream:
        source = json.load(stream)
    with (E / "expected.json").open(encoding="utf-8") as stream:
        expected = json.load(stream)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
        allowed_assurances=ALLOWED_ASSURANCES,
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    # Mathematical correctness is evaluated independently of the envelope and
    # input binding so a protocol, assurance, or input-validity failure is not
    # misreported as wrong mathematics.  Input validity is reported as its own
    # diagnostic and only aggregate reward is gated on it.
    math_correct = _valid_countermodel(result, source)
    input_bound = workspace_input_is_bound()
    evidence_valid = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        and evidence_list_is_bound(
            submission["evidence"], expected_path="evidence/answer.txt"
        )
        and _evidence_matches_result(submission)
    )
    scope_correct = bool(
        contract
        and isinstance(submission, dict)
        and submission.get("scope") == expected["required_scope"]
        and submission.get("limitations") == expected["limitations"]
    )
    assurance_correct = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    # Aggregate reward is zero for wrong mathematics, false certification,
    # malformed or escaped evidence, or unbound input.  Scope and assurance
    # failures reduce reward but do not zero it, preserving diagnostic
    # independence.
    aggregate_eligible = bool(
        contract
        and math_correct
        and input_bound
        and evidence_valid
        and not false_certification
    )
    reward = (
        0.0
        if not aggregate_eligible
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
