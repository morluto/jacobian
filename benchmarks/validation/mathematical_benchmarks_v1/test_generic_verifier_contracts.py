"""Generic verifier contract tests, split by behavior.

The behavior-owned test functions live in the ``_contracts_*`` modules and are
re-exported here so the planner's generic-contracts selector
(``test_generic_verifier_contracts.py``) continues to collect every cross-task
invariant. Each behavior area is independently editable in its own module;
this aggregator only wires them together for collection. The ``_contracts_*``
modules are underscore-prefixed so pytest does not collect them as separate
test files, avoiding duplicate collection.
"""

from __future__ import annotations

from benchmarks.validation.mathematical_benchmarks_v1._contracts_assurance_protocol import *  # noqa: F403
from benchmarks.validation.mathematical_benchmarks_v1._contracts_evidence_binding import *  # noqa: F403
from benchmarks.validation.mathematical_benchmarks_v1._contracts_input_binding import *  # noqa: F403
from benchmarks.validation.mathematical_benchmarks_v1._contracts_submission_attacks import *  # noqa: F403
