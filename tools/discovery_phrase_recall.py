"""Report first-page owner recall for authored multiword discovery phrases.

Run: uv run python tools/discovery_phrase_recall.py --output phrase-recall.json
This measures retrieval only, not mathematical correctness or agent selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jacobian.catalog.models import OperationMatchRequest
from jacobian.catalog.search import (
    OperationSearchIndex,
    SearchableOperation,
    _authored_phrase,
)


@dataclass(frozen=True)
class PhraseRank:
    phrase: str
    authored_spellings: tuple[str, ...]
    owner: str
    owners_for_phrase: int
    first_page_rank: int | None
    total_matches: int


@dataclass(frozen=True)
class PhraseRecallReport:
    operation_count: int
    distinct_phrases: int
    owner_phrase_count: int
    page_limit: int
    first_page_hits: int
    owner_recall: float | None
    rows: tuple[PhraseRank, ...]


def authored_phrase_recall(
    operations: Sequence[SearchableOperation], *, page_limit: int = 5
) -> PhraseRecallReport:
    """Report each normalized multiword phrase and its distinct owning IDs."""
    if type(page_limit) is not int or not 1 <= page_limit <= 20:
        raise ValueError("page_limit must be between 1 and 20")
    owners: dict[str, set[str]] = defaultdict(set)
    spellings: dict[str, set[str]] = defaultdict(set)
    for operation in operations:
        for phrase in operation.discovery_terms:
            if len(phrase.split()) > 1:
                key = _authored_phrase(phrase)
                owners[key].add(operation.operation_id)
                spellings[key].add(phrase)
    index = OperationSearchIndex(operations)
    rows: list[PhraseRank] = []
    for phrase, declared_owners in sorted(owners.items()):
        result = index.match(OperationMatchRequest(need=phrase, limit=page_limit))
        ranks = {
            match.operation_id: rank
            for rank, match in enumerate(result.matches, start=1)
        }
        rows.extend(
            PhraseRank(
                phrase=phrase,
                authored_spellings=tuple(sorted(spellings[phrase])),
                owner=owner,
                owners_for_phrase=len(declared_owners),
                first_page_rank=ranks.get(owner),
                total_matches=result.total_matches,
            )
            for owner in sorted(declared_owners)
        )
    hits = sum(row.first_page_rank is not None for row in rows)
    return PhraseRecallReport(
        operation_count=len(operations),
        distinct_phrases=len(owners),
        owner_phrase_count=len(rows),
        page_limit=page_limit,
        first_page_hits=hits,
        owner_recall=hits / len(rows) if rows else None,
        rows=tuple(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    report = asdict(authored_phrase_recall(BUILTIN_TOOLS, page_limit=arguments.limit))
    if arguments.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "rows"},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
