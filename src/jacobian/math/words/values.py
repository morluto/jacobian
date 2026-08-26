"""Provider-independent values for bounded combinatorics on words."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import AfterValidator, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers

MAX_WORD_LENGTH = 500
MAX_ALPHABET_SIZE = 50
MAX_SYMBOL_LENGTH = 64
MAX_MORPHISM_IMAGE_LENGTH = 10_000
MAX_MORPHISM_OUTPUT_LENGTH = MAX_WORD_LENGTH
MAX_SUBSTITUTION_DEPENDENCY_OCCURRENCES = 10_000
MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES = 20_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"word.{reason}", message)


def _require_unicode_scalar_string(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "symbol_not_unicode_scalar",
            "symbol must contain only Unicode scalar values",
        )
    return value


Symbol = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SYMBOL_LENGTH),
    AfterValidator(_require_unicode_scalar_string),
]


class FiniteWord(StrictModel):
    """A bounded word over an explicitly ordered finite alphabet.

    The empty alphabet carries exactly the empty word.  Keeping that
    degenerate value representable lets canonical word values compose through
    operations such as RSK without inventing an ambient symbol.
    """

    alphabet: tuple[Symbol, ...] = Field(max_length=MAX_ALPHABET_SIZE)
    letters: tuple[str, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_word_over_ordered_alphabet(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise _validation_error(
                "alphabet_symbols_not_distinct", "alphabet symbols must be distinct"
            )
        if any(letter not in self.alphabet for letter in self.letters):
            raise _validation_error(
                "word_letter_outside_alphabet",
                "word letter is outside the declared alphabet",
            )
        return self


class WordMorphism(StrictModel):
    source_alphabet: tuple[Symbol, ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )
    target_alphabet: tuple[Symbol, ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )
    images: tuple[tuple[str, ...], ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )

    @model_validator(mode="after")
    def require_total_bounded_morphism(self) -> Self:
        if len(set(self.source_alphabet)) != len(self.source_alphabet):
            raise _validation_error(
                "source_alphabet_not_distinct",
                "source alphabet symbols must be distinct",
            )
        if len(set(self.target_alphabet)) != len(self.target_alphabet):
            raise _validation_error(
                "target_alphabet_not_distinct",
                "target alphabet symbols must be distinct",
            )
        if len(self.images) != len(self.source_alphabet):
            raise _validation_error(
                "morphism_image_count_mismatch",
                "morphism must have one image per source symbol",
            )
        if any(len(image) > MAX_MORPHISM_IMAGE_LENGTH for image in self.images):
            raise _validation_error(
                "morphism_image_too_long", "morphism image exceeds the length bound"
            )
        if any(
            letter not in self.target_alphabet
            for image in self.images
            for letter in image
        ):
            raise _validation_error(
                "morphism_image_outside_target",
                "morphism image uses a symbol outside the target alphabet",
            )
        return self


class Substitution(StrictModel):
    """A finite-alphabet endomorphism, retaining its canonical morphism."""

    morphism: WordMorphism

    @model_validator(mode="after")
    def require_endomorphism(self) -> Self:
        if self.morphism.source_alphabet != self.morphism.target_alphabet:
            raise _validation_error(
                "substitution_not_endomorphism",
                "a substitution must have identical source and target alphabets",
            )
        return self


def _require_dependency_occurrence_bound(substitution: Substitution) -> None:
    occurrence_count = sum(len(image) for image in substitution.morphism.images)
    if occurrence_count > MAX_SUBSTITUTION_DEPENDENCY_OCCURRENCES:
        raise ValueError(
            "dependency occurrence output exceeds the aggregate bound "
            f"({occurrence_count} > {MAX_SUBSTITUTION_DEPENDENCY_OCCURRENCES})"
        )


class ProlongableSubstitution(StrictModel):
    """A substitution with a certified unbounded nested seed iterate."""

    substitution: Substitution
    seed: Symbol

    @model_validator(mode="before")
    @classmethod
    def require_bounded_source_before_mortality_analysis(cls, value: Any) -> Any:
        value = canonicalize_json_containers(value)
        _require_prolongable_source_occurrence_bound(value)
        return _prepare_prolongable_substitution_input(value)

    @model_validator(mode="after")
    def require_growing_seed(self) -> Self:
        morphism = self.substitution.morphism
        if self.seed not in morphism.source_alphabet:
            raise _validation_error(
                "seed_outside_alphabet", "seed must belong to the substitution alphabet"
            )
        seed_index = morphism.source_alphabet.index(self.seed)
        seed_image = morphism.images[seed_index]
        if not seed_image or seed_image[0] != self.seed:
            raise _validation_error(
                "seed_image_not_prolongable", "the seed image must begin with the seed"
            )
        if len(seed_image) == 1:
            raise _validation_error(
                "seed_image_not_growing",
                "the seed image must contain a nonempty growing suffix",
            )
        image_map = dict(zip(morphism.source_alphabet, morphism.images, strict=True))
        mortal = {symbol for symbol, image in image_map.items() if not image}
        changed = True
        while changed:
            changed = False
            for symbol, image in image_map.items():
                if symbol not in mortal and all(letter in mortal for letter in image):
                    mortal.add(symbol)
                    changed = True
        if all(letter in mortal for letter in seed_image[1:]):
            raise _validation_error(
                "seed_suffix_eventually_erases",
                "the seed suffix must not eventually erase",
            )
        return self


def _require_prolongable_source_occurrence_bound(value: object) -> None:
    """Reject a fixed-prefix source before recursively proving prolongability."""

    substitution: object | None
    if isinstance(value, ProlongableSubstitution):
        substitution = value.substitution
    elif isinstance(value, Mapping):
        substitution = value.get("substitution")
    else:
        return

    morphism: object | None
    if isinstance(substitution, Substitution):
        morphism = substitution.morphism
    elif isinstance(substitution, Mapping):
        morphism = substitution.get("morphism")
    else:
        return

    if isinstance(morphism, WordMorphism):
        images: object = morphism.images
    elif isinstance(morphism, Mapping):
        images = morphism.get("images")
    else:
        return

    if not isinstance(images, (list, tuple)) or not all(
        isinstance(image, (list, tuple)) for image in images
    ):
        return
    occurrence_count = sum(len(image) for image in images)
    if occurrence_count > MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES:
        raise _validation_error(
            "fixed_point_source_occurrence_bound",
            "fixed-point source exceeds the aggregate occurrence bound "
            f"({occurrence_count} > {MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES})",
        )


def _prepare_prolongable_substitution_input(value: object) -> object:
    """Preserve JSON-array tuple decoding after the raw source preflight.

    Pydantic passes JSON arrays to model ``before`` validators as Python lists.
    The strict canonical models below require tuples, so prepare just the
    sequence fields owned by a prolongable substitution before nested parsing.
    """

    if not isinstance(value, Mapping):
        return value
    substitution = value.get("substitution")
    if not isinstance(substitution, Mapping):
        return value
    morphism = substitution.get("morphism")
    if not isinstance(morphism, Mapping):
        return value

    prepared_morphism = dict(morphism)
    for field_name in ("source_alphabet", "target_alphabet"):
        sequence = prepared_morphism.get(field_name)
        if isinstance(sequence, list):
            prepared_morphism[field_name] = tuple(sequence)
    images = prepared_morphism.get("images")
    if isinstance(images, list):
        prepared_morphism["images"] = tuple(
            tuple(image) if isinstance(image, list) else image for image in images
        )

    prepared_substitution = dict(substitution)
    prepared_substitution["morphism"] = prepared_morphism
    prepared = dict(value)
    prepared["substitution"] = prepared_substitution
    return prepared


class SubstitutionDependencyEdge(StrictModel):
    """One letter dependency with all occurrence positions in its source image."""

    source: Symbol
    target: Symbol
    multiplicity: int = Field(ge=1, le=MAX_MORPHISM_IMAGE_LENGTH)
    positions: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_MORPHISM_IMAGE_LENGTH
    )

    @model_validator(mode="after")
    def require_canonical_positions(self) -> Self:
        if self.positions != tuple(sorted(set(self.positions))):
            raise _validation_error(
                "dependency_positions_not_increasing",
                "dependency positions must be strictly increasing",
            )
        if self.positions[0] < 0:
            raise _validation_error(
                "dependency_positions_negative",
                "dependency positions must be nonnegative",
            )
        if self.multiplicity != len(self.positions):
            raise _validation_error(
                "dependency_multiplicity_mismatch",
                "dependency multiplicity must equal the position count",
            )
        return self


class SubstitutionDependencyGraph(StrictModel):
    """The exact alphabet-labelled dependency graph of one substitution."""

    substitution: Substitution
    edges: tuple[SubstitutionDependencyEdge, ...] = Field(
        max_length=MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE
    )

    @field_validator("substitution")
    @classmethod
    def require_bounded_source_before_edges(
        cls, substitution: Substitution
    ) -> Substitution:
        try:
            _require_dependency_occurrence_bound(substitution)
        except ValueError as error:
            raise _validation_error(
                "dependency_occurrence_bound", str(error)
            ) from error
        return substitution

    @model_validator(mode="after")
    def require_structural_edges(self) -> Self:
        """Validate graph shape without re-running the graph construction kernel."""

        try:
            _require_dependency_occurrence_bound(self.substitution)
        except ValueError as error:
            raise _validation_error(
                "dependency_occurrence_bound", str(error)
            ) from error
        morphism = self.substitution.morphism
        images = dict(zip(morphism.source_alphabet, morphism.images, strict=True))
        edge_pairs = tuple((edge.source, edge.target) for edge in self.edges)
        if len(edge_pairs) != len(set(edge_pairs)):
            raise _validation_error(
                "dependency_graph_duplicate_edge",
                "dependency graph may contain at most one edge per source-target pair",
            )
        for edge in self.edges:
            if edge.source not in images or edge.target not in morphism.target_alphabet:
                raise _validation_error(
                    "dependency_graph_endpoint",
                    "dependency edge endpoint is outside the alphabet",
                )
            image = images[edge.source]
            if any(
                position >= len(image) or image[position] != edge.target
                for position in edge.positions
            ):
                raise _validation_error(
                    "dependency_graph_position",
                    "dependency edge positions must point to its target in the source image",
                )
        return self


__all__ = [
    "MAX_ALPHABET_SIZE",
    "MAX_MORPHISM_IMAGE_LENGTH",
    "MAX_MORPHISM_OUTPUT_LENGTH",
    "MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES",
    "MAX_SUBSTITUTION_DEPENDENCY_OCCURRENCES",
    "MAX_SYMBOL_LENGTH",
    "MAX_WORD_LENGTH",
    "FiniteWord",
    "ProlongableSubstitution",
    "Substitution",
    "SubstitutionDependencyEdge",
    "SubstitutionDependencyGraph",
    "Symbol",
    "WordMorphism",
]
