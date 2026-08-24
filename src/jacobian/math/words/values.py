"""Provider-independent values for bounded combinatorics on words."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import Field, field_validator, model_validator

from jacobian._models import StrictModel

MAX_WORD_LENGTH = 500
MAX_ALPHABET_SIZE = 50
MAX_SYMBOL_LENGTH = 64
MAX_MORPHISM_IMAGE_LENGTH = 10_000
MAX_MORPHISM_OUTPUT_LENGTH = MAX_WORD_LENGTH
MAX_SUBSTITUTION_DEPENDENCY_OCCURRENCES = 10_000
MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES = 20_000

Symbol = Annotated[str, Field(min_length=1, max_length=MAX_SYMBOL_LENGTH)]


class FiniteWord(StrictModel):
    alphabet: tuple[Symbol, ...] = Field(min_length=1, max_length=MAX_ALPHABET_SIZE)
    letters: tuple[str, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_word_over_ordered_alphabet(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet symbols must be distinct")
        if any(letter not in self.alphabet for letter in self.letters):
            raise ValueError("word letter is outside the declared alphabet")
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
            raise ValueError("source alphabet symbols must be distinct")
        if len(set(self.target_alphabet)) != len(self.target_alphabet):
            raise ValueError("target alphabet symbols must be distinct")
        if len(self.images) != len(self.source_alphabet):
            raise ValueError("morphism must have one image per source symbol")
        if any(len(image) > MAX_MORPHISM_IMAGE_LENGTH for image in self.images):
            raise ValueError("morphism image exceeds the length bound")
        if any(
            letter not in self.target_alphabet
            for image in self.images
            for letter in image
        ):
            raise ValueError("morphism image uses a symbol outside the target alphabet")
        return self


class Substitution(StrictModel):
    """A finite-alphabet endomorphism, retaining its canonical morphism."""

    morphism: WordMorphism

    @model_validator(mode="after")
    def require_endomorphism(self) -> Self:
        if self.morphism.source_alphabet != self.morphism.target_alphabet:
            raise ValueError(
                "a substitution must have identical source and target alphabets"
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
        _require_prolongable_source_occurrence_bound(value)
        return _prepare_prolongable_substitution_input(value)

    @model_validator(mode="after")
    def require_growing_seed(self) -> Self:
        morphism = self.substitution.morphism
        if self.seed not in morphism.source_alphabet:
            raise ValueError("seed must belong to the substitution alphabet")
        seed_index = morphism.source_alphabet.index(self.seed)
        seed_image = morphism.images[seed_index]
        if not seed_image or seed_image[0] != self.seed:
            raise ValueError("the seed image must begin with the seed")
        if len(seed_image) == 1:
            raise ValueError("the seed image must contain a nonempty growing suffix")
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
            raise ValueError("the seed suffix must not eventually erase")
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
        raise ValueError(
            "fixed-point source exceeds the aggregate occurrence bound "
            f"({occurrence_count} > {MAX_PROLONGABLE_SUBSTITUTION_SOURCE_OCCURRENCES})"
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
            raise ValueError("dependency positions must be strictly increasing")
        if self.positions[0] < 0:
            raise ValueError("dependency positions must be nonnegative")
        if self.multiplicity != len(self.positions):
            raise ValueError("dependency multiplicity must equal the position count")
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
        _require_dependency_occurrence_bound(substitution)
        return substitution

    @model_validator(mode="after")
    def bind_graph_to_substitution(self) -> Self:
        # Recheck before replay when Pydantic receives an existing model instance;
        # instance revalidation may not revisit the field validator above.
        _require_dependency_occurrence_bound(self.substitution)
        morphism = self.substitution.morphism
        expected = tuple(
            SubstitutionDependencyEdge(
                source=source,
                target=target,
                multiplicity=image.count(target),
                positions=tuple(
                    position
                    for position, letter in enumerate(image)
                    if letter == target
                ),
            )
            for source, image in zip(
                morphism.source_alphabet, morphism.images, strict=True
            )
            for target in morphism.target_alphabet
            if target in image
        )
        if self.edges != expected:
            raise ValueError("dependency graph is not bound to the substitution")
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
