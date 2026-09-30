"""Bound and snapshot authored Koszul domain carriers before revalidation."""

from typing import Any, Never

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleChainMapMatrix,
    ModuleDifferential,
    ModuleKoszulChainMap,
    ModuleKoszulComplex,
)

_MAX_INPUT_CELLS = 8 * 1024 * 1024
_MODEL_TYPES = (
    FiniteCommutativeAlgebra,
    BasedFiniteModule,
    ModuleKoszulComplex,
    ModuleKoszulChainMap,
    ModuleDifferential,
    ModuleChainMapMatrix,
    CanonicalRational,
)


def _invalid(message: str) -> Never:
    raise OperationDomainValidationError(
        location=("arguments",), code="koszul.module.native_shape", message=message
    )


def _budget(message: str) -> Never:
    raise OperationResourceAdmissionError(
        location=("arguments",),
        code="koszul.module.native_shape_budget",
        message=message,
    )


class _Snapshot:
    """Request-local bounded copier for the finite Koszul carrier graph."""

    def __init__(self) -> None:
        request_checkpoint("before native Koszul argument snapshot")
        self.remaining = _MAX_INPUT_CELLS
        self.checks = 0

    def charge(self, count: int) -> None:
        self.remaining -= count
        self.checks += 1
        if self.checks % 256 == 0:
            request_checkpoint("during native Koszul argument snapshot")
        if self.remaining < 0:
            _budget("native Koszul arguments exceed the retained input envelope")

    def scalar(self, value: object, kind: type) -> Any:
        if type(value) is not kind:
            _invalid("native Koszul scalar has the wrong exact type")
        if kind is str:
            self.charge(len(value))  # type: ignore[arg-type]
        elif kind is int:
            bits = value.bit_length()  # type: ignore[attr-defined]
            if bits > 4 * MAX_CANONICAL_RATIONAL_DIGITS:
                _budget("native Koszul integer exceeds the canonical scalar envelope")
            # 2**3 < 10, so ceil(bits/3) bounds decimal magnitude without
            # converting the integer to text. Charge every retained occurrence.
            self.charge(max(1, (bits + 2) // 3))
        return value

    def axis(
        self, value: object, limits: tuple[int, ...], leaf: type
    ) -> tuple[Any, ...]:
        if type(value) is not tuple:
            _invalid("native Koszul axes must be exact tuples")
        if len(value) > limits[0]:
            _budget("native Koszul axis exceeds its declared dimension envelope")
        self.charge(1 + len(value))
        if len(limits) > 1:
            return tuple(self.axis(item, limits[1:], leaf) for item in value)
        return tuple(
            self.model(item, leaf) if leaf in _MODEL_TYPES else self.scalar(item, leaf)
            for item in value
        )

    def model(self, value: object, expected: type) -> dict[str, Any]:
        if type(value) is not expected or expected not in _MODEL_TYPES:
            _invalid("native Koszul arguments must use their exact domain carriers")
        self.charge(1)
        fields = value.__dict__  # exact model type, no user override
        if type(fields) is not dict:
            _invalid("native Koszul fields must be an exact dictionary")
        try:
            if expected is CanonicalRational:
                return {name: self.scalar(fields[name], int) for name in ("num", "den")}
            if expected is FiniteCommutativeAlgebra:
                return {
                    "basis": self.axis(fields["basis"], (6,), str),
                    "multiplication": self.axis(
                        fields["multiplication"], (6, 6, 6), CanonicalRational
                    ),
                    "unit": None
                    if fields["unit"] is None
                    else self.axis(fields["unit"], (6,), CanonicalRational),
                }
            if expected is BasedFiniteModule:
                return {
                    "algebra": self.model(fields["algebra"], FiniteCommutativeAlgebra),
                    "basis": self.axis(fields["basis"], (8,), str),
                    "action": self.axis(fields["action"], (6, 8, 8), CanonicalRational),
                }
            if expected in (ModuleDifferential, ModuleChainMapMatrix):
                rows, columns = (
                    self.scalar(fields["row_count"], int),
                    self.scalar(fields["column_count"], int),
                )
                if not 0 <= rows <= 256 or not 0 <= columns <= 256:
                    _budget(
                        "native Koszul matrix axes exceed the admitted basis envelope"
                    )
                entries = fields["entries"]
                if type(entries) is not tuple or len(entries) > rows * columns:
                    _invalid("native Koszul matrix entries exceed their declared axes")
                self.charge(1 + 3 * len(entries))
                copied = []
                for entry in entries:
                    if type(entry) is not tuple or len(entry) != 3:
                        _invalid("native Koszul matrix entries must be triples")
                    copied.append(
                        (
                            self.scalar(entry[0], int),
                            self.scalar(entry[1], int),
                            self.model(entry[2], CanonicalRational),
                        )
                    )
                return {
                    "row_count": rows,
                    "column_count": columns,
                    "entries": tuple(copied),
                }
            result = {
                "algebra": self.model(fields["algebra"], FiniteCommutativeAlgebra),
                "sequence": self.axis(fields["sequence"], (6, 6), CanonicalRational),
            }
            if expected is ModuleKoszulComplex:
                result.update(
                    module=self.model(fields["module"], BasedFiniteModule),
                    basis_sizes=self.axis(fields["basis_sizes"], (7,), int),
                    differentials=self.axis(
                        fields["differentials"], (6,), ModuleDifferential
                    ),
                    square_zero=self.scalar(fields["square_zero"], bool),
                )
            else:
                result.update(
                    source=self.model(fields["source"], BasedFiniteModule),
                    target=self.model(fields["target"], BasedFiniteModule),
                    module_map=self.axis(
                        fields["module_map"], (8, 8), CanonicalRational
                    ),
                    source_complex=self.model(
                        fields["source_complex"], ModuleKoszulComplex
                    ),
                    target_complex=self.model(
                        fields["target_complex"], ModuleKoszulComplex
                    ),
                    degree_maps=self.axis(
                        fields["degree_maps"], (7,), ModuleChainMapMatrix
                    ),
                )
            return result
        except (KeyError, AttributeError) as exc:
            raise OperationDomainValidationError(
                location=("arguments",),
                code="koszul.module.native_shape",
                message="native Koszul carrier is missing required fields",
            ) from exc


def native_payload(**arguments: object) -> dict[str, Any]:
    """Snapshot exact bounded domain carriers before kernel revalidation."""
    snapshot = _Snapshot()
    payload: dict[str, Any] = {}
    for name, value in arguments.items():
        if name == "sequence":
            payload[name] = snapshot.axis(value, (6, 6), CanonicalRational)
        elif name in ("map_matrix", "change_matrix"):
            payload[name] = snapshot.axis(
                value, (8, 8) if name == "map_matrix" else (6, 6), CanonicalRational
            )
        else:
            expected = {
                "algebra": FiniteCommutativeAlgebra,
                "source": BasedFiniteModule,
                "target": BasedFiniteModule,
                "complex": ModuleKoszulComplex,
                "chain_map": ModuleKoszulChainMap,
            }[name]
            payload[name] = snapshot.model(value, expected)
    return payload
