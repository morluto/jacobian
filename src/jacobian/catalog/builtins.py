"""Deterministic discovery of owner-local built-in registrations."""

from __future__ import annotations

import pkgutil
from importlib import import_module

import jacobian.math
from jacobian.catalog.admission import (
    OperationAdmission,
    OperationRegistration,
    curate_public_tools,
)
from jacobian.catalog.models import MathTools


def _registration_module_names() -> tuple[str, ...]:
    prefix = f"{jacobian.math.__name__}."
    return tuple(
        sorted(
            module.name
            for module in pkgutil.walk_packages(jacobian.math.__path__, prefix)
            if module.name.endswith("._admission")
        )
    )


def _load_registrations(
    module_names: tuple[str, ...],
) -> tuple[OperationRegistration, ...]:
    registrations: list[OperationRegistration] = []
    for module_name in module_names:
        module = import_module(module_name)
        registration = getattr(module, "REGISTRATION", None)
        if not isinstance(registration, OperationRegistration):
            raise TypeError(
                f"{module_name} must export an OperationRegistration as REGISTRATION"
            )
        registrations.append(registration)
    return tuple(registrations)


_BUILTIN_REGISTRATION_MODULES = _registration_module_names()
_BUILTIN_REGISTRATIONS = _load_registrations(_BUILTIN_REGISTRATION_MODULES)
_BUILTIN_CANDIDATES: MathTools = tuple(
    tool for registration in _BUILTIN_REGISTRATIONS for tool in registration.candidates
)
_RAW_ADMISSIONS: tuple[OperationAdmission, ...] = tuple(
    admission
    for registration in _BUILTIN_REGISTRATIONS
    for admission in registration.admissions
)
_ALL_ADMISSIONS: tuple[OperationAdmission, ...] = tuple(
    sorted(_RAW_ADMISSIONS, key=lambda admission: admission.operation_id)
)

BUILTIN_TOOLS: MathTools = curate_public_tools(_BUILTIN_CANDIDATES, _ALL_ADMISSIONS)

__all__ = ["BUILTIN_TOOLS"]
