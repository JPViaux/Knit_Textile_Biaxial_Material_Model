"""Discover material models and pack named parameters for their UMATs."""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (CODE_ROOT as RELEASE_CODE_ROOT)

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (CODE_ROOT as RELEASE_CODE_ROOT,
    UTILITIES_ROOT as RELEASE_UTILITIES_ROOT, INPUT_ROOT as RELEASE_INPUT_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT, EXPERIMENT_ROOT as RELEASE_EXPERIMENT_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT, FIGURE_ROOT as RELEASE_FIGURE_ROOT)

from collections.abc import Mapping
from dataclasses import dataclass
from release_paths import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = RELEASE_CODE_ROOT / '1_Material_model'
MODEL_SPEC_FILENAME = "model.json"
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_PARAMETER_TYPES = frozenset({"float", "int"})


class ModelSpecificationError(ValueError):
    """Raised when a model specification or parameter mapping is invalid."""


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata for one entry in a UMAT ``PROPS`` vector."""

    name: str
    data_type: str
    units: str
    description: str


@dataclass(frozen=True)
class ModelSpec:
    """Validated metadata and parameter interface for one material model."""

    schema_version: int
    model_id: str
    display_name: str
    description: str
    fortran_source: Path
    analysis_family: str | None
    parent_model_id: str | None
    parameters: tuple[ParameterSpec, ...]
    specification_path: Path

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return parameter names in the UMAT's required ``PROPS`` order."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def nprops(self) -> int:
        """Return the number of entries expected by the UMAT."""
        return len(self.parameters)

    def validate_parameters(self, values: Mapping[str, Any]) -> None:
        """Validate that named values exactly satisfy this model's interface."""
        if not isinstance(values, Mapping):
            raise ModelSpecificationError(
                f"Parameters for {self.model_id} must be supplied as a mapping"
            )

        expected = self.parameter_names
        expected_set = set(expected)
        actual_set = set(values)
        missing = [name for name in expected if name not in actual_set]
        extra = sorted(str(name) for name in actual_set - expected_set)

        problems: list[str] = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected: {', '.join(extra)}")
        if problems:
            raise ModelSpecificationError(
                f"Parameters are incompatible with {self.model_id}; "
                + "; ".join(problems)
            )

        invalid: list[str] = []
        for name in expected:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                invalid.append(f"{name}={value!r} is not numeric")
            elif not math.isfinite(float(value)):
                invalid.append(f"{name}={value!r} is not finite")
        if invalid:
            raise ModelSpecificationError(
                f"Invalid parameters for {self.model_id}: " + "; ".join(invalid)
            )

    def pack_parameters(self, values: Mapping[str, Any]) -> list[float]:
        """Return named values in the exact ``PROPS`` order required by the UMAT."""
        self.validate_parameters(values)
        return [float(values[name]) for name in self.parameter_names]


def discover_model_specs(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, ModelSpec]:
    """Load all model specifications found directly below ``model_dir``."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Material-model directory was not found: {model_dir}")

    models: dict[str, ModelSpec] = {}
    for specification_path in sorted(list(model_dir.glob(f"*/{MODEL_SPEC_FILENAME}")) + list(model_dir.glob(MODEL_SPEC_FILENAME))):
        model = load_model_spec_file(specification_path)
        if model.model_id in models:
            other = models[model.model_id].specification_path
            raise ModelSpecificationError(
                f"Duplicate model_id {model.model_id!r} in {other} and "
                f"{specification_path}"
            )
        models[model.model_id] = model
    return models


def available_model_ids(model_dir: Path = DEFAULT_MODEL_DIR) -> tuple[str, ...]:
    """Return the identifiers of all discoverable material models."""
    return tuple(discover_model_specs(model_dir))


def load_model_spec(model_id: str, model_dir: Path = DEFAULT_MODEL_DIR) -> ModelSpec:
    """Load a model by its stable identifier."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    specification_path = (Path(model_dir) / MODEL_SPEC_FILENAME if model_id == "SV_exp_polycon_mo" and (Path(model_dir) / MODEL_SPEC_FILENAME).is_file() else Path(model_dir) / model_id / MODEL_SPEC_FILENAME)
    if not specification_path.is_file():
        available = ", ".join(available_model_ids(model_dir)) or "none"
        raise FileNotFoundError(
            f"Unknown material model {model_id!r}; available models: {available}"
        )

    model = load_model_spec_file(specification_path)
    if model.model_id != model_id:
        raise ModelSpecificationError(
            f"Requested model {model_id!r}, but {specification_path} declares "
            f"{model.model_id!r}"
        )
    return model


def load_model_spec_file(specification_path: Path) -> ModelSpec:
    """Load and validate one ``model.json`` file."""
    specification_path = Path(specification_path).resolve()
    try:
        with specification_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ModelSpecificationError(
            f"Invalid JSON in model specification {specification_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ModelSpecificationError(
            f"Model specification must contain a JSON object: {specification_path}"
        )

    schema_version = _required_integer(payload, "schema_version", specification_path)
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ModelSpecificationError(
            f"Unsupported schema_version {schema_version} in {specification_path}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    model_id = _required_string(payload, "model_id", specification_path)
    display_name = _required_string(payload, "display_name", specification_path)
    description = _required_string(payload, "description", specification_path)
    source_name = _required_string(payload, "fortran_source", specification_path)

    source_relative = Path(source_name)
    if source_relative.is_absolute():
        raise ModelSpecificationError(
            f"fortran_source must be relative in {specification_path}"
        )
    fortran_source = (specification_path.parent / source_relative).resolve()
    if not fortran_source.is_file():
        raise ModelSpecificationError(
            f"Fortran source declared by {specification_path} was not found: "
            f"{fortran_source}"
        )

    analysis_family = _optional_string(payload, "analysis_family", specification_path)
    parent_model_id = _optional_string(payload, "parent_model_id", specification_path)
    parameters = _load_parameter_specs(payload.get("parameters"), specification_path)

    return ModelSpec(
        schema_version=schema_version,
        model_id=model_id,
        display_name=display_name,
        description=description,
        fortran_source=fortran_source,
        analysis_family=analysis_family,
        parent_model_id=parent_model_id,
        parameters=parameters,
        specification_path=specification_path,
    )


def pack_parameters(
    model_id: str,
    values: Mapping[str, Any],
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> list[float]:
    """Convenience wrapper that loads a model and packs its named values."""
    return load_model_spec(model_id, model_dir).pack_parameters(values)


def _load_parameter_specs(
    raw_parameters: Any,
    specification_path: Path,
) -> tuple[ParameterSpec, ...]:
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ModelSpecificationError(
            f"parameters must be a non-empty array in {specification_path}"
        )

    parameters: list[ParameterSpec] = []
    names: set[str] = set()
    for position, raw_parameter in enumerate(raw_parameters, start=1):
        location = f"parameter {position} in {specification_path}"
        if not isinstance(raw_parameter, dict):
            raise ModelSpecificationError(f"{location} must be an object")

        name = _required_string(raw_parameter, "name", location)
        if name in names:
            raise ModelSpecificationError(
                f"Duplicate parameter name {name!r} in {specification_path}"
            )
        names.add(name)

        data_type = _required_string(raw_parameter, "type", location)
        if data_type not in SUPPORTED_PARAMETER_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_PARAMETER_TYPES))
            raise ModelSpecificationError(
                f"Unsupported type {data_type!r} for {location}; use {allowed}"
            )

        parameters.append(
            ParameterSpec(
                name=name,
                data_type=data_type,
                units=_required_string(raw_parameter, "units", location),
                description=_required_string(raw_parameter, "description", location),
            )
        )
    return tuple(parameters)


def _required_string(payload: Mapping[str, Any], key: str, location: Any) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelSpecificationError(
            f"{key} must be a non-empty string in {location}"
        )
    return value.strip()


def _optional_string(
    payload: Mapping[str, Any],
    key: str,
    location: Any,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelSpecificationError(
            f"{key} must be null or a non-empty string in {location}"
        )
    return value.strip()


def _required_integer(payload: Mapping[str, Any], key: str, location: Any) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelSpecificationError(f"{key} must be an integer in {location}")
    return value
