"""Deterministic separable least-squares solver for compatible SV models.

The solver leaves the existing simulated-annealing implementation untouched.
For fixed ``beta_1``, ``gamma_1``, and ``zeta``, it obtains unit stress-response
columns from the compiled UMAT and solves the six linear coefficients with a
bounded convex least-squares calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import os
from typing import Any, Callable, Mapping, Sequence

# Use one compatible OpenMP runtime when the optional Windows SCIP wheel is
# loaded after NumPy/SciPy. This does not change deterministic solver APIs.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear, minimize


FloatArray = NDArray[np.float64]
ProgressCallback = Callable[[int, "ProfiledSolution"], None]


@dataclass(frozen=True)
class AffineDerivedParameter:
    """Derive one material parameter, for example ``a = 1 - b``."""

    target: str
    offset: float
    coefficients: tuple[tuple[str, float], ...]

    def evaluate(self, values: Mapping[str, float | int]) -> float:
        """Evaluate the affine rule from supplied independent shape values."""
        return float(
            self.offset
            + sum(float(weight) * float(values[name]) for name, weight in self.coefficients)
        )


@dataclass(frozen=True)
class RotatedConeConstraint:
    """Require ``factor*x*y >= z**2`` for three linear coefficients."""

    first: str
    second: str
    cross: str
    factor: float = 4.0
    relative_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class SeparableModelDefinition:
    """Declare which model parameters are linear and which define shape."""

    model_id: str
    linear_parameters: tuple[str, ...]
    continuous_shape_parameters: tuple[str, ...]
    integer_shape_parameters: tuple[str, ...]
    # Map each coefficient to the shape values used by its basis. For example,
    # ("K_1ani1", ("beta_1",)) ties that coefficient to beta_1.
    basis_shape_parameters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Derived parameters remain in the UMAT interface but are not independently
    # searched. Example: target="a", offset=1, coefficients=(("b", -1),).
    derived_parameters: tuple[AffineDerivedParameter, ...] = ()
    # Convex restrictions on the profiled linear coefficients.
    rotated_cone_constraints: tuple[RotatedConeConstraint, ...] = ()
    # Numerical coordinates for generic grids/refinement: "log" or "linear".
    continuous_shape_scales: tuple[tuple[str, str], ...] = ()
    # Select the numerical backend. Examples: "grid_powell" or "scip_sos1".
    backend: str = "grid_powell"
    # Name of the coupling exponent used by grid manifests; for example
    # ``zeta`` in SV_exp_polycon or ``xi`` in SV_pl_polycon.
    coupling_exponent_parameter: str = "zeta"

    def basis_dependencies(self) -> dict[str, tuple[str, ...]]:
        """Return the shape dependencies for every declared linear coefficient."""
        explicit = dict(self.basis_shape_parameters)
        conventional = {
            "K_1ani1": ("beta_1",),
            "K_2ani1": ("beta_2",),
            "K_1ani2": ("gamma_1",),
            "K_2ani2": ("gamma_2",),
            "K_coup": ("zeta",),
        }
        dependencies: dict[str, tuple[str, ...]] = {}
        for name in self.linear_parameters:
            dependencies[name] = tuple(explicit.get(name, conventional.get(name, ())))
        unknown = sorted(set(explicit) - set(self.linear_parameters))
        if unknown:
            raise ValueError(
                "Basis dependencies reference non-linear parameters: "
                + ", ".join(unknown)
            )
        return dependencies

    @property
    def shape_parameters(self) -> tuple[str, ...]:
        """Return all continuous and integer shapes in their declared order."""
        return self.continuous_shape_parameters + self.integer_shape_parameters

    @property
    def derived_parameter_names(self) -> tuple[str, ...]:
        """Return derived UMAT parameter names, for example ``("a",)``."""
        return tuple(rule.target for rule in self.derived_parameters)

    def shape_scale(self, name: str) -> str:
        """Return a search coordinate, for example ``beta_1 -> log``."""
        if name not in self.continuous_shape_parameters:
            raise ValueError(f"{name} is not a continuous shape parameter")
        explicit = dict(self.continuous_shape_scales)
        scale = explicit.get(name, "linear")
        if scale not in {"linear", "log"}:
            raise ValueError(f"Unsupported shape scale {scale!r} for {name}")
        return scale


# SV_exp and SV_tanh share six stress magnitudes and three shape parameters.
SV_EXP_DEFINITION = SeparableModelDefinition(
    model_id="SV_exp",
    linear_parameters=(
        "mu",
        "mu2",
        "mu3",
        "K_1ani1",
        "K_1ani2",
        "K_coup",
    ),
    continuous_shape_parameters=("beta_1", "gamma_1"),
    integer_shape_parameters=("zeta",),
    basis_shape_parameters=(
        ("K_1ani1", ("beta_1",)),
        ("K_1ani2", ("gamma_1",)),
        ("K_coup", ("zeta",)),
    ),
    continuous_shape_scales=(("beta_1", "log"), ("gamma_1", "log")),
)

SV_EXP_POLYCON_DEFINITION = SeparableModelDefinition(
    model_id="SV_exp_polycon",
    linear_parameters=SV_EXP_DEFINITION.linear_parameters,
    continuous_shape_parameters=("beta_1", "gamma_1", "b"),
    integer_shape_parameters=("zeta",),
    basis_shape_parameters=(
        ("K_1ani1", ("beta_1",)),
        ("K_1ani2", ("gamma_1",)),
        ("K_coup", ("b", "zeta")),
    ),
    derived_parameters=(
        AffineDerivedParameter(
            target="a",
            offset=1.0,
            coefficients=(("b", -1.0),),
        ),
    ),
    rotated_cone_constraints=(
        RotatedConeConstraint(first="mu", second="mu2", cross="mu3"),
    ),
    continuous_shape_scales=(
        ("beta_1", "log"),
        ("gamma_1", "log"),
        ("b", "linear"),
    ),
)

# The combined-max variant has the same separable parameter partition as
# SV_exp_polycon; only the compiled UMAT basis response differs.
SV_EXP_POLYCON_MO_DEFINITION = SeparableModelDefinition(
    model_id="SV_exp_polycon_mo",
    linear_parameters=SV_EXP_POLYCON_DEFINITION.linear_parameters,
    continuous_shape_parameters=SV_EXP_POLYCON_DEFINITION.continuous_shape_parameters,
    integer_shape_parameters=SV_EXP_POLYCON_DEFINITION.integer_shape_parameters,
    basis_shape_parameters=SV_EXP_POLYCON_DEFINITION.basis_shape_parameters,
    derived_parameters=SV_EXP_POLYCON_DEFINITION.derived_parameters,
    rotated_cone_constraints=SV_EXP_POLYCON_DEFINITION.rotated_cone_constraints,
    continuous_shape_scales=SV_EXP_POLYCON_DEFINITION.continuous_shape_scales,
)

# This model keeps the combined-max equations and constraints unchanged while
# refining the coupling exponent continuously (for example, xi=6.25).
SV_EXP_PC_MO_XI_REAL_DEFINITION = SeparableModelDefinition(
    model_id="SV_exp_pc_mo_xi_real",
    linear_parameters=SV_EXP_POLYCON_MO_DEFINITION.linear_parameters,
    continuous_shape_parameters=("beta_1", "gamma_1", "b", "xi"),
    integer_shape_parameters=(),
    basis_shape_parameters=(
        ("K_1ani1", ("beta_1",)),
        ("K_1ani2", ("gamma_1",)),
        ("K_coup", ("b", "xi")),
    ),
    derived_parameters=SV_EXP_POLYCON_MO_DEFINITION.derived_parameters,
    rotated_cone_constraints=SV_EXP_POLYCON_MO_DEFINITION.rotated_cone_constraints,
    continuous_shape_scales=(
        ("beta_1", "log"),
        ("gamma_1", "log"),
        ("b", "linear"),
        ("xi", "linear"),
    ),
    coupling_exponent_parameter="xi",
)

SV_PL_POLYCON_DEFINITION = SeparableModelDefinition(
    model_id="SV_pl_polycon",
    linear_parameters=(
        "mu",
        "mu2",
        "mu3",
        "K_course",
        "K_wale",
        "K_coup",
    ),
    continuous_shape_parameters=("b",),
    integer_shape_parameters=("beta", "gamma", "xi"),
    basis_shape_parameters=(
        ("K_course", ("beta",)),
        ("K_wale", ("gamma",)),
        ("K_coup", ("b", "xi")),
    ),
    derived_parameters=(
        AffineDerivedParameter(
            target="a",
            offset=1.0,
            coefficients=(("b", -1.0),),
        ),
    ),
    rotated_cone_constraints=(
        RotatedConeConstraint(first="mu", second="mu2", cross="mu3"),
    ),
    continuous_shape_scales=(("b", "linear"),),
    coupling_exponent_parameter="xi",
)

SV_TANH_DEFINITION = SeparableModelDefinition(
    model_id="SV_tanh",
    linear_parameters=SV_EXP_DEFINITION.linear_parameters,
    continuous_shape_parameters=SV_EXP_DEFINITION.continuous_shape_parameters,
    integer_shape_parameters=SV_EXP_DEFINITION.integer_shape_parameters,
    basis_shape_parameters=SV_EXP_DEFINITION.basis_shape_parameters,
)

STAPLETON_VIAUX_DEFINITION = SeparableModelDefinition(
    model_id="StapletonViaux",
    linear_parameters=(
        "mu",
        "mu2",
        "mu3",
        "K_1ani1",
        "K_2ani1",
        "K_1ani2",
        "K_2ani2",
        "K_coup",
    ),
    continuous_shape_parameters=(),
    integer_shape_parameters=(
        "beta_1",
        "beta_2",
        "gamma_1",
        "gamma_2",
        "zeta",
    ),
    basis_shape_parameters=(
        ("K_1ani1", ("beta_1",)),
        ("K_2ani1", ("beta_2",)),
        ("K_1ani2", ("gamma_1",)),
        ("K_2ani2", ("gamma_2",)),
        ("K_coup", ("zeta",)),
    ),
    backend="scip_sos1",
)

# Stable manifest identifiers keep the constitutive model and solver partition paired.
SEPARABLE_DEFINITIONS = {"sv_exp_polycon_mo_v1": SV_EXP_POLYCON_MO_DEFINITION}


def load_separable_definition(
    definition_id: str,
    model_id: str | None = None,
) -> SeparableModelDefinition:
    """Return a registered definition, for example ``sv_tanh_v1``."""
    try:
        definition = SEPARABLE_DEFINITIONS[str(definition_id)]
    except KeyError as exc:
        available = ", ".join(sorted(SEPARABLE_DEFINITIONS))
        raise ValueError(
            f"Unknown separable definition {definition_id!r}; available: {available}"
        ) from exc
    if model_id is not None and definition.model_id != model_id:
        raise ValueError(
            f"Separable definition {definition_id!r} belongs to "
            f"{definition.model_id}, not {model_id}"
        )
    return definition


@dataclass(frozen=True)
class WeightedFittingProblem:
    """Store the stacked experimental target and its fixed cost scales."""

    axes: tuple[int, ...]
    target: FloatArray
    scales: FloatArray
    weighted_target: FloatArray


@dataclass(frozen=True)
class ProfiledSolution:
    """Store the exact bounded-linear solution at one set of shape values."""

    beta_1: float
    gamma_1: float
    zeta: int
    cost: float
    mean_cost: float
    linear_values: dict[str, float]
    active_bounds: dict[str, int]
    optimality: float
    iterations: int
    # Store additional shapes for models such as Stapleton-Viaux. Existing
    # beta_1/gamma_1/zeta attributes remain the compatibility interface.
    shape_values: dict[str, float | int] = field(default_factory=dict)
    # Derived values are recorded separately so provenance distinguishes them
    # from independently searched shapes; for example a=0.4 when b=0.6.
    derived_shape_values: dict[str, float] = field(default_factory=dict)
    linear_solver_backend: str = "bvls"
    constraint_margins: dict[str, float] = field(default_factory=dict)
    constraint_tolerances: dict[str, float] = field(default_factory=dict)
    effective_linear_upper_bounds: dict[str, float] = field(default_factory=dict)

    def complete_shape_values(self) -> dict[str, float | int]:
        """Return all shapes, including the legacy beta/gamma/zeta fields."""
        # New definitions can use names such as beta/gamma/xi. Do not leak
        # zero-valued legacy aliases into those complete parameter dictionaries.
        values: dict[str, float | int] = {}
        for name, value in (
            ("beta_1", self.beta_1),
            ("gamma_1", self.gamma_1),
            ("zeta", self.zeta),
        ):
            if name in self.shape_values or value != 0:
                values[name] = value
        values.update(self.shape_values)
        values.update(self.derived_shape_values)
        return values


@dataclass(frozen=True)
class ShapeGridSearchResult:
    """Store a generic continuous-grid map grouped by integer shapes."""

    continuous_names: tuple[str, ...]
    continuous_grids: dict[str, FloatArray]
    integer_names: tuple[str, ...]
    integer_combinations: tuple[tuple[int, ...], ...]
    costs: FloatArray
    best_by_integer_combination: tuple[ProfiledSolution, ...]
    best_solution: ProfiledSolution
    # Rejected constrained points remain explicit instead of being accepted or
    # aborting an otherwise feasible grid; for example one failed SLSQP point.
    rejected_evaluations: tuple[dict[str, float | int | str], ...] = ()


@dataclass(frozen=True)
class SolutionValidation:
    """Compare the profiled matrix prediction with a direct full-UMAT call."""

    direct_cost: float
    existing_wrapper_cost: float
    profiled_cost: float
    absolute_cost_difference: float
    existing_wrapper_cost_difference: float
    maximum_weighted_prediction_difference: float
    design_matrix_rank: int
    design_matrix_singular_values: tuple[float, ...]


@dataclass(frozen=True)
class GridSearchResult:
    """Store a deterministic beta/gamma map for every enumerated zeta value."""

    beta_values: FloatArray
    gamma_values: FloatArray
    zeta_values: tuple[int, ...]
    costs: FloatArray
    best_by_zeta: tuple[ProfiledSolution, ...]
    best_solution: ProfiledSolution


def _stack_axes(values: FloatArray, axes: Sequence[int]) -> FloatArray:
    """Stack selected Voigt columns in the same axis-major order as the cost."""
    return np.concatenate([np.asarray(values[:, axis], dtype=float) for axis in axes])


def apply_parameter_specs_in_memory(
    umat_wrapper: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Apply saved fit bounds and guesses without rewriting the source ``.dat``.

    Each specification follows the fit-artifact schema, for example
    ``{"type": "float", "initial_guess": 0.1, "minimum": 0, ...}``.
    The operation is validated completely before any wrapper value is changed.
    """
    current_parameters = umat_wrapper.get_props_dic()
    current_names = set(current_parameters)
    supplied_names = set(parameter_specs)
    missing = sorted(current_names - supplied_names)
    extra = sorted(supplied_names - current_names)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError("Parameter specifications do not match the model (" + "; ".join(details) + ")")

    validated_records: dict[str, tuple[Any, ...]] = {}
    type_lookup = {"float": float, "int": int}
    for name in current_parameters:
        specification = parameter_specs[name]
        required_fields = {"type", "initial_guess", "minimum", "maximum", "fixed"}
        missing_fields = sorted(required_fields - set(specification))
        if missing_fields:
            raise ValueError(
                f"Parameter {name} is missing specification fields: "
                + ", ".join(missing_fields)
            )

        type_name = str(specification["type"]).lower()
        if type_name not in type_lookup:
            raise ValueError(f"Parameter {name} has unsupported type {type_name!r}")
        data_type = type_lookup[type_name]
        lower = float(specification["minimum"])
        upper = float(specification["maximum"])
        initial_numeric = float(specification["initial_guess"])
        if not np.all(np.isfinite([lower, upper, initial_numeric])):
            raise ValueError(f"Parameter {name} contains a non-finite saved value")
        if upper < lower:
            raise ValueError(f"Parameter {name} has reversed bounds [{lower}, {upper}]")
        if initial_numeric < lower or initial_numeric > upper:
            raise ValueError(
                f"Parameter {name} initial guess {initial_numeric} is outside "
                f"[{lower}, {upper}]"
            )

        if data_type is int:
            integer_values = (initial_numeric, lower, upper)
            if any(not float(value).is_integer() for value in integer_values):
                raise ValueError(f"Integer parameter {name} has a non-integer saved value")
            initial_value: float | int = int(initial_numeric)
            stored_lower: float | int = int(lower)
            stored_upper: float | int = int(upper)
        else:
            initial_value = float(initial_numeric)
            stored_lower = float(lower)
            stored_upper = float(upper)

        fixed = specification["fixed"]
        if not isinstance(fixed, bool):
            raise ValueError(f"Parameter {name} fixed flag must be Boolean")
        counter = current_parameters[name][4]
        validated_records[name] = (
            initial_value,
            data_type,
            stored_lower,
            stored_upper,
            counter,
            fixed,
        )

    for name, record in validated_records.items():
        umat_wrapper.setv(name, *record)


class SeparableMaterialSolver:
    """Profile six linear coefficients for a registered separable SV model."""

    def __init__(
        self,
        umat_wrapper: Any,
        fit_axes: Sequence[int] = (0, 1),
        *,
        definition: SeparableModelDefinition = SV_EXP_DEFINITION,
        linear_method: str = "bvls",
        linear_tolerance: float = 1.0e-12,
        linear_max_iterations: int = 100,
        inactive_linear_parameters: Sequence[str] = (),
    ) -> None:
        """Create a solver around an already configured and loaded UMAT wrapper.

        ``fit_axes`` uses Voigt indices, for example ``(0, 1)`` for xx and yy.
        ``linear_method`` may be ``"bvls"`` or ``"trf"`` in SciPy.
        ``linear_max_iterations`` may be 100 for ordinary six-column solves.
        """
        self.wrapper = umat_wrapper
        self.definition = definition
        self.fit_axes = self._validate_axes(fit_axes)
        self.linear_method = linear_method
        self.linear_tolerance = float(linear_tolerance)
        requested_inactive = tuple(str(name) for name in inactive_linear_parameters)
        if len(set(requested_inactive)) != len(requested_inactive):
            raise ValueError("Inactive linear parameter names cannot repeat")
        unknown_inactive = sorted(
            set(requested_inactive) - set(definition.linear_parameters)
        )
        if unknown_inactive:
            raise ValueError(
                "Inactive parameters are not separable linear coefficients: "
                + ", ".join(unknown_inactive)
            )
        dependencies = definition.basis_dependencies()
        dependency_free = sorted(
            name for name in requested_inactive if not dependencies[name]
        )
        if dependency_free:
            raise ValueError(
                "Inactive parameters must be nonlinear basis families: "
                + ", ".join(dependency_free)
            )
        inactive_set = set(requested_inactive)
        # Keep both sets in canonical model order; for example, the certified
        # Stapleton solve omits K_2ani1 and K_2ani2 from its optimization design.
        self.inactive_linear_parameters = tuple(
            name for name in definition.linear_parameters if name in inactive_set
        )
        self.active_linear_parameters = tuple(
            name for name in definition.linear_parameters if name not in inactive_set
        )
        self.inactive_shape_parameters = tuple(
            shape
            for name in self.inactive_linear_parameters
            for shape in dependencies[name]
        )
        if isinstance(linear_max_iterations, bool) or int(linear_max_iterations) <= 0:
            raise ValueError("Linear maximum iterations must be a positive integer")
        self.linear_max_iterations = int(linear_max_iterations)
        self._validate_model_and_parameters()
        self.problem = self._build_weighted_problem()

        # Parameter bounds are read from the selected .dat file; for example,
        # Rib SV_exp currently uses mu in [0, 0.05] and K_coup in [0, 15].
        self.linear_lower_bounds = np.array(
            [float(self.wrapper.getv(name)[2]) for name in definition.linear_parameters],
            dtype=float,
        )
        self.linear_upper_bounds = np.array(
            [float(self.wrapper.getv(name)[3]) for name in definition.linear_parameters],
            dtype=float,
        )
        self._validate_linear_bounds()
        self._validate_derived_parameters_and_constraints()

        # The generic cache key is (coefficient, controlling shape values). An
        # example is ("K_1ani1", (("beta_1", 1.0),)).
        self._basis_columns: dict[
            tuple[str, tuple[tuple[str, float | int], ...]], FloatArray
        ] = {}
        # Retain the named caches as compatibility diagnostics for SV_exp/tanh.
        self._constant_columns: dict[str, FloatArray] = {}
        self._beta_columns: dict[float, FloatArray] = {}
        self._gamma_columns: dict[float, FloatArray] = {}
        self._zeta_columns: dict[int, FloatArray] = {}
        # Count inner bounded solves; for example, a 21x21x9 map starts at 3969.
        self._profiled_evaluation_count = 0
        self._linear_backend_counts = {"bvls": 0, "slsqp_soc": 0}
        self._minimum_constraint_margins: dict[str, float] = {}

    @staticmethod
    def _validate_axes(axes: Sequence[int]) -> tuple[int, ...]:
        """Return unique valid Voigt axes, for example ``(0, 1, 3)``."""
        validated = tuple(int(axis) for axis in axes)
        if not validated:
            raise ValueError("At least one fitted stress component is required")
        if len(set(validated)) != len(validated):
            raise ValueError(f"Fitted stress components must be unique: {validated}")
        if any(axis < 0 or axis > 5 for axis in validated):
            raise ValueError(f"Fitted stress components must be between 0 and 5: {validated}")
        return validated

    def _validate_model_and_parameters(self) -> None:
        """Validate active/free and inactive/fixed-zero parameter contracts."""
        model_spec = getattr(self.wrapper.get_model(), "model_spec", None)
        model_id = getattr(model_spec, "model_id", None)
        if model_id is not None and model_id != self.definition.model_id:
            raise ValueError(
                f"The separable definition supports {self.definition.model_id}, not {model_id}"
            )

        parameter_names = set(self.wrapper.get_props_dic())
        required = set(
            self.definition.linear_parameters
            + self.definition.continuous_shape_parameters
            + self.definition.integer_shape_parameters
            + self.definition.derived_parameter_names
        )
        missing = sorted(required - parameter_names)
        if missing:
            raise ValueError(f"Missing separable-solver parameters: {', '.join(missing)}")

        inactive = set(self.inactive_linear_parameters)
        free_inactive = [
            name for name in self.inactive_linear_parameters if not self.wrapper.getv(name)[5]
        ]
        if free_inactive:
            raise ValueError(
                "Inactive linear parameters must be fixed: "
                + ", ".join(free_inactive)
            )
        nonzero_inactive = [
            name
            for name in self.inactive_linear_parameters
            if not np.isclose(float(self.wrapper.getv(name)[0]), 0.0)
        ]
        if nonzero_inactive:
            raise ValueError(
                "Inactive linear parameters must equal zero: "
                + ", ".join(nonzero_inactive)
            )
        free_inactive_shapes = [
            name
            for name in self.inactive_shape_parameters
            if not self.wrapper.getv(name)[5]
        ]
        if free_inactive_shapes:
            raise ValueError(
                "Shapes belonging to inactive linear parameters must be fixed: "
                + ", ".join(free_inactive_shapes)
            )
        fixed_linear = [
            name
            for name in self.definition.linear_parameters
            if self.wrapper.getv(name)[5] and name not in inactive
        ]
        if fixed_linear:
            raise ValueError(
                "The separable solver requires its linear coefficients to be free; "
                f"fixed parameters: {', '.join(fixed_linear)}"
            )

    def _validate_derived_parameters_and_constraints(self) -> None:
        """Validate relation inputs, target bounds, and convex-cone contracts."""
        independent = set(self.definition.shape_parameters)
        linear = set(self.definition.linear_parameters)
        targets = self.definition.derived_parameter_names
        if len(set(targets)) != len(targets):
            raise ValueError("Derived parameter targets cannot repeat")
        for rule in self.definition.derived_parameters:
            sources = tuple(name for name, _ in rule.coefficients)
            if not sources or len(set(sources)) != len(sources):
                raise ValueError(f"Derived parameter {rule.target} requires unique sources")
            unknown = sorted(set(sources) - independent)
            if unknown:
                raise ValueError(
                    f"Derived parameter {rule.target} references unknown shapes: "
                    + ", ".join(unknown)
                )
            if rule.target in independent or rule.target in linear:
                raise ValueError(f"Derived target {rule.target} cannot also be optimized")

            # Affine extrema over a box occur at source endpoints. Confirm that
            # the entire derived interval fits inside the target's .dat bounds.
            derived_min = float(rule.offset)
            derived_max = float(rule.offset)
            for source, weight in rule.coefficients:
                lower, upper = self.parameter_bounds(source)
                contributions = (float(weight) * lower, float(weight) * upper)
                derived_min += min(contributions)
                derived_max += max(contributions)
            target_lower, target_upper = self.parameter_bounds(rule.target)
            tolerance = 16.0 * np.finfo(float).eps * max(
                1.0, abs(target_lower), abs(target_upper)
            )
            if derived_min < target_lower - tolerance or derived_max > target_upper + tolerance:
                raise ValueError(
                    f"Derived {rule.target} interval [{derived_min}, {derived_max}] "
                    f"exceeds [{target_lower}, {target_upper}]"
                )

        for constraint in self.definition.rotated_cone_constraints:
            names = (constraint.first, constraint.second, constraint.cross)
            missing = sorted(set(names) - linear)
            if missing:
                raise ValueError(
                    "Rotated-cone constraints reference non-linear parameters: "
                    + ", ".join(missing)
                )
            if constraint.factor <= 0 or constraint.relative_tolerance <= 0:
                raise ValueError("Rotated-cone factors and tolerances must be positive")
            first_index = self.definition.linear_parameters.index(constraint.first)
            second_index = self.definition.linear_parameters.index(constraint.second)
            cross_index = self.definition.linear_parameters.index(constraint.cross)
            if (
                self.linear_lower_bounds[first_index] < 0
                or self.linear_lower_bounds[second_index] < 0
            ):
                raise ValueError("Rotated-cone diagonal coefficients must be nonnegative")
            if not (
                self.linear_lower_bounds[cross_index]
                <= 0
                <= self.linear_upper_bounds[cross_index]
            ):
                raise ValueError("Rotated-cone cross-coefficient bounds must contain zero")

    def _validate_linear_bounds(self) -> None:
        """Require finite, nonzero-width bounds that contain the zero reference."""
        if not np.all(np.isfinite(self.linear_lower_bounds)) or not np.all(
            np.isfinite(self.linear_upper_bounds)
        ):
            raise ValueError("Linear parameter bounds must be finite")
        if np.any(self.linear_upper_bounds <= self.linear_lower_bounds):
            raise ValueError("Every linear parameter requires a nonzero-width bound")
        if np.any(self.linear_lower_bounds > 0) or np.any(self.linear_upper_bounds < 0):
            raise ValueError(
                "The separable solver requires zero to lie inside every linear bound"
            )

    def _build_weighted_problem(self) -> WeightedFittingProblem:
        """Reproduce the wrapper's legacy or normalized PK2 objective weights."""
        experimental_pk2 = np.asarray(self.wrapper.expstressPK2, dtype=float)
        if experimental_pk2.ndim != 2 or experimental_pk2.shape[1] != 6:
            raise ValueError("Experimental PK2 stresses must have shape (rows, 6)")

        target_parts: list[FloatArray] = []
        scale_parts: list[FloatArray] = []
        cost_type = str(self.wrapper.cost_type).lower()
        for axis in self.fit_axes:
            # One component contains values such as all experimental xx stresses.
            component = np.asarray(experimental_pk2[:, axis], dtype=float)
            target_parts.append(component)
            if cost_type == "legacy":
                scale_parts.append(np.ones_like(component))
            elif cost_type == "normalized":
                reference_maxima = getattr(
                    self.wrapper,
                    "normalized_stress_reference_maxima",
                    {},
                )
                maximum = float(
                    reference_maxima.get(axis, np.max(np.abs(component)))
                )
                floor = float(self.wrapper.normalized_stress_floor_factor) * maximum
                scale = np.abs(component) + floor
                if np.any(scale == 0):
                    raise ValueError(
                        "Normalized cost has a zero scale; use a positive floor or legacy cost"
                    )
                scale_parts.append(scale)
            else:
                raise ValueError(f"Unsupported cost type for separable solver: {cost_type}")

        target = np.concatenate(target_parts)
        scales = np.concatenate(scale_parts)
        return WeightedFittingProblem(
            axes=self.fit_axes,
            target=target,
            scales=scales,
            weighted_target=target / scales,
        )

    def parameter_bounds(self, name: str) -> tuple[float, float]:
        """Return one parameter's .dat bounds, for example beta_1 -> (0.1, 10)."""
        _, _, lower, upper, _, _ = self.wrapper.getv(name)
        return float(lower), float(upper)

    def integer_values(self, name: str = "zeta") -> tuple[int, ...]:
        """Enumerate an integer bound inclusively, for example 2 through 15."""
        value, dtype, lower, upper, _, _ = self.wrapper.getv(name)
        if dtype is not int:
            raise ValueError(f"Parameter {name} is not declared as an integer: {value}")
        return tuple(range(int(lower), int(upper) + 1))

    def validate_shape_values(
        self,
        shape_values: Mapping[str, float | int],
        *,
        require_all: bool = True,
    ) -> dict[str, float | int]:
        """Validate generic shape values, for example all five SV exponents."""
        supplied = dict(shape_values)
        required = set(self.definition.shape_parameters)
        derived_names = set(self.definition.derived_parameter_names)
        unknown = sorted(set(supplied) - required - derived_names)
        if unknown:
            raise ValueError("Unknown shape parameters: " + ", ".join(unknown))
        missing = sorted(required - set(supplied))
        if require_all and missing:
            raise ValueError("Missing shape parameters: " + ", ".join(missing))

        validated: dict[str, float | int] = {}
        integer_names = set(self.definition.integer_shape_parameters)
        for name, value in supplied.items():
            if name in derived_names:
                continue
            lower, upper = self.parameter_bounds(name)
            tolerance = 16.0 * np.finfo(float).eps * max(
                1.0, abs(lower), abs(upper)
            )
            numeric = float(value)
            if numeric < lower - tolerance or numeric > upper + tolerance:
                raise ValueError(f"{name}={value} is outside [{lower}, {upper}]")
            numeric = float(np.clip(numeric, lower, upper))
            if name in integer_names:
                rounded = int(round(numeric))
                if not np.isclose(numeric, rounded, rtol=0.0, atol=tolerance):
                    raise ValueError(f"{name}={value} is not an integer")
                if rounded not in self.integer_values(name):
                    raise ValueError(f"{name}={value} is outside its integer domain")
                validated[name] = rounded
            else:
                validated[name] = numeric
        for rule in self.definition.derived_parameters:
            sources = {name for name, _ in rule.coefficients}
            if sources <= set(validated):
                derived = rule.evaluate(validated)
                lower, upper = self.parameter_bounds(rule.target)
                tolerance = 16.0 * np.finfo(float).eps * max(
                    1.0, abs(lower), abs(upper), abs(derived)
                )
                if derived < lower - tolerance or derived > upper + tolerance:
                    raise ValueError(
                        f"Derived {rule.target}={derived} is outside [{lower}, {upper}]"
                    )
                if rule.target in supplied and not np.isclose(
                    float(supplied[rule.target]),
                    derived,
                    rtol=0.0,
                    atol=tolerance,
                ):
                    raise ValueError(
                        f"{rule.target} must equal its derived value {derived}"
                    )
            elif rule.target in supplied:
                raise ValueError(
                    f"Cannot supply derived {rule.target} without {sorted(sources)}"
                )
        return validated

    def derived_parameter_values(
        self,
        shape_values: Mapping[str, float | int],
    ) -> dict[str, float]:
        """Return derivable UMAT values, for example ``{"a": 1-b}``."""
        values = dict(shape_values)
        derived: dict[str, float] = {}
        for rule in self.definition.derived_parameters:
            sources = {name for name, _ in rule.coefficients}
            if sources <= set(values):
                derived[rule.target] = rule.evaluate(values)
        return derived

    def expanded_shape_parameter_values(
        self,
        shape_values: Mapping[str, float | int],
    ) -> dict[str, float | int]:
        """Add derived UMAT values to an independent shape assignment."""
        expanded = dict(shape_values)
        expanded.update(self.derived_parameter_values(shape_values))
        return expanded

    def linear_bounds_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> tuple[FloatArray, FloatArray]:
        """Return effective bounds, allowing subclasses to vary them by shape."""
        del shape_values
        return self.linear_lower_bounds.copy(), self.linear_upper_bounds.copy()

    def _snapshot_parameters(self) -> dict[str, tuple[Any, ...]]:
        """Copy parameter records so temporary basis values can be restored."""
        return {name: tuple(record) for name, record in self.wrapper.get_props_dic().items()}

    def _restore_parameters(self, snapshot: dict[str, tuple[Any, ...]]) -> None:
        """Restore values and metadata after a temporary UMAT basis evaluation."""
        for name, record in snapshot.items():
            self.wrapper.setv(name, *record)

    def _set_parameter_value(self, name: str, value: float | int) -> None:
        """Change only one value while retaining its type, bounds, counter, and fix flag."""
        _, dtype, lower, upper, counter, fixed = self.wrapper.getv(name)
        typed_value: float | int = int(round(value)) if dtype is int else float(value)
        self.wrapper.setv(name, typed_value, dtype, lower, upper, counter, fixed)

    def _set_all_linear_to_zero(self) -> None:
        """Set the six stress magnitudes to the exact zero-reference state."""
        for name in self.definition.linear_parameters:
            self._set_parameter_value(name, 0.0)

    def _selected_prediction_pk2(self) -> FloatArray:
        """Run the complete UMAT and stack the selected predicted PK2 components."""
        calculated = np.asarray(self.wrapper.calc_stress_umat_Abaqus(), dtype=float)
        if self.wrapper.umat_out_stress_type != "S":
            calculated = np.asarray(
                self.wrapper.stress_to_PK2(calculated, self.wrapper.umat_out_stress_type),
                dtype=float,
            )
        return _stack_axes(calculated, self.fit_axes)

    def _basis_column(
        self,
        linear_parameter: str,
        shape_values: dict[str, float | int],
    ) -> FloatArray:
        """Return one normalized unit-response column from a full UMAT sweep."""
        snapshot = self._snapshot_parameters()
        try:
            self._set_all_linear_to_zero()
            for name, value in self.expanded_shape_parameter_values(shape_values).items():
                self._set_parameter_value(name, value)

            # Use the farther nonzero bound as a stable finite amplitude; for
            # example, K_coup uses 15 rather than an arbitrarily tiny change.
            _, _, lower, upper, _, _ = self.wrapper.getv(linear_parameter)
            amplitude = float(upper if abs(float(upper)) >= abs(float(lower)) else lower)
            if amplitude == 0:
                raise ValueError(f"Cannot build a zero-amplitude basis for {linear_parameter}")
            self._set_parameter_value(linear_parameter, amplitude)
            prediction = self._selected_prediction_pk2()
            return prediction / amplitude / self.problem.scales
        finally:
            self._restore_parameters(snapshot)

    def basis_column(
        self,
        linear_parameter: str,
        shape_values: Mapping[str, float | int] | None = None,
    ) -> FloatArray:
        """Return a cached weighted basis column for any registered model term."""
        dependencies = self.definition.basis_dependencies()
        if linear_parameter not in dependencies:
            raise ValueError(f"Unknown linear parameter: {linear_parameter}")
        supplied = {} if shape_values is None else dict(shape_values)
        expected = dependencies[linear_parameter]
        if set(supplied) != set(expected):
            raise ValueError(
                f"{linear_parameter} requires shape values {expected}, "
                f"not {tuple(supplied)}"
            )
        validated = self.validate_shape_values(supplied, require_all=False)
        key_values = tuple((name, validated[name]) for name in expected)
        key = (linear_parameter, key_values)
        if key not in self._basis_columns:
            self._basis_columns[key] = self._basis_column(
                linear_parameter,
                dict(key_values),
            )
        return self._basis_columns[key]

    def validate_basis_contributions(
        self,
        shape_values: Mapping[str, float | int],
    ) -> dict[str, float]:
        """Re-run every unit UMAT contribution and compare it with its cached basis."""
        validated = self.validate_shape_values(shape_values)
        dependencies = self.definition.basis_dependencies()
        differences: dict[str, float] = {}
        for linear_name in self.definition.linear_parameters:
            controlling = {
                name: validated[name] for name in dependencies[linear_name]
            }
            cached = self.basis_column(linear_name, controlling)
            direct = self._basis_column(linear_name, controlling)
            differences[linear_name] = float(np.max(np.abs(cached - direct)))
        return differences

    def _constant_column(self, name: str) -> FloatArray:
        """Cache a rate-independent column, for example the mu response."""
        if name not in self._constant_columns:
            self._constant_columns[name] = self.basis_column(name, {})
        return self._constant_columns[name]

    def _family_one_column(self, beta_1: float) -> FloatArray:
        """Cache the first-family column at one rate, for example beta_1=1.0."""
        key = float(beta_1)
        if key not in self._beta_columns:
            self._beta_columns[key] = self.basis_column("K_1ani1", {"beta_1": key})
        return self._beta_columns[key]

    def _family_two_column(self, gamma_1: float) -> FloatArray:
        """Cache the second-family column at one rate, for example gamma_1=0.2."""
        key = float(gamma_1)
        if key not in self._gamma_columns:
            self._gamma_columns[key] = self.basis_column("K_1ani2", {"gamma_1": key})
        return self._gamma_columns[key]

    def _coupling_column(self, zeta: int) -> FloatArray:
        """Cache the coupling column at one integer exponent, for example zeta=5."""
        key = int(zeta)
        if key not in self._zeta_columns:
            self._zeta_columns[key] = self.basis_column("K_coup", {"zeta": key})
        return self._zeta_columns[key]

    def build_weighted_design_from_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> FloatArray:
        """Build the complete weighted design for a generic shape assignment."""
        validated = self.validate_shape_values(shape_values)
        dependencies = self.definition.basis_dependencies()
        columns = []
        for name in self.definition.linear_parameters:
            controlling = {
                shape_name: validated[shape_name]
                for shape_name in dependencies[name]
            }
            columns.append(self.basis_column(name, controlling))
        return np.column_stack(columns)

    def build_weighted_optimization_design_from_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> FloatArray:
        """Build the design used for optimization, excluding fixed-zero families."""
        validated = self.validate_shape_values(shape_values)
        dependencies = self.definition.basis_dependencies()
        columns = []
        for name in self.active_linear_parameters:
            controlling = {
                shape_name: validated[shape_name]
                for shape_name in dependencies[name]
            }
            columns.append(self.basis_column(name, controlling))
        return np.column_stack(columns)

    def build_weighted_design(
        self,
        beta_1: float,
        gamma_1: float,
        zeta: int,
    ) -> FloatArray:
        """Build the weighted observation-by-coefficient matrix at one shape."""
        beta_1, gamma_1, zeta = self._validated_shape_values(
            beta_1, gamma_1, zeta
        )
        columns: dict[str, FloatArray] = {}
        for name in self.definition.linear_parameters:
            if name in {"mu", "mu2", "mu3"}:
                columns[name] = self._constant_column(name)
            elif name == "K_1ani1":
                columns[name] = self._family_one_column(beta_1)
            elif name == "K_1ani2":
                columns[name] = self._family_two_column(gamma_1)
            elif name == "K_coup":
                columns[name] = self._coupling_column(zeta)
            else:
                raise ValueError(f"Unsupported separable linear parameter: {name}")
        return np.column_stack(
            [columns[name] for name in self.definition.linear_parameters]
        )

    def _validated_shape_values(
        self,
        beta_1: float,
        gamma_1: float,
        zeta: int,
    ) -> tuple[float, float, int]:
        """Validate shapes and clip roundoff such as 10.000000000000002 to 10."""
        continuous_values: list[float] = []
        for name, value in (("beta_1", beta_1), ("gamma_1", gamma_1)):
            lower, upper = self.parameter_bounds(name)
            numeric_value = float(value)
            tolerance = 16.0 * np.finfo(float).eps * max(
                1.0, abs(lower), abs(upper)
            )
            if numeric_value < lower - tolerance or numeric_value > upper + tolerance:
                raise ValueError(f"{name}={value} is outside [{lower}, {upper}]")
            continuous_values.append(float(np.clip(numeric_value, lower, upper)))
        integer_options = self.integer_values("zeta")
        if int(zeta) != zeta or int(zeta) not in integer_options:
            raise ValueError(f"zeta={zeta} is not one of {integer_options}")
        return continuous_values[0], continuous_values[1], int(zeta)

    def evaluate(self, beta_1: float, gamma_1: float, zeta: int) -> ProfiledSolution:
        """Solve the six bounded linear coefficients at one nonlinear shape."""
        beta_1, gamma_1, zeta = self._validated_shape_values(
            beta_1, gamma_1, zeta
        )
        weighted_design = self.build_weighted_design(beta_1, gamma_1, zeta)
        shape_values: dict[str, float | int] = {
            "beta_1": beta_1,
            "gamma_1": gamma_1,
        }
        if "zeta" in self.definition.integer_shape_parameters:
            shape_values["zeta"] = zeta
        return self._solve_weighted_design(weighted_design, shape_values)

    def solve_linear_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> ProfiledSolution:
        """Profile all linear coefficients at one generic shape assignment."""
        validated = self.validate_shape_values(shape_values)
        weighted_design = self.build_weighted_design_from_shapes(validated)
        return self._solve_weighted_design(weighted_design, validated)

    def _solve_weighted_design(
        self,
        weighted_design: FloatArray,
        shape_values: Mapping[str, float | int],
    ) -> ProfiledSolution:
        """Solve bounded coefficients for an already assembled weighted design."""
        lower_bounds, upper_bounds = self.linear_bounds_at_shapes(shape_values)
        spans = upper_bounds - lower_bounds
        if np.any(spans < 0):
            raise ValueError("Shape-dependent linear upper bounds cannot be below lower bounds")
        self._profiled_evaluation_count += 1

        # Solve dimensionless variables u in [0, 1], for example u=0.5 means
        # the physical coefficient is halfway between its .dat bounds.
        scaled_design = weighted_design * spans[np.newaxis, :]
        shifted_target = (
            self.problem.weighted_target
            - weighted_design @ lower_bounds
        )
        result = lsq_linear(
            scaled_design,
            shifted_target,
            bounds=(0.0, 1.0),
            method=self.linear_method,
            tol=self.linear_tolerance,
            max_iter=self.linear_max_iterations,
        )
        if not result.success:
            raise RuntimeError(f"Bounded linear solve failed: {result.message}")

        dimensionless = np.asarray(result.x, dtype=float)
        coefficients = lower_bounds + spans * dimensionless
        backend = "bvls"
        iterations = int(result.nit)
        optimality = float(result.optimality)
        active_mask = np.asarray(result.active_mask, dtype=int)

        margins, tolerances = self._constraint_diagnostics(
            coefficients,
            lower_bounds,
            upper_bounds,
        )
        if any(margins[name] < -tolerances[name] for name in margins):
            (
                coefficients,
                dimensionless,
                active_mask,
                optimality,
                iterations,
            ) = self._solve_cone_constrained_design(
                scaled_design,
                shifted_target,
                lower_bounds,
                upper_bounds,
                dimensionless,
            )
            backend = "slsqp_soc"
            margins, tolerances = self._constraint_diagnostics(
                coefficients,
                lower_bounds,
                upper_bounds,
            )
            violations = {
                name: margin
                for name, margin in margins.items()
                if margin < -tolerances[name]
            }
            if violations:
                raise RuntimeError(
                    "Cone-constrained solve returned infeasible coefficients: "
                    + ", ".join(f"{name}={value:g}" for name, value in violations.items())
                )

        residual = weighted_design @ coefficients - self.problem.weighted_target
        cost = float(residual @ residual)
        self._linear_backend_counts[backend] += 1
        for name, margin in margins.items():
            self._minimum_constraint_margins[name] = min(
                margin,
                self._minimum_constraint_margins.get(name, np.inf),
            )
        complete_shapes = dict(shape_values)
        derived_shapes = self.derived_parameter_values(complete_shapes)
        beta_1 = float(complete_shapes.get("beta_1", 0.0))
        gamma_1 = float(complete_shapes.get("gamma_1", 0.0))
        zeta = int(complete_shapes.get("zeta", 0))
        return ProfiledSolution(
            beta_1=beta_1,
            gamma_1=gamma_1,
            zeta=zeta,
            cost=cost,
            mean_cost=cost / float(self.problem.target.size),
            linear_values={
                name: float(value)
                for name, value in zip(self.definition.linear_parameters, coefficients)
            },
            active_bounds={
                name: int(mask)
                for name, mask in zip(self.definition.linear_parameters, active_mask)
            },
            optimality=optimality,
            iterations=iterations,
            shape_values=complete_shapes,
            derived_shape_values=derived_shapes,
            linear_solver_backend=backend,
            constraint_margins=margins,
            constraint_tolerances=tolerances,
            effective_linear_upper_bounds={
                name: float(value)
                for name, value in zip(self.definition.linear_parameters, upper_bounds)
            },
        )

    def _constraint_diagnostics(
        self,
        coefficients: FloatArray,
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return physical cone margins and bound-scaled feasibility tolerances."""
        margins: dict[str, float] = {}
        tolerances: dict[str, float] = {}
        names = self.definition.linear_parameters
        for constraint in self.definition.rotated_cone_constraints:
            first_index = names.index(constraint.first)
            second_index = names.index(constraint.second)
            cross_index = names.index(constraint.cross)
            first = float(coefficients[first_index])
            second = float(coefficients[second_index])
            cross = float(coefficients[cross_index])
            label = (
                f"{constraint.factor:g}*{constraint.first}*{constraint.second}"
                f"-{constraint.cross}^2"
            )
            margin = constraint.factor * first * second - cross * cross
            cross_scale = max(
                abs(float(lower_bounds[cross_index])),
                abs(float(upper_bounds[cross_index])),
            ) ** 2
            product_scale = constraint.factor * max(
                abs(float(lower_bounds[first_index])),
                abs(float(upper_bounds[first_index])),
            ) * max(
                abs(float(lower_bounds[second_index])),
                abs(float(upper_bounds[second_index])),
            )
            scale = max(product_scale, cross_scale, np.finfo(float).tiny)
            margins[label] = float(margin)
            tolerances[label] = float(constraint.relative_tolerance * scale)
        return margins, tolerances

    def _project_cone_coefficients(self, coefficients: FloatArray) -> FloatArray:
        """Clip each cross coefficient to its exact rotated-cone boundary."""
        projected = np.asarray(coefficients, dtype=float).copy()
        names = self.definition.linear_parameters
        for constraint in self.definition.rotated_cone_constraints:
            first_index = names.index(constraint.first)
            second_index = names.index(constraint.second)
            cross_index = names.index(constraint.cross)
            limit = np.sqrt(
                max(
                    0.0,
                    constraint.factor
                    * projected[first_index]
                    * projected[second_index],
                )
            )
            projected[cross_index] = float(
                np.clip(projected[cross_index], -limit, limit)
            )
        return projected

    def _solve_cone_constrained_design(
        self,
        scaled_design: FloatArray,
        shifted_target: FloatArray,
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
        bvls_dimensionless: FloatArray,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.int_], float, int]:
        """Solve convex bounded least squares with rotated-cone restrictions."""
        spans = upper_bounds - lower_bounds
        names = self.definition.linear_parameters

        def physical(values: FloatArray) -> FloatArray:
            return lower_bounds + spans * np.asarray(values, dtype=float)

        def objective(values: FloatArray) -> float:
            residual = scaled_design @ values - shifted_target
            return float(residual @ residual)

        def gradient(values: FloatArray) -> FloatArray:
            residual = scaled_design @ values - shifted_target
            return 2.0 * scaled_design.T @ residual

        scipy_constraints: list[dict[str, Any]] = []
        for constraint in self.definition.rotated_cone_constraints:
            first_index = names.index(constraint.first)
            second_index = names.index(constraint.second)
            cross_index = names.index(constraint.cross)

            def cone_value(
                values: FloatArray,
                first_index: int = first_index,
                second_index: int = second_index,
                cross_index: int = cross_index,
                factor: float = constraint.factor,
            ) -> float:
                coeff = physical(values)
                return float(
                    factor * coeff[first_index] * coeff[second_index]
                    - coeff[cross_index] ** 2
                )

            def cone_jacobian(
                values: FloatArray,
                first_index: int = first_index,
                second_index: int = second_index,
                cross_index: int = cross_index,
                factor: float = constraint.factor,
            ) -> FloatArray:
                coeff = physical(values)
                jacobian = np.zeros_like(values, dtype=float)
                jacobian[first_index] = (
                    factor * coeff[second_index] * spans[first_index]
                )
                jacobian[second_index] = (
                    factor * coeff[first_index] * spans[second_index]
                )
                jacobian[cross_index] = -2.0 * coeff[cross_index] * spans[cross_index]
                return jacobian

            scipy_constraints.append(
                {"type": "ineq", "fun": cone_value, "jac": cone_jacobian}
            )

        projected = self._project_cone_coefficients(physical(bvls_dimensionless))
        projected_start = np.divide(
            projected - lower_bounds,
            spans,
            out=np.zeros_like(spans),
            where=spans > 0,
        )
        strict_coefficients = physical(bvls_dimensionless)
        for constraint in self.definition.rotated_cone_constraints:
            first_index = names.index(constraint.first)
            second_index = names.index(constraint.second)
            cross_index = names.index(constraint.cross)
            strict_coefficients[first_index] = (
                lower_bounds[first_index] + 0.25 * spans[first_index]
            )
            strict_coefficients[second_index] = (
                lower_bounds[second_index] + 0.25 * spans[second_index]
            )
            strict_coefficients[cross_index] = 0.0
        strict_start = np.divide(
            strict_coefficients - lower_bounds,
            spans,
            out=np.zeros_like(spans),
            where=spans > 0,
        )

        feasible_results = []
        for start in (projected_start, strict_start):
            result = minimize(
                objective,
                np.clip(start, 0.0, 1.0),
                method="SLSQP",
                jac=gradient,
                bounds=[(0.0, 1.0)] * len(spans),
                constraints=scipy_constraints,
                options={
                    "ftol": min(self.linear_tolerance, 1.0e-12),
                    "maxiter": max(500, self.linear_max_iterations),
                    "disp": False,
                },
            )
            candidate = self._project_cone_coefficients(physical(result.x))
            margins, tolerances = self._constraint_diagnostics(
                candidate,
                lower_bounds,
                upper_bounds,
            )
            if result.success and all(
                margins[name] >= -tolerances[name] for name in margins
            ):
                candidate_values = np.divide(
                    candidate - lower_bounds,
                    spans,
                    out=np.zeros_like(spans),
                    where=spans > 0,
                )
                feasible_results.append((objective(candidate_values), result, candidate, candidate_values))

        if not feasible_results:
            raise RuntimeError("SLSQP could not find a feasible cone-constrained solution")
        _, best, coefficients, dimensionless = min(feasible_results, key=lambda item: item[0])
        bound_tolerance = 1.0e-8
        active_mask = np.where(
            dimensionless <= bound_tolerance,
            -1,
            np.where(dimensionless >= 1.0 - bound_tolerance, 1, 0),
        ).astype(int)
        optimality = float(np.linalg.norm(np.asarray(best.jac, dtype=float), ord=np.inf))
        return coefficients, dimensionless, active_mask, optimality, int(best.nit)

    def validate_solution(self, solution: ProfiledSolution) -> SolutionValidation:
        """Verify a profiled solution against an ordinary full-parameter UMAT call."""
        complete_shapes = solution.complete_shape_values()
        selected_shapes = {
            name: complete_shapes[name] for name in self.definition.shape_parameters
        }
        weighted_design = self.build_weighted_design_from_shapes(selected_shapes)
        coefficient_vector = np.array(
            [solution.linear_values[name] for name in self.definition.linear_parameters],
            dtype=float,
        )
        profiled_prediction = weighted_design @ coefficient_vector

        snapshot = self._snapshot_parameters()
        try:
            for name, value in self.expanded_shape_parameter_values(selected_shapes).items():
                self._set_parameter_value(name, value)
            for name, value in solution.linear_values.items():
                self._set_parameter_value(name, value)
            direct_prediction = self._selected_prediction_pk2() / self.problem.scales
            # This second full sweep exercises the established SSMCCF cost path;
            # for example, it should agree with a profiled cost of 3.20.
            existing_wrapper_cost = float(
                self.wrapper.cost_calculations(
                    get_plots=False,
                    axis=list(self.fit_axes),
                )
            )
        finally:
            self._restore_parameters(snapshot)

        direct_residual = direct_prediction - self.problem.weighted_target
        direct_cost = float(direct_residual @ direct_residual)
        singular_values = np.linalg.svd(weighted_design, compute_uv=False)
        return SolutionValidation(
            direct_cost=direct_cost,
            existing_wrapper_cost=existing_wrapper_cost,
            profiled_cost=solution.cost,
            absolute_cost_difference=abs(direct_cost - solution.cost),
            existing_wrapper_cost_difference=abs(
                existing_wrapper_cost - solution.cost
            ),
            maximum_weighted_prediction_difference=float(
                np.max(np.abs(direct_prediction - profiled_prediction))
            ),
            design_matrix_rank=int(np.linalg.matrix_rank(weighted_design)),
            design_matrix_singular_values=tuple(float(value) for value in singular_values),
        )

    def cache_sizes(self) -> dict[str, int]:
        """Return basis-cache counts, for example 21 beta and 21 gamma columns."""
        return {
            "constant": len(self._constant_columns),
            "beta_1": len(self._beta_columns),
            "gamma_1": len(self._gamma_columns),
            "zeta": len(self._zeta_columns),
            "generic": len(self._basis_columns),
        }

    def profiled_evaluation_count(self) -> int:
        """Return bounded-linear solve calls, for example 3969 after a full map."""
        return self._profiled_evaluation_count

    def linear_backend_counts(self) -> dict[str, int]:
        """Return accepted inner solves by backend, for example 179 BVLS and 1 SLSQP."""
        return dict(self._linear_backend_counts)

    def minimum_constraint_margins(self) -> dict[str, float]:
        """Return the smallest accepted physical margin across all profiles."""
        return dict(self._minimum_constraint_margins)


# Preserve the original import used by the completed SV_exp studies.
SVExpSeparableSolver = SeparableMaterialSolver


def logarithmic_grid(bounds: tuple[float, float], points: int) -> FloatArray:
    """Return a positive log-spaced grid, for example 0.1 through 10."""
    lower, upper = (float(value) for value in bounds)
    if lower <= 0 or upper <= lower:
        raise ValueError(f"Logarithmic grid requires 0 < lower < upper, got {bounds}")
    if int(points) < 2:
        raise ValueError("A grid requires at least two points")
    return np.geomspace(lower, upper, int(points), dtype=float)


def linear_grid(bounds: tuple[float, float], points: int) -> FloatArray:
    """Return a uniform grid including both endpoints, for example 0.01 to 0.99."""
    lower, upper = (float(value) for value in bounds)
    if upper <= lower:
        raise ValueError(f"Linear grid requires lower < upper, got {bounds}")
    if int(points) < 2:
        raise ValueError("A grid requires at least two points")
    return np.linspace(lower, upper, int(points), dtype=float)


def deterministic_shape_grid_search(
    solver: SeparableMaterialSolver,
    continuous_grids: Mapping[str, Sequence[float]],
    integer_shape_values: Mapping[str, Sequence[int]],
    *,
    progress_callback: ProgressCallback | None = None,
    reject_infeasible_cone_points: bool = False,
) -> ShapeGridSearchResult:
    """Map a shape grid, optionally rejecting failed constrained inner solves."""
    continuous_names = solver.definition.continuous_shape_parameters
    integer_names = solver.definition.integer_shape_parameters
    if set(continuous_grids) != set(continuous_names):
        raise ValueError(
            f"Continuous grids must match {continuous_names}, not {tuple(continuous_grids)}"
        )
    if set(integer_shape_values) != set(integer_names):
        raise ValueError(
            f"Integer domains must match {integer_names}, not {tuple(integer_shape_values)}"
        )
    grids = {
        name: np.asarray(continuous_grids[name], dtype=float)
        for name in continuous_names
    }
    if any(values.ndim != 1 or values.size == 0 for values in grids.values()):
        raise ValueError("Every continuous grid must be a nonempty one-dimensional sequence")
    integer_domains = {
        name: tuple(int(value) for value in integer_shape_values[name])
        for name in integer_names
    }
    if any(not values for values in integer_domains.values()):
        raise ValueError("Every integer shape domain must be nonempty")
    integer_combinations = tuple(
        itertools.product(*(integer_domains[name] for name in integer_names))
    )
    continuous_shape = tuple(grids[name].size for name in continuous_names)
    costs = np.full((len(integer_combinations), *continuous_shape), np.inf, dtype=float)
    best_by_integer: list[ProfiledSolution] = []
    overall_best: ProfiledSolution | None = None
    rejected_evaluations: list[dict[str, float | int | str]] = []

    for integer_index, integer_values in enumerate(integer_combinations):
        fixed_shapes = dict(zip(integer_names, integer_values))
        integer_best: ProfiledSolution | None = None
        for continuous_index in np.ndindex(continuous_shape):
            shapes = {
                name: float(grids[name][index])
                for name, index in zip(continuous_names, continuous_index)
            }
            shapes.update(fixed_shapes)
            try:
                solution = solver.solve_linear_at_shapes(shapes)
            except RuntimeError as error:
                message = str(error)
                is_cone_rejection = (
                    message == "SLSQP could not find a feasible cone-constrained solution"
                    or message.startswith(
                        "Cone-constrained solve returned infeasible coefficients:"
                    )
                )
                if not reject_infeasible_cone_points or not is_cone_rejection:
                    raise
                rejected_evaluations.append({**shapes, "reason": message})
                continue
            costs[(integer_index, *continuous_index)] = solution.cost
            if integer_best is None or solution.cost < integer_best.cost:
                integer_best = solution
            if overall_best is None or solution.cost < overall_best.cost:
                overall_best = solution
        if integer_best is None:
            raise RuntimeError(f"No solution was evaluated for {fixed_shapes}")
        best_by_integer.append(integer_best)
        if progress_callback is not None and "zeta" in fixed_shapes:
            progress_callback(int(fixed_shapes["zeta"]), integer_best)

    if overall_best is None:
        raise RuntimeError("The deterministic shape-grid search evaluated no solutions")
    return ShapeGridSearchResult(
        continuous_names=continuous_names,
        continuous_grids=grids,
        integer_names=integer_names,
        integer_combinations=integer_combinations,
        costs=costs,
        best_by_integer_combination=tuple(best_by_integer),
        best_solution=overall_best,
        rejected_evaluations=tuple(rejected_evaluations),
    )


def refine_profiled_shapes(
    solver: SeparableMaterialSolver,
    initial: ProfiledSolution,
    *,
    maximum_iterations: int = 100,
    parameter_tolerance: float = 1.0e-6,
    cost_tolerance: float = 1.0e-10,
) -> ProfiledSolution:
    """Polish all continuous shapes while holding integer shapes fixed."""
    continuous_names = solver.definition.continuous_shape_parameters
    if not continuous_names:
        return initial
    complete = initial.complete_shape_values()
    transformed_initial: list[float] = []
    transformed_bounds: list[tuple[float, float]] = []
    for name in continuous_names:
        lower, upper = solver.parameter_bounds(name)
        value = float(complete[name])
        if solver.definition.shape_scale(name) == "log":
            if lower <= 0:
                raise ValueError(f"Log-refined shape {name} requires a positive lower bound")
            transformed_initial.append(float(np.log(value)))
            transformed_bounds.append((float(np.log(lower)), float(np.log(upper))))
        else:
            transformed_initial.append(value)
            transformed_bounds.append((lower, upper))
    fixed_integer = {
        name: int(complete[name]) for name in solver.definition.integer_shape_parameters
    }

    def decode(values: FloatArray) -> dict[str, float | int]:
        shapes: dict[str, float | int] = {}
        for name, transformed in zip(continuous_names, values):
            shapes[name] = (
                float(np.exp(transformed))
                if solver.definition.shape_scale(name) == "log"
                else float(transformed)
            )
        shapes.update(fixed_integer)
        return shapes

    def objective(values: FloatArray) -> float:
        return solver.solve_linear_at_shapes(decode(values)).cost

    result = minimize(
        objective,
        np.asarray(transformed_initial, dtype=float),
        method="Powell",
        bounds=transformed_bounds,
        options={
            "maxiter": int(maximum_iterations),
            "xtol": float(parameter_tolerance),
            "ftol": float(cost_tolerance),
        },
    )
    refined = solver.solve_linear_at_shapes(decode(np.asarray(result.x, dtype=float)))
    return refined if refined.cost < initial.cost else initial


def deterministic_grid_search(
    solver: SeparableMaterialSolver,
    beta_values: Sequence[float],
    gamma_values: Sequence[float],
    zeta_values: Sequence[int],
    *,
    progress_callback: ProgressCallback | None = None,
) -> GridSearchResult:
    """Map the complete supplied beta/gamma grid for every supplied zeta."""
    beta_array = np.asarray(beta_values, dtype=float)
    gamma_array = np.asarray(gamma_values, dtype=float)
    zeta_tuple = tuple(int(value) for value in zeta_values)
    if beta_array.ndim != 1 or beta_array.size == 0:
        raise ValueError("beta_values must be a nonempty one-dimensional sequence")
    if gamma_array.ndim != 1 or gamma_array.size == 0:
        raise ValueError("gamma_values must be a nonempty one-dimensional sequence")
    if not zeta_tuple:
        raise ValueError("zeta_values must not be empty")

    # The cost cube uses [zeta, beta, gamma], for example shape (14, 21, 21).
    costs = np.empty((len(zeta_tuple), beta_array.size, gamma_array.size), dtype=float)
    best_by_zeta: list[ProfiledSolution] = []
    overall_best: ProfiledSolution | None = None

    for zeta_index, zeta in enumerate(zeta_tuple):
        zeta_best: ProfiledSolution | None = None
        for beta_index, beta_1 in enumerate(beta_array):
            for gamma_index, gamma_1 in enumerate(gamma_array):
                solution = solver.evaluate(float(beta_1), float(gamma_1), int(zeta))
                costs[zeta_index, beta_index, gamma_index] = solution.cost
                if zeta_best is None or solution.cost < zeta_best.cost:
                    zeta_best = solution
                if overall_best is None or solution.cost < overall_best.cost:
                    overall_best = solution
        if zeta_best is None:
            raise RuntimeError(f"No solution was evaluated for zeta={zeta}")
        best_by_zeta.append(zeta_best)
        if progress_callback is not None:
            progress_callback(zeta, zeta_best)

    if overall_best is None:
        raise RuntimeError("The deterministic grid search evaluated no solutions")
    return GridSearchResult(
        beta_values=beta_array,
        gamma_values=gamma_array,
        zeta_values=zeta_tuple,
        costs=costs,
        best_by_zeta=tuple(best_by_zeta),
        best_solution=overall_best,
    )


def refine_profiled_solution(
    solver: SeparableMaterialSolver,
    initial: ProfiledSolution,
    *,
    maximum_iterations: int = 100,
    parameter_tolerance: float = 1.0e-6,
    cost_tolerance: float = 1.0e-10,
) -> ProfiledSolution:
    """Compatibility wrapper for generic continuous-shape Powell polishing."""
    return refine_profiled_shapes(
        solver,
        initial,
        maximum_iterations=maximum_iterations,
        parameter_tolerance=parameter_tolerance,
        cost_tolerance=cost_tolerance,
    )
