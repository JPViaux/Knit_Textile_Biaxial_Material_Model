"""Certified SOS1 solver for discrete separable material-model shapes.

PySCIPOpt is imported only when ``CertifiedDiscreteSeparableSolver.solve`` is
called. Existing grid/Powell SV_exp and SV_tanh workflows therefore do not
depend on SCIP.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import itertools
from release_paths import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Mapping

# Avoid loading a second Intel OpenMP runtime when the Windows SCIP wheel is
# imported later. This selects MKL's compatible GNU threading layer up front.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from Python_Wrapper.separable_solver import (
    ProfiledSolution,
    SeparableMaterialSolver,
    SolutionValidation,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BasisMetric:
    """Describe one exponent-specific weighted unit-response basis."""

    linear_parameter: str
    shape_parameter: str
    exponent: int
    weighted_rms: float
    excited: bool


@dataclass(frozen=True)
class CertifiedDiscreteSolution:
    """Store one SCIP solution and its independently recomputed diagnostics."""

    amplitude_ceiling: float
    shape_values: dict[str, int]
    linear_values: dict[str, float]
    amplitudes: dict[str, float]
    inactive_shapes: tuple[str, ...]
    active_bounds: dict[str, int]
    cost: float
    mean_cost: float
    status: str
    certificate_satisfied: bool
    relative_gap: float
    absolute_gap: float
    primal_bound: float
    dual_bound: float
    solve_seconds: float
    design_matrix_rank: int
    design_matrix_singular_values: tuple[float, ...]
    formulation_prediction_difference: float
    weighted_prediction: FloatArray
    weighted_residual: FloatArray
    validation: SolutionValidation
    scip_reported_relative_gap: float = 0.0
    scip_reported_primal_bound: float = 0.0
    certificate_backend: str = "scip_sos1"
    enumerated_shape_combinations: int = 0

    def as_profiled_solution(self) -> ProfiledSolution:
        """Return the shared solution form used by direct-UMAT validation."""
        return ProfiledSolution(
            beta_1=float(self.shape_values.get("beta_1", 0)),
            gamma_1=float(self.shape_values.get("gamma_1", 0)),
            zeta=int(self.shape_values.get("zeta", 0)),
            cost=self.cost,
            mean_cost=self.mean_cost,
            linear_values=dict(self.linear_values),
            active_bounds=dict(self.active_bounds),
            optimality=self.relative_gap,
            iterations=0,
            shape_values=dict(self.shape_values),
        )


class CertifiedDiscreteSeparableSolver:
    """Select integer bases globally while optimizing their linear amplitudes."""

    def __init__(
        self,
        solver: SeparableMaterialSolver,
        *,
        inactive_linear_parameters: tuple[str, ...] | None = None,
    ) -> None:
        """Prepare basis families, for example five Stapleton exponent groups."""
        if importlib.util.find_spec("pyscipopt") is None:
            raise RuntimeError(
                "The certified Stapleton backend requires PySCIPOpt. "
                "Install PySCIPOpt 6.2.0 in the selected Python environment."
            )
        self.solver = solver
        self.definition = solver.definition
        dependencies = self.definition.basis_dependencies()
        integer_shapes = set(self.definition.integer_shape_parameters)
        self.constant_parameters = tuple(
            name for name in self.definition.linear_parameters if not dependencies[name]
        )
        all_family_parameters = tuple(
            name for name in self.definition.linear_parameters if dependencies[name]
        )
        self.all_family_parameters = all_family_parameters
        core_inactive = tuple(solver.inactive_linear_parameters)
        requested_inactive = (
            core_inactive
            if inactive_linear_parameters is None
            else tuple(str(name) for name in inactive_linear_parameters)
        )
        if inactive_linear_parameters is not None and requested_inactive != core_inactive:
            raise ValueError(
                "Certified and core inactive linear parameters must agree exactly"
            )
        if len(set(requested_inactive)) != len(requested_inactive):
            raise ValueError("Inactive nonlinear family names cannot repeat")
        unknown_inactive = sorted(set(requested_inactive) - set(all_family_parameters))
        if unknown_inactive:
            raise ValueError(
                "Inactive parameters are not nonlinear basis families: "
                + ", ".join(unknown_inactive)
            )
        inactive_set = set(requested_inactive)
        self.inactive_family_parameters = tuple(
            name for name in all_family_parameters if name in inactive_set
        )
        self.family_parameters = tuple(
            name for name in all_family_parameters if name not in inactive_set
        )
        if not self.family_parameters:
            raise ValueError("Certified discrete solving requires nonlinear basis families")

        self.family_shapes: dict[str, str] = {}
        for name in all_family_parameters:
            controlling = dependencies[name]
            if len(controlling) != 1 or controlling[0] not in integer_shapes:
                raise ValueError(
                    f"{name} must depend on exactly one integer shape, not {controlling}"
                )
            self.family_shapes[name] = controlling[0]
            lower, _ = solver.parameter_bounds(name)
            if not np.isclose(lower, 0.0):
                raise ValueError(
                    f"Amplitude-equalized {name} requires a zero lower bound"
                )

        # Observable-amplitude normalization can safely scale any finite,
        # nonzero basis; for example, exponent 100 may have a unit RMS near
        # 1e-30 but a well-resolved normalized shape. Direct UMAT validation
        # remains the final guard against unsafe coefficient rescaling.
        self.excitation_tolerance = np.finfo(float).tiny
        self._basis_metrics: list[BasisMetric] = []
        self._normalized_columns: dict[tuple[str, int], FloatArray] = {}
        self._raw_columns: dict[tuple[str, int], FloatArray] = {}
        self.auto_inactive_family_parameters: tuple[str, ...] = ()
        self._prepare_bases()

    def _prepare_bases(self) -> None:
        """Compute and normalize every allowed exponent-specific UMAT basis."""
        unexcited_families: list[str] = []
        for linear_name in self.family_parameters:
            shape_name = self.family_shapes[linear_name]
            excited_count = 0
            for exponent in self.solver.integer_values(shape_name):
                raw = np.asarray(
                    self.solver.basis_column(
                        linear_name,
                        {shape_name: exponent},
                    ),
                    dtype=float,
                )
                rms = float(np.sqrt(np.mean(raw**2)))
                excited = bool(math.isfinite(rms) and rms > self.excitation_tolerance)
                self._basis_metrics.append(
                    BasisMetric(
                        linear_parameter=linear_name,
                        shape_parameter=shape_name,
                        exponent=int(exponent),
                        weighted_rms=rms,
                        excited=excited,
                    )
                )
                self._raw_columns[(linear_name, int(exponent))] = raw
                if excited:
                    self._normalized_columns[(linear_name, int(exponent))] = raw / rms
                    excited_count += 1
            if excited_count == 0:
                unexcited_families.append(linear_name)

        # A loading subset can make an otherwise valid family identically zero
        # (for example, one yarn family in a pure single-axis test). Such a
        # coefficient cannot be estimated. Canonically remove it from the
        # optimization exactly like a manifest-declared inactive family.
        if unexcited_families:
            auto_set = set(unexcited_families)
            inactive_set = set(self.inactive_family_parameters) | auto_set
            self.auto_inactive_family_parameters = tuple(
                name for name in self.all_family_parameters if name in auto_set
            )
            self.inactive_family_parameters = tuple(
                name for name in self.all_family_parameters if name in inactive_set
            )
            self.family_parameters = tuple(
                name for name in self.all_family_parameters if name not in inactive_set
            )
        if not self.family_parameters:
            raise ValueError(
                "Certified discrete solving requires at least one excited nonlinear basis family"
            )

    @property
    def basis_metrics(self) -> tuple[BasisMetric, ...]:
        """Return immutable basis diagnostics for CSV/JSON output."""
        return tuple(self._basis_metrics)

    def effective_raw_upper_bounds(
        self,
        amplitude_ceiling: float,
    ) -> dict[str, dict[int, float]]:
        """Convert one observable ceiling to exponent-specific raw K bounds."""
        ceiling = float(amplitude_ceiling)
        if not math.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("Amplitude ceiling must be finite and positive")
        converted: dict[str, dict[int, float]] = {}
        for metric in self._basis_metrics:
            if metric.excited:
                converted.setdefault(metric.linear_parameter, {})[metric.exponent] = (
                    ceiling / metric.weighted_rms
                )
        return converted

    def _shape_guess(self, name: str) -> int:
        """Return the canonical integer guess, for example beta_1=5."""
        value, dtype, _, _, _, _ = self.solver.wrapper.getv(name)
        if dtype is not int:
            raise ValueError(f"Certified shape {name} must be declared as int")
        return int(value)

    def canonicalize_inactive_shape(
        self,
        shape_name: str,
        selected_exponent: int,
        amplitude: float,
        amplitude_ceiling: float,
    ) -> tuple[int, bool]:
        """Replace a numerically inactive exponent with its `.dat` guess."""
        tolerance = max(1.0e-10, float(amplitude_ceiling) * 1.0e-8)
        if float(amplitude) <= tolerance:
            return self._shape_guess(shape_name), True
        return int(selected_exponent), False

    @staticmethod
    def _active_bound(value: float, lower: float, upper: float) -> int:
        """Return -1/0/1 for lower/interior/upper bound activity."""
        tolerance = 1.0e-8 * max(1.0, abs(lower), abs(upper))
        if abs(value - lower) <= tolerance:
            return -1
        if abs(value - upper) <= tolerance:
            return 1
        return 0

    @staticmethod
    def _clip_to_bounds(value: float, lower: float, upper: float) -> float:
        """Remove sub-tolerance SCIP bound leakage, for example -1e-10 to 0."""
        return float(min(upper, max(lower, value)))

    def solve_exhaustive(
        self,
        amplitude_ceiling: float,
        *,
        time_limit_seconds: float = 14_400.0,
    ) -> CertifiedDiscreteSolution:
        """Certify a reduced problem by enumerating every active exponent tuple."""
        ceiling = float(amplitude_ceiling)
        time_limit = float(time_limit_seconds)
        if not math.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("Amplitude ceiling must be finite and positive")
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("Time limit must be finite and positive")
        target = self.solver.problem.weighted_target
        constant_columns = {
            name: self.solver.basis_column(name, {})
            for name in self.constant_parameters
        }
        # Match the SCIP candidate policy if a basis truly underflows to zero.
        # Such a candidate has no observable direction and is equivalent to an
        # inactive family, which is represented explicitly by zero amplitude.
        domains = [
            tuple(
                exponent
                for exponent in self.solver.integer_values(self.family_shapes[name])
                if (name, int(exponent)) in self._normalized_columns
            )
            for name in self.family_parameters
        ]
        if any(not domain for domain in domains):
            raise ValueError("At least one active family has no excited exponent basis")
        variable_domain_count = sum(len(domain) > 1 for domain in domains)
        if variable_domain_count > 3:
            raise ValueError(
                "Exhaustive certification supports at most three non-singleton "
                "shape domains"
            )
        total_combinations = math.prod(len(domain) for domain in domains)
        lower_bounds = np.array(
            [self.solver.parameter_bounds(name)[0] for name in self.constant_parameters]
            + [0.0] * len(self.family_parameters),
            dtype=float,
        )
        upper_bounds = np.array(
            [self.solver.parameter_bounds(name)[1] for name in self.constant_parameters]
            + [ceiling] * len(self.family_parameters),
            dtype=float,
        )

        started = time.perf_counter()
        best_cost = math.inf
        best_exponents: tuple[int, ...] | None = None
        best_values: FloatArray | None = None
        best_prediction: FloatArray | None = None
        evaluated = 0
        completed = True
        for exponents in itertools.product(*domains):
            if evaluated and evaluated % 32 == 0:
                if time.perf_counter() - started >= time_limit:
                    completed = False
                    break
            columns = [constant_columns[name] for name in self.constant_parameters]
            columns.extend(
                self._normalized_columns[(name, int(exponent))]
                for name, exponent in zip(self.family_parameters, exponents)
            )
            design = np.column_stack(columns)
            result = lsq_linear(
                design,
                target,
                bounds=(lower_bounds, upper_bounds),
                method="bvls",
                tol=1.0e-12,
                max_iter=100,
            )
            if not result.success:
                raise RuntimeError(
                    "Bounded linear solve failed during exhaustive certification: "
                    + result.message
                )
            prediction = design @ result.x
            residual = prediction - target
            cost = float(residual @ residual)
            evaluated += 1
            # Product iteration is lexicographic, so strict improvement supplies
            # a stable smallest-exponent tie break.
            if best_exponents is None or cost < best_cost - 1.0e-14 * max(
                1.0, abs(best_cost)
            ):
                best_cost = cost
                best_exponents = tuple(int(value) for value in exponents)
                best_values = np.asarray(result.x, dtype=float)
                best_prediction = prediction

        if best_exponents is None or best_values is None or best_prediction is None:
            raise RuntimeError("Exhaustive certification evaluated no shape combinations")

        shape_values: dict[str, int] = {}
        linear_values = {
            name: self._clip_to_bounds(
                float(best_values[index]), *self.solver.parameter_bounds(name)
            )
            for index, name in enumerate(self.constant_parameters)
        }
        amplitudes: dict[str, float] = {}
        inactive_shapes: list[str] = []
        active_bounds: dict[str, int] = {}
        for name, value in linear_values.items():
            active_bounds[name] = self._active_bound(
                value, *self.solver.parameter_bounds(name)
            )

        for linear_name in self.inactive_family_parameters:
            shape_name = self.family_shapes[linear_name]
            shape_values[shape_name] = self._shape_guess(shape_name)
            linear_values[linear_name] = 0.0
            amplitudes[linear_name] = 0.0
            inactive_shapes.append(shape_name)
            active_bounds[linear_name] = -1
            active_bounds[shape_name] = 0

        offset = len(self.constant_parameters)
        for family_index, (linear_name, exponent) in enumerate(
            zip(self.family_parameters, best_exponents)
        ):
            shape_name = self.family_shapes[linear_name]
            amplitude = self._clip_to_bounds(
                float(best_values[offset + family_index]), 0.0, ceiling
            )
            selected_exponent, inactive = self.canonicalize_inactive_shape(
                shape_name, exponent, amplitude, ceiling
            )
            if inactive:
                amplitude = 0.0
                coefficient = 0.0
                inactive_shapes.append(shape_name)
                active_bounds[linear_name] = -1
                active_bounds[shape_name] = 0
            else:
                rms = next(
                    metric.weighted_rms
                    for metric in self._basis_metrics
                    if metric.linear_parameter == linear_name
                    and metric.exponent == selected_exponent
                )
                coefficient = amplitude / rms
                active_bounds[linear_name] = self._active_bound(
                    amplitude, 0.0, ceiling
                )
                active_bounds[shape_name] = self._active_bound(
                    float(selected_exponent),
                    *self.solver.parameter_bounds(shape_name),
                )
            shape_values[shape_name] = int(selected_exponent)
            linear_values[linear_name] = float(coefficient)
            amplitudes[linear_name] = float(amplitude)

        weighted_design = self.solver.build_weighted_design_from_shapes(shape_values)
        coefficient_vector = np.array(
            [linear_values[name] for name in self.definition.linear_parameters],
            dtype=float,
        )
        weighted_prediction = weighted_design @ coefficient_vector
        weighted_residual = weighted_prediction - target
        cost = float(weighted_residual @ weighted_residual)
        optimization_design = self.solver.build_weighted_optimization_design_from_shapes(
            shape_values
        )
        singular_values = np.linalg.svd(optimization_design, compute_uv=False)
        status = "enumeration_complete" if completed else "timelimit"
        certificate_satisfied = completed and evaluated == total_combinations
        profiled = ProfiledSolution(
            beta_1=float(shape_values.get("beta_1", 0)),
            gamma_1=float(shape_values.get("gamma_1", 0)),
            zeta=int(shape_values.get("zeta", 0)),
            cost=cost,
            mean_cost=cost / float(target.size),
            linear_values=linear_values,
            active_bounds=active_bounds,
            optimality=0.0 if certificate_satisfied else 1.0,
            iterations=evaluated,
            shape_values=shape_values,
        )
        validation = self.solver.validate_solution(profiled)
        return CertifiedDiscreteSolution(
            amplitude_ceiling=ceiling,
            shape_values=shape_values,
            linear_values=linear_values,
            amplitudes=amplitudes,
            inactive_shapes=tuple(inactive_shapes),
            active_bounds=active_bounds,
            cost=cost,
            mean_cost=cost / float(target.size),
            status=status,
            certificate_satisfied=certificate_satisfied,
            relative_gap=0.0 if certificate_satisfied else 1.0,
            absolute_gap=0.0 if certificate_satisfied else cost,
            primal_bound=cost,
            dual_bound=cost if certificate_satisfied else 0.0,
            solve_seconds=float(time.perf_counter() - started),
            design_matrix_rank=int(np.linalg.matrix_rank(optimization_design)),
            design_matrix_singular_values=tuple(float(value) for value in singular_values),
            formulation_prediction_difference=float(
                np.max(np.abs(best_prediction - weighted_prediction))
            ),
            weighted_prediction=weighted_prediction,
            weighted_residual=weighted_residual,
            validation=validation,
            scip_reported_relative_gap=0.0,
            scip_reported_primal_bound=cost,
            certificate_backend="exhaustive_enumeration",
            enumerated_shape_combinations=evaluated,
        )

    def solve(
        self,
        amplitude_ceiling: float,
        *,
        relative_gap: float = 1.0e-6,
        time_limit_seconds: float = 14_400.0,
        problem_path: pathlib.Path | None = None,
        solution_path: pathlib.Path | None = None,
        display: bool = False,
    ) -> CertifiedDiscreteSolution:
        """Solve one amplitude ceiling and return a certified material solution."""
        ceiling = float(amplitude_ceiling)
        gap_limit = float(relative_gap)
        time_limit = float(time_limit_seconds)
        if not math.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("Amplitude ceiling must be finite and positive")
        if not math.isfinite(gap_limit) or gap_limit <= 0:
            raise ValueError("Relative gap must be finite and positive")
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("Time limit must be finite and positive")

        target = self.solver.problem.weighted_target
        constant_columns = {
            name: self.solver.basis_column(name, {})
            for name in self.constant_parameters
        }
        payload = {
            "target": target.tolist(),
            "amplitude_ceiling": ceiling,
            "relative_gap": gap_limit,
            "time_limit_seconds": time_limit,
            "display": bool(display),
            "problem_path": None if problem_path is None else str(pathlib.Path(problem_path)),
            "solution_path": None if solution_path is None else str(pathlib.Path(solution_path)),
            "constants": {
                name: {
                    "lower": self.solver.parameter_bounds(name)[0],
                    "upper": self.solver.parameter_bounds(name)[1],
                    "column": constant_columns[name].tolist(),
                }
                for name in self.constant_parameters
            },
            "families": {
                linear_name: {
                    "shape_parameter": self.family_shapes[linear_name],
                    "candidates": [
                        {
                            "exponent": exponent,
                            "column": self._normalized_columns[(linear_name, exponent)].tolist(),
                        }
                        for exponent in self.solver.integer_values(
                            self.family_shapes[linear_name]
                        )
                        if (linear_name, exponent) in self._normalized_columns
                    ],
                }
                for linear_name in self.family_parameters
            },
        }
        worker = pathlib.Path(__file__).with_name("separable_solver_scip_worker.py")
        with tempfile.TemporaryDirectory(prefix="ssmccf_scip_") as directory:
            temporary = pathlib.Path(directory)
            input_path = temporary / "problem.json"
            output_path = temporary / "solution.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            environment = os.environ.copy()
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[variable] = "1"
            completed = subprocess.run(
                [sys.executable, str(worker), str(input_path), str(output_path)],
                text=True,
                capture_output=not display,
                env=environment,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"SCIP worker failed with exit code {completed.returncode}: {detail}"
                )
            worker_result = json.loads(output_path.read_text(encoding="utf-8"))

        if not worker_result.get("has_solution"):
            raise RuntimeError(
                "SCIP returned no feasible solution; "
                f"status={worker_result.get('status')}"
            )
        constant_values = {}
        for name, value in worker_result["constant_values"].items():
            lower, upper = self.solver.parameter_bounds(name)
            constant_values[name] = self._clip_to_bounds(
                float(value), lower, upper
            )
        amplitude_values = {
            (name, int(exponent)): self._clip_to_bounds(
                float(value), 0.0, ceiling
            )
            for name, exponent, value in worker_result["amplitude_values"]
        }
        shape_values: dict[str, int] = {}
        linear_values = dict(constant_values)
        realized_amplitudes: dict[str, float] = {}
        inactive_shapes: list[str] = []
        active_bounds: dict[str, int] = {}

        for name, value in constant_values.items():
            lower, upper = self.solver.parameter_bounds(name)
            active_bounds[name] = self._active_bound(value, lower, upper)

        # Structurally removed families remain explicit zero coefficients. Their
        # exponents are unidentifiable and therefore use the declared `.dat` guess.
        for linear_name in self.inactive_family_parameters:
            shape_name = self.family_shapes[linear_name]
            shape_values[shape_name] = self._shape_guess(shape_name)
            linear_values[linear_name] = 0.0
            realized_amplitudes[linear_name] = 0.0
            inactive_shapes.append(shape_name)
            active_bounds[linear_name] = -1
            active_bounds[shape_name] = 0

        for linear_name in self.family_parameters:
            shape_name = self.family_shapes[linear_name]
            candidates = {
                exponent: amplitude_values.get((linear_name, exponent), 0.0)
                for exponent in self.solver.integer_values(shape_name)
            }
            selected_exponent, amplitude = max(
                candidates.items(), key=lambda item: item[1]
            )
            selected_exponent, inactive = self.canonicalize_inactive_shape(
                shape_name,
                selected_exponent,
                amplitude,
                ceiling,
            )
            if inactive:
                amplitude = 0.0
                coefficient = 0.0
                inactive_shapes.append(shape_name)
                active_bounds[linear_name] = -1
            else:
                rms = next(
                    metric.weighted_rms
                    for metric in self._basis_metrics
                    if metric.linear_parameter == linear_name
                    and metric.exponent == selected_exponent
                )
                coefficient = amplitude / rms
                active_bounds[linear_name] = self._active_bound(
                    amplitude, 0.0, ceiling
                )
            shape_values[shape_name] = int(selected_exponent)
            shape_lower, shape_upper = self.solver.parameter_bounds(shape_name)
            active_bounds[shape_name] = (
                0
                if inactive
                else self._active_bound(
                    float(selected_exponent), shape_lower, shape_upper
                )
            )
            linear_values[linear_name] = float(coefficient)
            realized_amplitudes[linear_name] = float(amplitude)

        weighted_design = self.solver.build_weighted_design_from_shapes(shape_values)
        coefficient_vector = np.array(
            [linear_values[name] for name in self.definition.linear_parameters],
            dtype=float,
        )
        weighted_prediction = weighted_design @ coefficient_vector
        weighted_residual = weighted_prediction - target
        cost = float(weighted_residual @ weighted_residual)

        scip_prediction = np.zeros_like(target)
        for name, value in constant_values.items():
            scip_prediction += constant_columns[name] * value
        for key, amplitude in amplitude_values.items():
            scip_prediction += self._normalized_columns[key] * amplitude

        optimization_design = self.solver.build_weighted_optimization_design_from_shapes(
            shape_values
        )
        singular_values = np.linalg.svd(optimization_design, compute_uv=False)
        scip_reported_gap = float(worker_result["relative_gap"])
        scip_reported_primal = float(worker_result["primal_bound"])
        dual_bound = float(worker_result["dual_bound"])
        relative_gap_value = scip_reported_gap
        status = str(worker_result["status"])
        certificate_satisfied = bool(
            math.isfinite(relative_gap_value)
            and relative_gap_value <= gap_limit * (1.0 + 1.0e-9)
            and status in {"optimal", "gaplimit"}
        )
        profiled = ProfiledSolution(
            beta_1=float(shape_values.get("beta_1", 0)),
            gamma_1=float(shape_values.get("gamma_1", 0)),
            zeta=int(shape_values.get("zeta", 0)),
            cost=cost,
            mean_cost=cost / float(target.size),
            linear_values=linear_values,
            active_bounds=active_bounds,
            optimality=relative_gap_value,
            iterations=0,
            shape_values=shape_values,
        )
        validation = self.solver.validate_solution(profiled)

        return CertifiedDiscreteSolution(
            amplitude_ceiling=ceiling,
            shape_values=shape_values,
            linear_values=linear_values,
            amplitudes=realized_amplitudes,
            inactive_shapes=tuple(inactive_shapes),
            active_bounds=active_bounds,
            cost=cost,
            mean_cost=cost / float(target.size),
            status=status,
            certificate_satisfied=certificate_satisfied,
            relative_gap=relative_gap_value,
            absolute_gap=max(0.0, scip_reported_primal - dual_bound),
            primal_bound=scip_reported_primal,
            dual_bound=dual_bound,
            solve_seconds=float(worker_result["solve_seconds"]),
            design_matrix_rank=int(np.linalg.matrix_rank(optimization_design)),
            design_matrix_singular_values=tuple(float(value) for value in singular_values),
            formulation_prediction_difference=float(
                np.max(np.abs(scip_prediction - weighted_prediction))
            ),
            weighted_prediction=weighted_prediction,
            weighted_residual=weighted_residual,
            validation=validation,
            scip_reported_relative_gap=scip_reported_gap,
            scip_reported_primal_bound=scip_reported_primal,
        )
