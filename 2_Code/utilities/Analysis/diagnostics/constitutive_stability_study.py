"""Test polyconvexity and Legendre--Hadamard stability of the raw B09 fits.

Edit the USER CONFIGURATION block and press Run in VSCode. The study reads the
eight immutable raw prescribed-B09 fit artifacts, proves a global rank-one
convexity counterexample, and screens measured and interpolated biaxial states.
It never fits parameters, changes a UMAT, or allocates a fit code.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import argparse
import csv
import ctypes
from dataclasses import dataclass
import hashlib
from release_paths import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
import numpy as np
from scipy.optimize import minimize, minimize_scalar


SUPPORT_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis").parent
CODE_DIR = SUPPORT_DIR.parent
for directory in (SUPPORT_DIR, CODE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import uml_plot_style as uml_style  # noqa: E402

uml_style.apply()

from Analysis.diagnostics import model_component_contribution_study as components  # noqa: E402
from Python_Wrapper import mech, umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# Raw prescribed-B09 fit codes; example: SV_exp Rib is fit 722.
FIT_CODES: dict[str, dict[str, int]] = {
    "SV_exp": {
        "Interlock": 726,
        "Rib": 722,
        "Purl": 723,
        "SingleJersey": 725,
    },
    "StapletonViaux": {
        "Interlock": 724,
        "Rib": 729,
        "Purl": 727,
        "SingleJersey": 728,
    },
}
# Output folder; example: Outputs/constitutive_stability_study.
OUTPUT_DIRECTORY = RELEASE_ARCHIVE_ROOT / "constitutive_stability_study"
# Stretch samples on each envelope axis; example: 121 gives 14,641 states.
STRETCH_GRID_POINTS = 121
# Propagation directions on [0, pi); example: 721 is about 0.25 degrees apart.
ANGLE_GRID_POINTS = 721
# Low grid candidates refined per scope; example: 32 catches distinct valleys.
LOCAL_REFINEMENT_STARTS = 32
# Absolute sign tolerance in model stress units; example: 1e-10.
SIGN_ABSOLUTE_TOLERANCE = 1.0e-10
# Relative sign tolerance times the case stiffness scale; example: 1e-8.
SIGN_RELATIVE_TOLERANCE = 1.0e-8
# Activation tolerance; example: |I4-C1|<=1e-9 is treated as nonsmooth.
ACTIVATION_TOLERANCE = 1.0e-9
# Offset for explicit one-sided activation checks; example: 1e-7 stretch.
ACTIVATION_OFFSET = 1.0e-7
# UMAT difference steps; example: 3e-8 checks a state 1e-7 from activation.
UMAT_STEP_SIZES = (
    1.0e-4,
    3.0e-5,
    1.0e-5,
    3.0e-6,
    1.0e-6,
    3.0e-7,
    1.0e-7,
    3.0e-8,
)
# Relative UMAT-curvature tolerance; example: 5e-5.
UMAT_RELATIVE_TOLERANCE = 5.0e-5
# Absolute UMAT-curvature tolerance; example: 1e-8 stress units.
UMAT_ABSOLUTE_TOLERANCE = 1.0e-8
# True compiles both frozen Fortran UMATs; use False only for debugging.
VALIDATE_WITH_COMPILED_UMAT = True
# True replaces this study's files; False protects an existing completed study.
OVERWRITE_EXISTING = False


# ---------------------------------------------------------------------------
# FIXED STUDY DEFINITIONS - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

MODEL_ORDER = ("SV_exp", "StapletonViaux")
KNIT_ORDER = ("Interlock", "Rib", "Purl", "SingleJersey")
SCOPE_ORDER = ("full_3d", "membrane", "incompressible_membrane")
DOMAIN_ORDER = ("measured_paths", "incompressible_envelope")
POLYCONVEXITY_CSV = "polyconvexity_summary.csv"
LH_SUMMARY_CSV = "legendre_hadamard_summary.csv"
MEASURED_CSV = "measured_state_results.csv"
ENVELOPE_CSV = "envelope_grid_results.csv"
UMAT_CSV = "umat_validation.csv"
SUMMARY_JSON = "study_summary.json"
README_FILE = "README.md"
FIGURE_DIRECTORY = "heatmaps"


@dataclass(frozen=True)
class InvariantDerivatives:
    """Energy derivatives with respect to I4 and I6."""

    value: Any
    first_4: Any
    first_6: Any
    second_44: Any
    second_66: Any
    second_46: Any


@dataclass(frozen=True)
class FitCase:
    """One validated model-knit fit and its nine authoritative raw curves."""

    model_id: str
    knit: str
    fit_code: int
    parameters: dict[str, float]
    payload: dict[str, Any]
    datasets: dict[str, components.RatioDataset]


@dataclass(frozen=True)
class ModeResult:
    """Minimum rank-one curvature and its propagation/polarization vectors."""

    value: float
    theta: float | None
    propagation: np.ndarray
    polarization: np.ndarray
    mode: str


def _parameter(parameters: Mapping[str, float], name: str) -> float:
    """Return one finite model parameter as a float."""
    value = float(parameters[name])
    if not math.isfinite(value):
        raise ValueError(f"Parameter {name} is not finite: {value}")
    return value


def fiber_directions(parameters: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the two reference fiber vectors from theta1/theta2 in degrees."""
    theta_1 = math.radians(_parameter(parameters, "theta1"))
    theta_2 = math.radians(_parameter(parameters, "theta2"))
    return (
        np.asarray([math.cos(theta_1), math.sin(theta_1), 0.0]),
        np.asarray([math.cos(theta_2), math.sin(theta_2), 0.0]),
    )


def invariant_values(
    deformation_gradient: np.ndarray,
    parameters: Mapping[str, float],
) -> tuple[float, float]:
    """Return I4=|F N1|^2 and I6=|F N2|^2."""
    F = np.asarray(deformation_gradient, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        raise ValueError("The deformation gradient must be a finite 3x3 matrix")
    n_1, n_2 = fiber_directions(parameters)
    g_1 = F @ n_1
    g_2 = F @ n_2
    return float(g_1 @ g_1), float(g_2 @ g_2)


def invariant_derivatives(
    model_id: str,
    parameters: Mapping[str, float],
    invariant_4: Any,
    invariant_6: Any,
) -> InvariantDerivatives:
    """Evaluate the exact reduced-model energy and I4/I6 derivatives."""
    if model_id not in {*MODEL_ORDER, "SV_exp_polycon_mo"}:
        raise ValueError(f"Unsupported model: {model_id}")
    i4, i6 = np.broadcast_arrays(
        np.asarray(invariant_4, dtype=float),
        np.asarray(invariant_6, dtype=float),
    )
    if not np.all(np.isfinite(i4)) or not np.all(np.isfinite(i6)):
        raise ValueError("Invariants must be finite")

    x = i4 - 1.0
    y = i6 - 1.0
    mu_1 = _parameter(parameters, "mu")
    mu_2 = _parameter(parameters, "mu2")
    mu_3 = _parameter(parameters, "mu3")
    value = mu_1 * x * x + mu_2 * y * y + mu_3 * x * y
    first_4 = 2.0 * mu_1 * x + mu_3 * y
    first_6 = 2.0 * mu_2 * y + mu_3 * x
    second_44 = np.full_like(x, 2.0 * mu_1)
    second_66 = np.full_like(y, 2.0 * mu_2)
    second_46 = np.full_like(x, mu_3)

    active_4 = i4 > _parameter(parameters, "C1")
    active_6 = i6 > _parameter(parameters, "C2")
    k_1 = _parameter(parameters, "K_1ani1")
    k_2 = _parameter(parameters, "K_1ani2")
    shape_1 = _parameter(parameters, "beta_1")
    shape_2 = _parameter(parameters, "gamma_1")

    if model_id in {"SV_exp", "SV_exp_polycon_mo"}:
        # Keep inactive compression branches out of the exponential entirely
        # (for example, x=-0.2 and shape_1=1000 would otherwise overflow).
        exp_4 = np.exp(np.where(active_4, -shape_1 * x, 0.0))
        exp_6 = np.exp(np.where(active_6, -shape_2 * y, 0.0))
        value = value + np.where(
            active_4,
            k_1 * (x + (exp_4 - 1.0) / shape_1),
            0.0,
        )
        value = value + np.where(
            active_6,
            k_2 * (y + (exp_6 - 1.0) / shape_2),
            0.0,
        )
        first_4 = first_4 + np.where(active_4, k_1 * (1.0 - exp_4), 0.0)
        first_6 = first_6 + np.where(active_6, k_2 * (1.0 - exp_6), 0.0)
        second_44 = second_44 + np.where(active_4, k_1 * shape_1 * exp_4, 0.0)
        second_66 = second_66 + np.where(active_6, k_2 * shape_2 * exp_6, 0.0)
    else:
        x_positive = np.maximum(x, 0.0)
        y_positive = np.maximum(y, 0.0)
        value = value + np.where(active_4, k_1 * x_positive**shape_1, 0.0)
        value = value + np.where(active_6, k_2 * y_positive**shape_2, 0.0)
        first_4 = first_4 + np.where(
            active_4,
            k_1 * shape_1 * x_positive ** (shape_1 - 1.0),
            0.0,
        )
        first_6 = first_6 + np.where(
            active_6,
            k_2 * shape_2 * y_positive ** (shape_2 - 1.0),
            0.0,
        )
        if shape_1 != 1.0:
            second_44 = second_44 + np.where(
                active_4,
                k_1 * shape_1 * (shape_1 - 1.0) * x_positive ** (shape_1 - 2.0),
                0.0,
            )
        if shape_2 != 1.0:
            second_66 = second_66 + np.where(
                active_6,
                k_2 * shape_2 * (shape_2 - 1.0) * y_positive ** (shape_2 - 2.0),
                0.0,
            )

    coupling = _parameter(parameters, "K_coup")
    zeta = _parameter(parameters, "zeta")
    if model_id == "SV_exp_polycon_mo":
        # Combined-max coupling: q=a(I4-C1)+b(I6-C2), Wc=K<q>_+^zeta.
        # The strict q>0 branch matches the Fortran activation policy. At q=0,
        # zeta=2 has a nonzero positive-side Hessian and zero inactive-side
        # Hessian, so the classical tangent is one-sided/undefined there.
        a_weight = _parameter(parameters, "a")
        b_weight = _parameter(parameters, "b")
        q = (
            a_weight * (i4 - _parameter(parameters, "C1"))
            + b_weight * (i6 - _parameter(parameters, "C2"))
        )
        active_q = q > 0.0
        q_positive = np.maximum(q, 0.0)
        value = value + np.where(active_q, coupling * q_positive**zeta, 0.0)
        common_first = np.where(
            active_q,
            coupling * zeta * q_positive ** (zeta - 1.0),
            0.0,
        )
        common_second = np.where(
            active_q,
            coupling * zeta * (zeta - 1.0) * q_positive ** (zeta - 2.0),
            0.0,
        )
        first_4 = first_4 + a_weight * common_first
        first_6 = first_6 + b_weight * common_first
        second_44 = second_44 + a_weight * a_weight * common_second
        second_66 = second_66 + b_weight * b_weight * common_second
        second_46 = second_46 + a_weight * b_weight * common_second
    else:
        both_active = active_4 & active_6
        x_positive = np.maximum(x, 0.0)
        y_positive = np.maximum(y, 0.0)
        value = value + np.where(both_active, coupling * x_positive**zeta * y_positive**zeta, 0.0)
        first_4 = first_4 + np.where(both_active, coupling * zeta * x_positive ** (zeta - 1.0) * y_positive**zeta, 0.0)
        first_6 = first_6 + np.where(both_active, coupling * zeta * x_positive**zeta * y_positive ** (zeta - 1.0), 0.0)
        second_44 = second_44 + np.where(both_active, coupling * zeta * (zeta - 1.0) * x_positive ** (zeta - 2.0) * y_positive**zeta, 0.0)
        second_66 = second_66 + np.where(both_active, coupling * zeta * (zeta - 1.0) * x_positive**zeta * y_positive ** (zeta - 2.0), 0.0)
        second_46 = second_46 + np.where(both_active, coupling * zeta * zeta * x_positive ** (zeta - 1.0) * y_positive ** (zeta - 1.0), 0.0)
    return InvariantDerivatives(
        value=value,
        first_4=first_4,
        first_6=first_6,
        second_44=second_44,
        second_66=second_66,
        second_46=second_46,
    )


def energy_density(
    deformation_gradient: np.ndarray,
    parameters: Mapping[str, float],
    model_id: str,
) -> float:
    """Return W(F) for the implemented reduced Stapleton/SV_exp energy."""
    i4, i6 = invariant_values(deformation_gradient, parameters)
    return float(invariant_derivatives(model_id, parameters, i4, i6).value)


def first_piola_stress(
    deformation_gradient: np.ndarray,
    parameters: Mapping[str, float],
    model_id: str,
) -> np.ndarray:
    """Return P=dW/dF from the invariant energy."""
    F = np.asarray(deformation_gradient, dtype=float)
    n_1, n_2 = fiber_directions(parameters)
    g_1 = F @ n_1
    g_2 = F @ n_2
    derivatives = invariant_derivatives(
        model_id,
        parameters,
        float(g_1 @ g_1),
        float(g_2 @ g_2),
    )
    return (
        2.0 * float(derivatives.first_4) * np.outer(g_1, n_1)
        + 2.0 * float(derivatives.first_6) * np.outer(g_2, n_2)
    )


def acoustic_tensor(
    deformation_gradient: np.ndarray,
    propagation: np.ndarray,
    parameters: Mapping[str, float],
    model_id: str,
) -> np.ndarray:
    """Return Q(n)=A_iAjB n_A n_B for A=d2W/dF2."""
    F = np.asarray(deformation_gradient, dtype=float)
    direction = np.asarray(propagation, dtype=float)
    if F.shape != (3, 3) or direction.shape != (3,):
        raise ValueError("Expected F with shape (3,3) and n with shape (3,)")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Propagation direction must have positive finite norm")
    direction = direction / norm
    n_1, n_2 = fiber_directions(parameters)
    g_1 = F @ n_1
    g_2 = F @ n_2
    q_1 = float(direction @ n_1)
    q_2 = float(direction @ n_2)
    derivatives = invariant_derivatives(
        model_id,
        parameters,
        float(g_1 @ g_1),
        float(g_2 @ g_2),
    )
    first_4 = float(derivatives.first_4)
    first_6 = float(derivatives.first_6)
    second_44 = float(derivatives.second_44)
    second_66 = float(derivatives.second_66)
    second_46 = float(derivatives.second_46)
    tensor = 2.0 * (first_4 * q_1 * q_1 + first_6 * q_2 * q_2) * np.eye(3)
    tensor += 4.0 * second_44 * q_1 * q_1 * np.outer(g_1, g_1)
    tensor += 4.0 * second_66 * q_2 * q_2 * np.outer(g_2, g_2)
    tensor += 4.0 * second_46 * q_1 * q_2 * (
        np.outer(g_1, g_2) + np.outer(g_2, g_1)
    )
    return 0.5 * (tensor + tensor.T)


def rank_one_curvature(
    deformation_gradient: np.ndarray,
    polarization: np.ndarray,
    propagation: np.ndarray,
    parameters: Mapping[str, float],
    model_id: str,
) -> float:
    """Return (a tensor n):A:(a tensor n) for unit a and n."""
    a = np.asarray(polarization, dtype=float)
    a_norm = float(np.linalg.norm(a))
    if a.shape != (3,) or not np.isfinite(a_norm) or a_norm <= 0.0:
        raise ValueError("Polarization must be a positive finite 3-vector")
    a = a / a_norm
    return float(a @ acoustic_tensor(deformation_gradient, propagation, parameters, model_id) @ a)


def activation_boundary(
    deformation_gradient: np.ndarray,
    parameters: Mapping[str, float],
    tolerance: float = ACTIVATION_TOLERANCE,
) -> tuple[bool, bool]:
    """Identify exact I4=C1 and I6=C2 activation surfaces."""
    i4, i6 = invariant_values(deformation_gradient, parameters)
    return (
        abs(i4 - _parameter(parameters, "C1")) <= tolerance,
        abs(i6 - _parameter(parameters, "C2")) <= tolerance,
    )


def combined_coupling_activation_boundary(
    deformation_gradient: np.ndarray,
    parameters: Mapping[str, float],
    tolerance: float = ACTIVATION_TOLERANCE,
) -> bool:
    """Return whether the SV_exp_polycon_mo combined coupling has q=0."""

    i4, i6 = invariant_values(deformation_gradient, parameters)
    q = (
        _parameter(parameters, "a") * (i4 - _parameter(parameters, "C1"))
        + _parameter(parameters, "b") * (i6 - _parameter(parameters, "C2"))
    )
    return abs(q) <= tolerance


def case_activation_boundaries(
    deformation_gradient: np.ndarray,
    case: FitCase,
) -> tuple[bool, bool, bool]:
    """Return directional and combined-max activation flags for one case."""

    boundary_4, boundary_6 = activation_boundary(deformation_gradient, case.parameters)
    combined = (
        combined_coupling_activation_boundary(deformation_gradient, case.parameters)
        if case.model_id == "SV_exp_polycon_mo"
        else False
    )
    return boundary_4, boundary_6, combined


def polyconvexity_counterexample(case: FitCase, stretch: float = 0.5) -> dict[str, Any]:
    """Return the exact compressive rank-one counterexample W''=4 mu(3t^2-1)."""
    if not 0.0 < stretch < 1.0:
        raise ValueError("Counterexample stretch must lie in (0,1)")
    mu_1 = _parameter(case.parameters, "mu")
    second_variation = 4.0 * mu_1 * (3.0 * stretch * stretch - 1.0)
    return {
        "model_id": case.model_id,
        "knit": case.knit,
        "fit_code": case.fit_code,
        "rank_one_path": "F(t)=I+(t-1)e1_tensor_e1",
        "counterexample_stretch": stretch,
        "determinant_F": stretch,
        "mu": mu_1,
        "second_variation": second_variation,
        "rank_one_convex": second_variation >= 0.0,
        "polyconvex": False if second_variation < 0.0 else "not_disproved",
        "conclusion": (
            "globally_not_polyconvex"
            if second_variation < 0.0
            else "counterexample_not_negative"
        ),
    }


def fit_configurations() -> tuple[components.FitConfiguration, ...]:
    """Return the eight prescribed raw-B09 fits in stable output order."""
    return tuple(
        components.FitConfiguration(
            model_id=model_id,
            knit=knit,
            fit_code=FIT_CODES[model_id][knit],
            sample_folder=components.SAMPLE_FOLDERS[knit],
            fit_series="prescribed_b09",
        )
        for model_id in MODEL_ORDER
        for knit in KNIT_ORDER
    )


def load_fit_cases() -> tuple[FitCase, ...]:
    """Load and validate all fit artifacts, hashes, parameters, and B09 data."""
    registry = components.registry_rows()
    dataset_cache: dict[str, dict[str, components.RatioDataset]] = {}
    cases: list[FitCase] = []
    for config in fit_configurations():
        fit, payload = components.load_and_validate_fit(config, registry)
        if config.knit not in dataset_cache:
            dataset_cache[config.knit] = components.load_ratio_datasets(config)
        datasets = dataset_cache[config.knit]
        components.verify_b09_training_data(config, payload, datasets)
        parameters = {name: float(value) for name, value in fit.parameters.items()}
        if config.model_id == "StapletonViaux":
            for name in ("K_2ani1", "K_2ani2"):
                if not math.isclose(parameters[name], 0.0, rel_tol=0.0, abs_tol=1.0e-14):
                    raise ValueError(f"Fit {config.fit_code} does not satisfy {name}=0")
        if not math.isclose(parameters["theta1"], 0.0, abs_tol=1.0e-12):
            raise ValueError(f"Fit {config.fit_code} does not use theta1=0")
        if not math.isclose(parameters["theta2"], 90.0, abs_tol=1.0e-12):
            raise ValueError(f"Fit {config.fit_code} does not use theta2=90")
        cases.append(
            FitCase(
                model_id=config.model_id,
                knit=config.knit,
                fit_code=config.fit_code,
                parameters=parameters,
                payload=payload,
                datasets=datasets,
            )
        )
    if len(cases) != 8 or len({case.fit_code for case in cases}) != 8:
        raise ValueError("The stability study requires eight unique fits")
    return tuple(cases)


def measured_states(case: FitCase) -> list[dict[str, Any]]:
    """Return every raw curve deformation with stable ratio and row identity."""
    states: list[dict[str, Any]] = []
    for ratio in components.RATIO_ORDER:
        dataset = case.datasets[ratio]
        for row_index, f_voigt in enumerate(dataset.dfgrd_voigt):
            F = mech.voigt_to_matrix(f_voigt)
            off_diagonal = F - np.diag(np.diag(F))
            if float(np.max(np.abs(off_diagonal))) > 1.0e-12:
                raise ValueError(
                    f"{case.knit} {ratio} row {row_index} is not diagonal"
                )
            if np.linalg.det(F) <= 0.0:
                raise ValueError(f"{case.knit} {ratio} row {row_index} has det(F)<=0")
            states.append(
                {
                    "ratio": ratio,
                    "ratio_row_index": row_index,
                    "F": F,
                    "lambda_1": float(F[0, 0]),
                    "lambda_2": float(F[1, 1]),
                    "lambda_3": float(F[2, 2]),
                    "determinant_F": float(np.linalg.det(F)),
                }
            )
    return states


def stretch_grid(lower: float, upper: float, points: int) -> np.ndarray:
    """Return a fixed-size grid that contains 1.0 exactly when it is in range."""
    if points < 3 or not 0.0 < lower <= upper:
        raise ValueError("Stretch grid requires positive ordered bounds and >=3 points")
    values = np.linspace(lower, upper, points)
    if lower <= 1.0 <= upper:
        nearest = int(np.argmin(np.abs(values - 1.0)))
        values[nearest] = 1.0
        values.sort()
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Stretch grid must remain strictly increasing")
    return values


def propagation_angles(points: int = ANGLE_GRID_POINTS) -> np.ndarray:
    """Return deterministic in-plane propagation angles on [0, pi)."""
    if points < 8:
        raise ValueError("At least eight propagation angles are required")
    return np.linspace(0.0, math.pi, points, endpoint=False)


def _vectorized_angle_scan(
    lambda_1: np.ndarray,
    lambda_2: np.ndarray,
    parameters: Mapping[str, float],
    model_id: str,
    angles: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Minimize all three LH scopes over a fixed angle grid for diagonal F."""
    l1 = np.asarray(lambda_1, dtype=float).reshape(-1, 1)
    l2 = np.asarray(lambda_2, dtype=float).reshape(-1, 1)
    theta = np.asarray(angles, dtype=float).reshape(1, -1)
    derivatives = invariant_derivatives(model_id, parameters, l1 * l1, l2 * l2)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    c2 = cosine * cosine
    s2 = sine * sine
    base = 2.0 * (
        np.asarray(derivatives.first_4) * c2
        + np.asarray(derivatives.first_6) * s2
    )
    q11 = base + 4.0 * np.asarray(derivatives.second_44) * l1 * l1 * c2
    q22 = base + 4.0 * np.asarray(derivatives.second_66) * l2 * l2 * s2
    q12 = (
        4.0
        * np.asarray(derivatives.second_46)
        * l1
        * l2
        * cosine
        * sine
    )
    discriminant = np.sqrt(np.maximum((q11 - q22) ** 2 + 4.0 * q12 * q12, 0.0))
    eigen_min = 0.5 * (q11 + q22 - discriminant)
    eigen_max = 0.5 * (q11 + q22 + discriminant)

    inverse_normal_1 = cosine / l1
    inverse_normal_2 = sine / l2
    inverse_norm = np.sqrt(inverse_normal_1**2 + inverse_normal_2**2)
    polarization_1 = -inverse_normal_2 / inverse_norm
    polarization_2 = inverse_normal_1 / inverse_norm
    incompressible = (
        polarization_1**2 * q11
        + 2.0 * polarization_1 * polarization_2 * q12
        + polarization_2**2 * q22
    )
    full_3d = np.minimum(np.minimum(eigen_min, base), 0.0)
    all_values = {
        "full_3d": full_3d,
        "membrane": eigen_min,
        "incompressible_membrane": incompressible,
    }
    minima: dict[str, np.ndarray] = {}
    minimum_angles: dict[str, np.ndarray] = {}
    for scope, values in all_values.items():
        indices = np.argmin(values, axis=1)
        minima[scope] = values[np.arange(values.shape[0]), indices]
        minimum_angles[scope] = angles[indices]
    maximum_scale = float(
        np.max(
            np.abs(
                np.concatenate(
                    (eigen_min.reshape(-1), eigen_max.reshape(-1), base.reshape(-1))
                )
            )
        )
    )
    return minima, minimum_angles, maximum_scale


def mode_at_angle(
    deformation_gradient: np.ndarray,
    theta: float,
    scope: str,
    parameters: Mapping[str, float],
    model_id: str,
) -> ModeResult:
    """Return one scope's minimum polarization for an in-plane direction."""
    if scope not in SCOPE_ORDER:
        raise ValueError(f"Unknown LH scope: {scope}")
    canonical_theta = float(theta % math.pi)
    propagation = np.asarray(
        [math.cos(canonical_theta), math.sin(canonical_theta), 0.0]
    )
    Q = acoustic_tensor(deformation_gradient, propagation, parameters, model_id)
    if scope == "incompressible_membrane":
        inverse_normal = np.linalg.inv(deformation_gradient).T @ propagation
        in_plane = inverse_normal[:2]
        norm = float(np.linalg.norm(in_plane))
        if norm <= 0.0:
            raise ValueError("Incompressibility constraint has zero in-plane normal")
        polarization = np.asarray([-in_plane[1] / norm, in_plane[0] / norm, 0.0])
        value = float(polarization @ Q @ polarization)
        return ModeResult(
            value=value,
            theta=canonical_theta,
            propagation=propagation,
            polarization=polarization,
            mode="incompressibility_compatible_in_plane",
        )

    if scope == "membrane":
        eigenvalues, eigenvectors = np.linalg.eigh(Q[:2, :2])
        polarization = np.asarray([eigenvectors[0, 0], eigenvectors[1, 0], 0.0])
        return ModeResult(
            value=float(eigenvalues[0]),
            theta=canonical_theta,
            propagation=propagation,
            polarization=polarization,
            mode="in_plane_eigenmode",
        )

    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    minimum_index = int(np.argmin(eigenvalues))
    minimum = float(eigenvalues[minimum_index])
    if minimum >= 0.0:
        return ModeResult(
            value=0.0,
            theta=None,
            propagation=np.asarray([0.0, 0.0, 1.0]),
            polarization=np.asarray([1.0, 0.0, 0.0]),
            mode="structural_n_equals_e3_zero",
        )
    return ModeResult(
        value=minimum,
        theta=canonical_theta,
        propagation=propagation,
        polarization=eigenvectors[:, minimum_index],
        mode=(
            "out_of_plane_polarization"
            if abs(float(eigenvectors[2, minimum_index])) > 1.0 - 1.0e-8
            else "three_dimensional_eigenmode"
        ),
    )


def minimize_mode_over_angles(
    deformation_gradient: np.ndarray,
    scope: str,
    parameters: Mapping[str, float],
    model_id: str,
    angles: np.ndarray | None = None,
) -> ModeResult:
    """Scan and locally refine one state's minimum propagation direction."""
    angle_values = propagation_angles() if angles is None else np.asarray(angles, dtype=float)
    F = np.asarray(deformation_gradient, dtype=float)
    l1 = float(F[0, 0])
    l2 = float(F[1, 1])
    minima, minimum_angles, _ = _vectorized_angle_scan(
        np.asarray([l1]),
        np.asarray([l2]),
        parameters,
        model_id,
        angle_values,
    )
    coarse_value = float(minima[scope][0])
    coarse_theta = float(minimum_angles[scope][0])
    if scope == "full_3d" and coarse_value >= 0.0:
        return mode_at_angle(F, coarse_theta, scope, parameters, model_id)

    spacing = math.pi / len(angle_values)

    def objective(raw_theta: float) -> float:
        return mode_at_angle(F, raw_theta, scope, parameters, model_id).value

    refined = minimize_scalar(
        objective,
        bounds=(coarse_theta - spacing, coarse_theta + spacing),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 100},
    )
    if refined.success and math.isfinite(float(refined.fun)):
        candidate = mode_at_angle(F, float(refined.x), scope, parameters, model_id)
        if candidate.value <= coarse_value + 1.0e-12:
            return candidate
    return mode_at_angle(F, coarse_theta, scope, parameters, model_id)


def _classification(value: float, tolerance: float, *, activation: bool) -> str:
    """Classify a raw LH minimum without overstating finite numerical coverage."""
    if activation:
        return "classically undefined at activation boundary"
    if value < -tolerance:
        return "violated"
    if abs(value) <= tolerance:
        return "nonnegative on tested set"
    return "strictly positive on tested set"


def analyze_measured_domain(
    case: FitCase,
    angles: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], float]:
    """Analyze all measured B09 states and return detailed and critical rows."""
    rows: list[dict[str, Any]] = []
    critical: dict[str, dict[str, Any]] = {}
    stiffness_scale = 0.0
    for state in measured_states(case):
        F = np.asarray(state["F"], dtype=float)
        _, _, state_scale = _vectorized_angle_scan(
            np.asarray([state["lambda_1"]]),
            np.asarray([state["lambda_2"]]),
            case.parameters,
            case.model_id,
            angles,
        )
        stiffness_scale = max(stiffness_scale, state_scale)
        boundary_4, boundary_6, boundary_combined = case_activation_boundaries(F, case)
        for scope in SCOPE_ORDER:
            mode = minimize_mode_over_angles(
                F,
                scope,
                case.parameters,
                case.model_id,
                angles,
            )
            row = {
                "model_id": case.model_id,
                "knit": case.knit,
                "fit_code": case.fit_code,
                "domain": "measured_paths",
                "scope": scope,
                "ratio": state["ratio"],
                "ratio_row_index": state["ratio_row_index"],
                "lambda_1": state["lambda_1"],
                "lambda_2": state["lambda_2"],
                "lambda_3": state["lambda_3"],
                "determinant_F": state["determinant_F"],
                "minimum_curvature": mode.value,
                "theta_radians": "" if mode.theta is None else mode.theta,
                "theta_degrees": "" if mode.theta is None else math.degrees(mode.theta),
                "propagation_x": float(mode.propagation[0]),
                "propagation_y": float(mode.propagation[1]),
                "propagation_z": float(mode.propagation[2]),
                "polarization_x": float(mode.polarization[0]),
                "polarization_y": float(mode.polarization[1]),
                "polarization_z": float(mode.polarization[2]),
                "mode": mode.mode,
                "on_I4_activation": boundary_4,
                "on_I6_activation": boundary_6,
                "on_combined_coupling_activation": boundary_combined,
            }
            rows.append(row)
            is_boundary = boundary_4 or boundary_6 or boundary_combined
            previous = critical.get(scope)
            previous_boundary = (
                True
                if previous is None
                else bool(
                    previous["on_I4_activation"]
                    or previous["on_I6_activation"]
                    or previous.get("on_combined_coupling_activation", False)
                )
            )
            if (
                previous is None
                or (not is_boundary and previous_boundary)
                or (
                    is_boundary == previous_boundary
                    and mode.value < float(previous["minimum_curvature"])
                )
            ):
                critical[scope] = dict(row)
    return rows, critical, stiffness_scale


def envelope_axes(
    case: FitCase,
    points: int = STRETCH_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the knit-specific lambda1/lambda2 rectangle from measured states."""
    states = measured_states(case)
    lambda_1 = np.asarray([state["lambda_1"] for state in states])
    lambda_2 = np.asarray([state["lambda_2"] for state in states])
    return (
        stretch_grid(float(np.min(lambda_1)), float(np.max(lambda_1)), points),
        stretch_grid(float(np.min(lambda_2)), float(np.max(lambda_2)), points),
    )


def _envelope_objective(
    variables: np.ndarray,
    case: FitCase,
    scope: str,
) -> float:
    """Return one continuous envelope objective for bounded local refinement."""
    lambda_1, lambda_2, theta = (float(value) for value in variables)
    F = np.diag([lambda_1, lambda_2, 1.0 / (lambda_1 * lambda_2)])
    return mode_at_angle(F, theta, scope, case.parameters, case.model_id).value


def _candidate_indices(
    values: np.ndarray,
    lambda_1_axis: np.ndarray,
    lambda_2_axis: np.ndarray,
    limit: int,
) -> list[tuple[int, int]]:
    """Return low values plus corners, edges, and activation-line candidates."""
    flattened = np.argsort(values, axis=None, kind="stable")
    candidates: list[tuple[int, int]] = []
    for raw_index in flattened[: max(limit * 4, limit)]:
        index = tuple(int(item) for item in np.unravel_index(raw_index, values.shape))
        if index not in candidates:
            candidates.append(index)
        if len(candidates) >= limit:
            break
    boundary_groups = [
        [(0, j) for j in range(values.shape[1])],
        [(values.shape[0] - 1, j) for j in range(values.shape[1])],
        [(i, 0) for i in range(values.shape[0])],
        [(i, values.shape[1] - 1) for i in range(values.shape[0])],
    ]
    if lambda_1_axis[0] <= 1.0 <= lambda_1_axis[-1]:
        index_1 = int(np.argmin(np.abs(lambda_1_axis - 1.0)))
        boundary_groups.append([(index_1, j) for j in range(values.shape[1])])
    if lambda_2_axis[0] <= 1.0 <= lambda_2_axis[-1]:
        index_2 = int(np.argmin(np.abs(lambda_2_axis - 1.0)))
        boundary_groups.append([(i, index_2) for i in range(values.shape[0])])
    for group in boundary_groups:
        best = min(group, key=lambda index: float(values[index]))
        if best not in candidates:
            candidates.append(best)
    for corner in (
        (0, 0),
        (0, values.shape[1] - 1),
        (values.shape[0] - 1, 0),
        (values.shape[0] - 1, values.shape[1] - 1),
    ):
        if corner not in candidates:
            candidates.append(corner)
    return candidates


def analyze_envelope_domain(
    case: FitCase,
    angles: np.ndarray,
    *,
    points: int = STRETCH_GRID_POINTS,
    refinement_starts: int = LOCAL_REFINEMENT_STARTS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], float, dict[str, Any]]:
    """Screen and refine the case-specific incompressible biaxial envelope."""
    lambda_1_axis, lambda_2_axis = envelope_axes(case, points)
    mesh_1, mesh_2 = np.meshgrid(lambda_1_axis, lambda_2_axis, indexing="ij")
    flattened_1 = mesh_1.reshape(-1)
    flattened_2 = mesh_2.reshape(-1)
    minimum_arrays = {
        scope: np.empty(flattened_1.size, dtype=float) for scope in SCOPE_ORDER
    }
    angle_arrays = {
        scope: np.empty(flattened_1.size, dtype=float) for scope in SCOPE_ORDER
    }
    stiffness_scale = 0.0
    chunk_size = 512
    for start in range(0, flattened_1.size, chunk_size):
        stop = min(start + chunk_size, flattened_1.size)
        minima, minimum_angles, chunk_scale = _vectorized_angle_scan(
            flattened_1[start:stop],
            flattened_2[start:stop],
            case.parameters,
            case.model_id,
            angles,
        )
        stiffness_scale = max(stiffness_scale, chunk_scale)
        for scope in SCOPE_ORDER:
            minimum_arrays[scope][start:stop] = minima[scope]
            angle_arrays[scope][start:stop] = minimum_angles[scope]

    reshaped_values = {
        scope: values.reshape(mesh_1.shape) for scope, values in minimum_arrays.items()
    }
    reshaped_angles = {
        scope: values.reshape(mesh_1.shape) for scope, values in angle_arrays.items()
    }
    rows: list[dict[str, Any]] = []
    for i, lambda_1 in enumerate(lambda_1_axis):
        for j, lambda_2 in enumerate(lambda_2_axis):
            F = np.diag([lambda_1, lambda_2, 1.0 / (lambda_1 * lambda_2)])
            boundary_4, boundary_6, boundary_combined = case_activation_boundaries(F, case)
            rows.append(
                {
                    "model_id": case.model_id,
                    "knit": case.knit,
                    "fit_code": case.fit_code,
                    "lambda_1": float(lambda_1),
                    "lambda_2": float(lambda_2),
                    "lambda_3": float(F[2, 2]),
                    "determinant_F": float(np.linalg.det(F)),
                    "on_I4_activation": boundary_4,
                    "on_I6_activation": boundary_6,
                    "on_combined_coupling_activation": boundary_combined,
                    "full_3d_minimum": float(reshaped_values["full_3d"][i, j]),
                    "full_3d_theta_degrees": math.degrees(
                        float(reshaped_angles["full_3d"][i, j])
                    ),
                    "membrane_minimum": float(reshaped_values["membrane"][i, j]),
                    "membrane_theta_degrees": math.degrees(
                        float(reshaped_angles["membrane"][i, j])
                    ),
                    "incompressible_membrane_minimum": float(
                        reshaped_values["incompressible_membrane"][i, j]
                    ),
                    "incompressible_membrane_theta_degrees": math.degrees(
                        float(reshaped_angles["incompressible_membrane"][i, j])
                    ),
                }
            )

    critical: dict[str, dict[str, Any]] = {}
    bounds = [
        (float(lambda_1_axis[0]), float(lambda_1_axis[-1])),
        (float(lambda_2_axis[0]), float(lambda_2_axis[-1])),
        (0.0, math.pi),
    ]
    for scope in SCOPE_ORDER:
        values = reshaped_values[scope]
        candidates = _candidate_indices(
            values,
            lambda_1_axis,
            lambda_2_axis,
            refinement_starts,
        )
        best_index = min(candidates, key=lambda index: float(values[index]))
        best_variables = np.asarray(
            [
                lambda_1_axis[best_index[0]],
                lambda_2_axis[best_index[1]],
                reshaped_angles[scope][best_index],
            ],
            dtype=float,
        )
        best_value = float(values[best_index])
        for i, j in candidates:
            initial = np.asarray(
                [lambda_1_axis[i], lambda_2_axis[j], reshaped_angles[scope][i, j]],
                dtype=float,
            )
            result = minimize(
                _envelope_objective,
                initial,
                args=(case, scope),
                method="Powell",
                bounds=bounds,
                options={"xtol": 1.0e-10, "ftol": 1.0e-12, "maxiter": 300},
            )
            if result.success and math.isfinite(float(result.fun)) and float(result.fun) < best_value:
                best_value = float(result.fun)
                best_variables = np.asarray(result.x, dtype=float)

        lambda_1, lambda_2, theta = (float(value) for value in best_variables)
        F = np.diag([lambda_1, lambda_2, 1.0 / (lambda_1 * lambda_2)])
        mode = mode_at_angle(F, theta, scope, case.parameters, case.model_id)
        boundary_4, boundary_6, boundary_combined = case_activation_boundaries(F, case)
        critical[scope] = {
            "model_id": case.model_id,
            "knit": case.knit,
            "fit_code": case.fit_code,
            "domain": "incompressible_envelope",
            "scope": scope,
            "ratio": "",
            "ratio_row_index": "",
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_3": float(F[2, 2]),
            "determinant_F": float(np.linalg.det(F)),
            "minimum_curvature": mode.value,
            "theta_radians": "" if mode.theta is None else mode.theta,
            "theta_degrees": "" if mode.theta is None else math.degrees(mode.theta),
            "propagation_x": float(mode.propagation[0]),
            "propagation_y": float(mode.propagation[1]),
            "propagation_z": float(mode.propagation[2]),
            "polarization_x": float(mode.polarization[0]),
            "polarization_y": float(mode.polarization[1]),
            "polarization_z": float(mode.polarization[2]),
            "mode": mode.mode,
            "on_I4_activation": boundary_4,
            "on_I6_activation": boundary_6,
            "on_combined_coupling_activation": boundary_combined,
            "grid_minimum": float(np.min(values)),
            "refinement_improvement": float(np.min(values)) - mode.value,
        }
        if boundary_4 or boundary_6 or boundary_combined:
            candidate_1 = [lambda_1]
            candidate_2 = [lambda_2]
            if boundary_4:
                candidate_1 = [
                    value
                    for value in (lambda_1 - ACTIVATION_OFFSET, lambda_1 + ACTIVATION_OFFSET)
                    if bounds[0][0] <= value <= bounds[0][1]
                ]
            if boundary_6:
                candidate_2 = [
                    value
                    for value in (lambda_2 - ACTIVATION_OFFSET, lambda_2 + ACTIVATION_OFFSET)
                    if bounds[1][0] <= value <= bounds[1][1]
                ]
            if boundary_combined and not boundary_4 and not boundary_6:
                candidate_1 = [
                    value
                    for value in (lambda_1 - ACTIVATION_OFFSET, lambda_1 + ACTIVATION_OFFSET)
                    if bounds[0][0] <= value <= bounds[0][1]
                ]
            smooth_candidates: list[dict[str, Any]] = []
            for shifted_1 in candidate_1:
                for shifted_2 in candidate_2:
                    shifted_F = np.diag(
                        [shifted_1, shifted_2, 1.0 / (shifted_1 * shifted_2)]
                    )
                    shifted_mode = minimize_mode_over_angles(
                        shifted_F,
                        scope,
                        case.parameters,
                        case.model_id,
                        angles,
                    )
                    shifted_boundary_4, shifted_boundary_6, shifted_combined = (
                        case_activation_boundaries(shifted_F, case)
                    )
                    smooth_candidates.append(
                        {
                            **critical[scope],
                            "lambda_1": shifted_1,
                            "lambda_2": shifted_2,
                            "lambda_3": float(shifted_F[2, 2]),
                            "minimum_curvature": shifted_mode.value,
                            "theta_radians": (
                                "" if shifted_mode.theta is None else shifted_mode.theta
                            ),
                            "theta_degrees": (
                                ""
                                if shifted_mode.theta is None
                                else math.degrees(shifted_mode.theta)
                            ),
                            "propagation_x": float(shifted_mode.propagation[0]),
                            "propagation_y": float(shifted_mode.propagation[1]),
                            "propagation_z": float(shifted_mode.propagation[2]),
                            "polarization_x": float(shifted_mode.polarization[0]),
                            "polarization_y": float(shifted_mode.polarization[1]),
                            "polarization_z": float(shifted_mode.polarization[2]),
                            "mode": shifted_mode.mode,
                            "on_I4_activation": shifted_boundary_4,
                            "on_I6_activation": shifted_boundary_6,
                            "on_combined_coupling_activation": shifted_combined,
                            "activation_limit_source": True,
                        }
                    )
            if smooth_candidates:
                critical[scope] = min(
                    smooth_candidates,
                    key=lambda item: float(item["minimum_curvature"]),
                )
    surface_data = {
        "lambda_1": lambda_1_axis,
        "lambda_2": lambda_2_axis,
        "values": reshaped_values,
    }
    return rows, critical, stiffness_scale, surface_data


def _summary_row(
    critical: Mapping[str, Any],
    *,
    tolerance: float,
    stiffness_scale: float,
    activation_boundary_present: bool,
) -> dict[str, Any]:
    """Add weak/strict classifications and numerical scale to one critical row."""
    value = float(critical["minimum_curvature"])
    smooth_classification = _classification(value, tolerance, activation=False)
    if value < -tolerance:
        domain_status = "violated"
    elif activation_boundary_present:
        domain_status = "classically undefined at activation boundary"
    else:
        domain_status = smooth_classification
    if critical["scope"] == "full_3d":
        strict_status = "violated by exact structural n=e3 zero mode"
    elif value > tolerance:
        strict_status = "strictly positive on smooth tested set"
    else:
        strict_status = "not strictly positive on smooth tested set"
    return {
        **critical,
        "stiffness_scale": stiffness_scale,
        "sign_tolerance": tolerance,
        "weak_lh_smooth_status": smooth_classification,
        "strict_lh_smooth_status": strict_status,
        "activation_boundary_present": activation_boundary_present,
        "domain_classical_status": domain_status,
        "coverage_statement": (
            "validated negative counterexample"
            if value < -tolerance
            else "no violation found on sampled and refined smooth states"
        ),
    }


def analyze_case(
    case: FitCase,
    angles: np.ndarray,
    *,
    points: int = STRETCH_GRID_POINTS,
    refinement_starts: int = LOCAL_REFINEMENT_STARTS,
) -> dict[str, Any]:
    """Analyze one model-knit fit across measured and envelope domains."""
    measured_rows, measured_critical, measured_scale = analyze_measured_domain(
        case, angles
    )
    envelope_rows, envelope_critical, envelope_scale, surface_data = (
        analyze_envelope_domain(
            case,
            angles,
            points=points,
            refinement_starts=refinement_starts,
        )
    )
    stiffness_scale = max(measured_scale, envelope_scale)
    sign_tolerance = max(
        SIGN_ABSOLUTE_TOLERANCE,
        SIGN_RELATIVE_TOLERANCE * stiffness_scale,
    )
    for row in measured_rows:
        row["stiffness_scale"] = stiffness_scale
        row["sign_tolerance"] = sign_tolerance
        row["classification"] = _classification(
            float(row["minimum_curvature"]),
            sign_tolerance,
            activation=bool(
                row["on_I4_activation"]
                or row["on_I6_activation"]
                or row.get("on_combined_coupling_activation", False)
            ),
        )
    for row in envelope_rows:
        row["stiffness_scale"] = stiffness_scale
        row["sign_tolerance"] = sign_tolerance
        for scope in SCOPE_ORDER:
            row[f"{scope}_classification"] = _classification(
                float(row[f"{scope}_minimum"]),
                sign_tolerance,
                activation=bool(
                    row["on_I4_activation"]
                    or row["on_I6_activation"]
                    or row.get("on_combined_coupling_activation", False)
                ),
            )

    measured_activation = any(
        bool(
            row["on_I4_activation"]
            or row["on_I6_activation"]
            or row.get("on_combined_coupling_activation", False)
        )
        for row in measured_rows
    )
    envelope_activation = any(
        bool(
            row["on_I4_activation"]
            or row["on_I6_activation"]
            or row.get("on_combined_coupling_activation", False)
        )
        for row in envelope_rows
    )
    summary_rows = []
    for scope in SCOPE_ORDER:
        summary_rows.append(
            _summary_row(
                measured_critical[scope],
                tolerance=sign_tolerance,
                stiffness_scale=stiffness_scale,
                activation_boundary_present=measured_activation,
            )
        )
        summary_rows.append(
            _summary_row(
                envelope_critical[scope],
                tolerance=sign_tolerance,
                stiffness_scale=stiffness_scale,
                activation_boundary_present=envelope_activation,
            )
        )
    return {
        "case": case,
        "polyconvexity": polyconvexity_counterexample(case),
        "measured_rows": measured_rows,
        "envelope_rows": envelope_rows,
        "summary_rows": summary_rows,
        "surface_data": surface_data,
        "stiffness_scale": stiffness_scale,
        "sign_tolerance": sign_tolerance,
    }


def _abaqus_voigt_to_matrix(values: Sequence[float]) -> np.ndarray:
    """Convert Abaqus [11,22,33,12,13,23] stress ordering to a matrix."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,):
        raise ValueError("Abaqus stress must contain six values")
    return np.asarray(
        [
            [vector[0], vector[3], vector[4]],
            [vector[3], vector[1], vector[5]],
            [vector[4], vector[5], vector[2]],
        ]
    )


def _compiled_cauchy_stress(F: np.ndarray, props: Sequence[float]) -> np.ndarray:
    """Call the loaded frozen Abaqus UMAT with a full, nonsymmetric-capable F."""
    # The source-level ``myumat`` helper leaves NTENS/NSTATV uninitialized.  The
    # production Abaqus entry point is therefore called directly with explicit,
    # deterministic arguments, matching the project's established wrapper.
    stress = np.zeros(6, dtype=np.double)
    statev = np.zeros(1, dtype=np.double)
    ddsdde = np.zeros(36, dtype=np.double)
    ddsddt = np.zeros(6, dtype=np.double)
    drplde = np.zeros(6, dtype=np.double)
    stran = np.zeros(6, dtype=np.double)
    dstran = np.zeros(6, dtype=np.double)
    time = np.zeros(2, dtype=np.double)
    predef = np.zeros(1, dtype=np.double)
    dpred = np.zeros(1, dtype=np.double)
    coords = np.zeros(3, dtype=np.double)
    drot = np.ascontiguousarray(np.eye(3).ravel(order="F"), dtype=np.double)
    properties = np.ascontiguousarray(props, dtype=np.double)
    deformation = np.ascontiguousarray(
        np.asarray(F, dtype=np.double).reshape(3, 3).ravel(order="F")
    )

    sse = ctypes.c_double(0.0)
    spd = ctypes.c_double(0.0)
    scd = ctypes.c_double(0.0)
    rpl = ctypes.c_double(0.0)
    drpldt = ctypes.c_double(0.0)
    dtime = ctypes.c_double(0.0)
    temp = ctypes.c_double(292.0)
    dtemp = ctypes.c_double(0.0)
    ndi = ctypes.c_int(3)
    nshr = ctypes.c_int(3)
    ntens = ctypes.c_int(6)
    nstatv = ctypes.c_int(1)
    nprops = ctypes.c_int(len(properties))
    pnewdt = ctypes.c_double(1.0)
    celent = ctypes.c_double(1.0)
    noel = ctypes.c_int(1)
    npt = ctypes.c_int(1)
    layer = ctypes.c_int(1)
    kspt = ctypes.c_int(1)
    kstep = ctypes.c_int(1)
    kinc = ctypes.c_int(1)
    cmname = ctypes.create_string_buffer(b"UMAT", 80)

    pointer = lambda array: array.ctypes.data_as(  # noqa: E731 - compact ABI helper
        ctypes.POINTER(ctypes.c_double)
    )
    umat.umatf.umat_(
        pointer(stress),
        pointer(statev),
        pointer(ddsdde),
        ctypes.byref(sse),
        ctypes.byref(spd),
        ctypes.byref(scd),
        ctypes.byref(rpl),
        pointer(ddsddt),
        pointer(drplde),
        ctypes.byref(drpldt),
        pointer(stran),
        pointer(dstran),
        pointer(time),
        ctypes.byref(dtime),
        ctypes.byref(temp),
        ctypes.byref(dtemp),
        pointer(predef),
        pointer(dpred),
        cmname,
        ctypes.byref(ndi),
        ctypes.byref(nshr),
        ctypes.byref(ntens),
        ctypes.byref(nstatv),
        pointer(properties),
        ctypes.byref(nprops),
        pointer(coords),
        pointer(drot),
        ctypes.byref(pnewdt),
        ctypes.byref(celent),
        pointer(deformation),
        pointer(deformation),
        ctypes.byref(noel),
        ctypes.byref(npt),
        ctypes.byref(layer),
        ctypes.byref(kspt),
        ctypes.byref(kstep),
        ctypes.byref(kinc),
    )
    return _abaqus_voigt_to_matrix(stress)


def _compiled_first_piola(F: np.ndarray, props: Sequence[float]) -> np.ndarray:
    """Convert the compiled Cauchy stress to P=J sigma F^-T."""
    sigma = _compiled_cauchy_stress(F, props)
    return float(np.linalg.det(F)) * sigma @ np.linalg.inv(F).T


def _same_activation_region(
    center: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    parameters: Mapping[str, float],
) -> bool:
    """Return True when a central difference stays within one smooth branch."""
    def region(F: np.ndarray) -> tuple[bool, bool]:
        i4, i6 = invariant_values(F, parameters)
        return (
            i4 > _parameter(parameters, "C1"),
            i6 > _parameter(parameters, "C2"),
        )

    return region(center) == region(plus) == region(minus)


def run_umat_validation_worker(
    model_id: str,
    request_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Compile one frozen model and validate requested smooth/one-sided states."""
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    source_path = pathlib.Path(payload["umat_source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Frozen UMAT source not found: {source_path}")
    umat.compile_umat_meson(source_path)
    specification = load_model_spec(model_id)
    results: list[dict[str, Any]] = []
    for request in payload["requests"]:
        print(f"UMAT validation: {request['request_id']}", flush=True)
        parameters = {name: float(value) for name, value in request["parameters"].items()}
        props = specification.pack_parameters(parameters)
        F = np.asarray(request["F"], dtype=float)
        if request["method"] == "one_sided_stress":
            analytic_P = first_piola_stress(F, parameters, model_id)
            compiled_P = _compiled_first_piola(F, props)
            difference = float(np.max(np.abs(analytic_P - compiled_P)))
            scale = max(float(np.max(np.abs(analytic_P))), 1.0e-14)
            tolerance = UMAT_ABSOLUTE_TOLERANCE + UMAT_RELATIVE_TOLERANCE * scale
            results.append(
                {
                    "request_id": request["request_id"],
                    "model_id": model_id,
                    "fit_code": request["fit_code"],
                    "knit": request["knit"],
                    "method": request["method"],
                    "scope": request["scope"],
                    "analytic_value": "",
                    "finite_difference_value": "",
                    "absolute_difference": difference,
                    "relative_difference": difference / scale,
                    "selected_step": "",
                    "step_spread": "",
                    "tolerance": tolerance,
                    "status": "passed" if difference <= tolerance else "failed",
                    "details": request.get("details", "activation one-sided branch"),
                }
            )
            continue

        polarization = np.asarray(request["polarization"], dtype=float)
        propagation = np.asarray(request["propagation"], dtype=float)
        polarization /= np.linalg.norm(polarization)
        propagation /= np.linalg.norm(propagation)
        perturbation = np.outer(polarization, propagation)
        analytic = rank_one_curvature(
            F, polarization, propagation, parameters, model_id
        )
        finite_differences: list[tuple[float, float]] = []
        for step in UMAT_STEP_SIZES:
            plus = F + step * perturbation
            minus = F - step * perturbation
            if np.linalg.det(plus) <= 0.0 or np.linalg.det(minus) <= 0.0:
                continue
            if not _same_activation_region(F, plus, minus, parameters):
                continue
            plus_P = _compiled_first_piola(plus, props)
            minus_P = _compiled_first_piola(minus, props)
            derivative = (plus_P - minus_P) / (2.0 * step)
            finite_differences.append((step, float(np.sum(derivative * perturbation))))
        if not finite_differences:
            raise ValueError(f"No smooth UMAT step was available for {request['request_id']}")
        selected_step, selected_value = min(
            finite_differences,
            key=lambda item: abs(item[1] - analytic),
        )
        difference = abs(selected_value - analytic)
        scale = max(abs(analytic), abs(selected_value), 1.0e-14)
        tolerance = UMAT_ABSOLUTE_TOLERANCE + UMAT_RELATIVE_TOLERANCE * scale
        step_spread = (
            abs(finite_differences[-1][1] - finite_differences[-2][1])
            if len(finite_differences) >= 2
            else 0.0
        )
        converged = step_spread <= 5.0 * tolerance
        results.append(
            {
                "request_id": request["request_id"],
                "model_id": model_id,
                "fit_code": request["fit_code"],
                "knit": request["knit"],
                "method": request["method"],
                "scope": request["scope"],
                "analytic_value": analytic,
                "finite_difference_value": selected_value,
                "absolute_difference": difference,
                "relative_difference": difference / scale,
                "selected_step": selected_step,
                "step_spread": step_spread,
                "tolerance": tolerance,
                "status": "passed" if difference <= tolerance and converged else "failed",
                "details": json.dumps(
                    {"steps": [step for step, _ in finite_differences]},
                    separators=(",", ":"),
                ),
            }
        )
    failed = [row for row in results if row["status"] != "passed"]
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if failed:
        raise ValueError(f"{len(failed)} compiled-UMAT validation rows failed")


def _critical_request(
    case: FitCase,
    summary: Mapping[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Convert one critical summary row into a UMAT curvature request."""
    F = np.diag(
        [
            float(summary["lambda_1"]),
            float(summary["lambda_2"]),
            float(summary["lambda_3"]),
        ]
    )
    return {
        "request_id": request_id,
        "method": "central_rank_one_curvature",
        "model_id": case.model_id,
        "knit": case.knit,
        "fit_code": case.fit_code,
        "scope": summary["scope"],
        "F": F.tolist(),
        "polarization": [
            float(summary["polarization_x"]),
            float(summary["polarization_y"]),
            float(summary["polarization_z"]),
        ],
        "propagation": [
            float(summary["propagation_x"]),
            float(summary["propagation_y"]),
            float(summary["propagation_z"]),
        ],
        "parameters": case.parameters,
        "details": f"{summary['domain']} critical mode",
    }


def build_umat_requests(case_results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build critical, representative, and activation-side UMAT requests by model."""
    requests_by_model: dict[str, dict[str, Any]] = {}
    for result in case_results:
        case: FitCase = result["case"]
        record = requests_by_model.setdefault(
            case.model_id,
            {"hashes": set(), "sources": set(), "requests": []},
        )
        umat_record = case.payload["input_files"]["umat_source"]
        record["hashes"].add(str(umat_record["sha256"]))
        record["sources"].add(str(umat_record["snapshot_path"]))
        for index, summary in enumerate(result["summary_rows"]):
            if (
                float(summary["minimum_curvature"]) < -float(summary["sign_tolerance"])
                and not bool(
                    summary["on_I4_activation"]
                    or summary["on_I6_activation"]
                    or summary.get("on_combined_coupling_activation", False)
                )
            ):
                record["requests"].append(
                    _critical_request(
                        case,
                        summary,
                        f"fit{case.fit_code}_critical_{index}_{summary['scope']}",
                    )
                )

        lambda_1_axis = result["surface_data"]["lambda_1"]
        lambda_2_axis = result["surface_data"]["lambda_2"]
        representative_1 = max(1.0 + 10.0 * ACTIVATION_OFFSET, float(np.mean(lambda_1_axis)))
        representative_2 = max(1.0 + 10.0 * ACTIVATION_OFFSET, float(np.mean(lambda_2_axis)))
        representative_F = np.diag(
            [
                representative_1,
                representative_2,
                1.0 / (representative_1 * representative_2),
            ]
        )
        for scope in SCOPE_ORDER:
            mode = minimize_mode_over_angles(
                representative_F,
                scope,
                case.parameters,
                case.model_id,
            )
            record["requests"].append(
                {
                    "request_id": f"fit{case.fit_code}_representative_{scope}",
                    "method": "central_rank_one_curvature",
                    "model_id": case.model_id,
                    "knit": case.knit,
                    "fit_code": case.fit_code,
                    "scope": scope,
                    "F": representative_F.tolist(),
                    "polarization": mode.polarization.tolist(),
                    "propagation": mode.propagation.tolist(),
                    "parameters": case.parameters,
                    "details": "smooth jointly active representative state",
                }
            )

        other_1 = min(max(1.05, float(lambda_1_axis[0])), float(lambda_1_axis[-1]))
        other_2 = min(max(1.05, float(lambda_2_axis[0])), float(lambda_2_axis[-1]))
        for family, side in ((1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)):
            lambda_1 = 1.0 + side * ACTIVATION_OFFSET if family == 1 else other_1
            lambda_2 = 1.0 + side * ACTIVATION_OFFSET if family == 2 else other_2
            F = np.diag([lambda_1, lambda_2, 1.0 / (lambda_1 * lambda_2)])
            record["requests"].append(
                {
                    "request_id": (
                        f"fit{case.fit_code}_activation_family{family}_"
                        f"{'plus' if side > 0 else 'minus'}"
                    ),
                    "method": "one_sided_stress",
                    "model_id": case.model_id,
                    "knit": case.knit,
                    "fit_code": case.fit_code,
                    "scope": f"activation_family_{family}",
                    "F": F.tolist(),
                    "parameters": case.parameters,
                    "details": "compiled/analytic first-Piola branch check",
                }
            )
    for model_id, record in requests_by_model.items():
        if len(record["hashes"]) != 1:
            raise ValueError(f"{model_id} fit artifacts do not share one frozen UMAT hash")
        # Any hash-identical frozen snapshot is sufficient for the isolated compile.
        record["umat_source"] = sorted(record["sources"])[0]
        record["hash"] = next(iter(record["hashes"]))
        del record["hashes"]
        del record["sources"]
    return requests_by_model


def run_isolated_umat_validations(
    case_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one temporary subprocess per frozen model and collect validation rows."""
    requests_by_model = build_umat_requests(case_results)
    all_rows: list[dict[str, Any]] = []
    temp_root = RELEASE_ARCHIVE_ROOT
    temp_root.mkdir(parents=True, exist_ok=True)
    for model_id in (model for model in MODEL_ORDER if model in requests_by_model):
        request_payload = requests_by_model[model_id]
        with tempfile.TemporaryDirectory(
            prefix=f"stability_{model_id.lower()}_",
            dir=temp_root,
        ) as directory:
            worker_directory = pathlib.Path(directory)
            request_path = worker_directory / "request.json"
            output_path = worker_directory / "result.json"
            request_path.write_text(
                json.dumps(request_payload, indent=2), encoding="utf-8"
            )
            command = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--worker-model",
                model_id,
                "--worker-request",
                str(request_path),
                "--worker-output",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                cwd=worker_directory,
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"UMAT validation worker for {model_id} failed with exit code "
                    f"{completed.returncode}"
                )
            all_rows.extend(json.loads(output_path.read_text(encoding="utf-8")))
    return all_rows


def _json_ready(value: Any) -> Any:
    """Convert NumPy/path/dataclass values to deterministic JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_csv_atomic(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a nonempty rectangular CSV atomically with stable field ordering."""
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    # Some domain-specific rows add refinement metadata.  Preserve first-seen
    # ordering while leaving those fields blank for rows where they do not apply.
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text_atomic(path: pathlib.Path, content: str) -> None:
    """Write one UTF-8 text artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write one indented JSON artifact atomically."""
    write_text_atomic(path, json.dumps(_json_ready(payload), indent=2) + "\n")


def _surface_norm(values: np.ndarray) -> Normalize:
    """Return a zero-aware color normalization for one LH surface."""
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot a surface without finite values")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower < 0.0 < upper:
        return TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1.0e-15):
        padding = max(abs(lower), 1.0) * 1.0e-6
        return Normalize(vmin=lower - padding, vmax=upper + padding)
    return Normalize(vmin=lower, vmax=upper)


def write_case_heatmap(
    output_path: pathlib.Path,
    result: Mapping[str, Any],
) -> None:
    """Write one three-panel LH heatmap with measured states and critical stars."""
    case: FitCase = result["case"]
    surface = result["surface_data"]
    lambda_1 = np.asarray(surface["lambda_1"], dtype=float)
    lambda_2 = np.asarray(surface["lambda_2"], dtype=float)
    measured = measured_states(case)
    measured_1 = np.asarray([state["lambda_1"] for state in measured])
    measured_2 = np.asarray([state["lambda_2"] for state in measured])
    summary_by_scope = {
        row["scope"]: row
        for row in result["summary_rows"]
        if row["domain"] == "incompressible_envelope"
    }
    titles = {
        "full_3d": "Full 3D weak LH minimum",
        "membrane": "Unconstrained membrane minimum",
        "incompressible_membrane": "Incompressible membrane minimum",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for axis, scope in zip(axes, SCOPE_ORDER):
        values = np.asarray(surface["values"][scope], dtype=float)
        image = axis.pcolormesh(
            lambda_1,
            lambda_2,
            values.T,
            shading="auto",
            cmap="coolwarm",
            norm=_surface_norm(values),
        )
        axis.scatter(
            measured_1,
            measured_2,
            s=5,
            c="black",
            alpha=0.24,
            linewidths=0.0,
            label="raw B09 states",
        )
        critical = summary_by_scope[scope]
        axis.scatter(
            [float(critical["lambda_1"])],
            [float(critical["lambda_2"])],
            marker="*",
            s=135,
            c="#FFD700",
            edgecolors="black",
            linewidths=0.8,
            label="refined minimum",
            zorder=5,
        )
        axis.axvline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.55)
        axis.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.55)
        axis.set_title(titles[scope])
        axis.set_xlabel(r"$\lambda_1$ (axis 1)")
        axis.set_ylabel(r"$\lambda_2$ (axis 2)")
        axis.legend(loc="best", fontsize=7)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
        colorbar.set_label("minimum rank-one curvature")
    figure.suptitle(
        f"{case.model_id} - {case.knit} - raw B09 fit {case.fit_code}",
        fontsize=14,
    )
    temporary = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output_path)


def _format_fit_table() -> list[str]:
    """Return README rows for the fixed eight-fit inventory."""
    lines = [
        "| Model | Interlock | Rib | Purl | Single Jersey |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        mapping = FIT_CODES[model_id]
        lines.append(
            f"| {model_id} | {mapping['Interlock']} | {mapping['Rib']} | "
            f"{mapping['Purl']} | {mapping['SingleJersey']} |"
        )
    return lines


def readme_text(
    case_results: Sequence[Mapping[str, Any]],
    umat_rows: Sequence[Mapping[str, Any]],
    *,
    compiled_umat_validated: bool,
) -> str:
    """Build the study's methods, provenance, results, and interpretation guide."""
    poly_rows = [result["polyconvexity"] for result in case_results]
    summary_rows = [
        row for result in case_results for row in result["summary_rows"]
    ]
    violated = [
        row for row in summary_rows if row["weak_lh_smooth_status"] == "violated"
    ]
    lines = [
        "# Polyconvexity and Legendre-Hadamard Stability Study",
        "",
        "This fixed-fit diagnostic analyzes the registered raw prescribed-B09",
        "training optima. It does not refit parameters, change either UMAT, or",
        "write to the fit registry.",
        "",
        "## Fit inventory",
        "",
        *_format_fit_table(),
        "",
        "Every fit was validated as a complete `training_optimum`; all recorded",
        "source and snapshot hashes were checked. Its B09 array was matched",
        "exactly to the prescribed concatenation of all nine authoritative raw",
        "ratio curves. StapletonViaux additionally requires",
        "`K_2ani1 = K_2ani2 = 0`.",
        "",
        "## Global polyconvexity result",
        "",
        "For each fit, use the admissible rank-one line",
        "`F(t) = I + (t-1) e1 tensor e1`. At `t=0.5`, all nonlinear terms are",
        "inactive, `W(t)=mu*(t^2-1)^2`, and `W''(0.5)=-mu`. Because every",
        "selected fit has `mu>0`, all eight second variations are negative.",
        "Therefore all eight selected fitted energies fail rank-one convexity and",
        "cannot be globally polyconvex. This is an analytical counterexample, not",
        "a sampling inference.",
        "",
        f"Validated analytical counterexamples: `{len(poly_rows)}`.",
        "",
        "## Legendre-Hadamard scopes",
        "",
        "- `full_3d`: all polarizations for in-plane propagation plus the exact",
        "  `n=e3` zero mode. Strict 3D strong ellipticity therefore always fails.",
        "- `membrane`: propagation and polarization are both in the material plane.",
        "- `incompressible_membrane`: membrane polarizations additionally satisfy",
        "  `a dot F^-T n = 0`.",
        "",
        "Each scope was evaluated at every raw B09 state and on each knit's",
        f"`{STRETCH_GRID_POINTS} x {STRETCH_GRID_POINTS}` lambda envelope using",
        f"`{ANGLE_GRID_POINTS}` propagation angles. The lowest",
        f"`{LOCAL_REFINEMENT_STARTS}` grid candidates were refined with explicit",
        "corners, edges, activation lines, and one-sided activation offsets.",
        "Envelope thickness is `lambda3=1/(lambda1*lambda2)`.",
        "",
        "## Operational result summary",
        "",
        "| Model | Knit | Domain | Scope | Minimum | Smooth-set status | Critical lambda1 | Critical lambda2 |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_id} | {knit} | {domain} | {scope} | "
            "{minimum_curvature:.9e} | {weak_lh_smooth_status} | "
            "{lambda_1:.9g} | {lambda_2:.9g} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Validated negative operational counterexamples: `{len(violated)}`",
            f"of `{len(summary_rows)}` domain/scope summaries.",
            "",
            "Exact activation states are labeled `classically undefined at activation boundary`",
            "because the implemented branch tangent need not",
            "be C2 there. Smooth one-sided limits are retained separately.",
            "",
            "## Sign and validation rules",
            "",
            "For each model-knit case, `tau=max(1e-10, 1e-8*stiffness_scale)`.",
            "A smooth value below `-tau` is a validated LH violation. A nonnegative",
            "finite screen is reported only as `no violation found`; it is not a",
            "continuous-domain proof of polyconvexity or ellipticity.",
            "",
            (
                f"Compiled-UMAT validation passed `{len(umat_rows)}` rows using "
                "central rank-one curvature differences and one-sided activation "
                "stress checks."
                if compiled_umat_validated
                else "Compiled-UMAT validation was skipped by explicit debug control."
            ),
            "",
            "## Files",
            "",
            f"- `{POLYCONVEXITY_CSV}`: analytical global counterexamples.",
            f"- `{LH_SUMMARY_CSV}`: one critical row per case/domain/scope.",
            f"- `{MEASURED_CSV}`: every raw state and its minimized directions.",
            f"- `{ENVELOPE_CSV}`: heatmap grid values and classifications.",
            f"- `{UMAT_CSV}`: compiled-Fortran agreement evidence.",
            f"- `{SUMMARY_JSON}`: configuration, provenance, and summary records.",
            f"- `{FIGURE_DIRECTORY}/`: eight three-panel heatmaps.",
            "",
            "## References",
            "",
            "- [Ball (1976), Convexity Conditions and Existence Theorems in Nonlinear Elasticity](https://doi.org/10.1007/BF00279992).",
            "- [On Strong Ellipticity](https://www2.egr.uh.edu/~ychen11/PUBLICATIONS/%5B6%5D.pdf).",
            "",
        ]
    )
    return "\n".join(lines)


def output_paths(output_directory: pathlib.Path = OUTPUT_DIRECTORY) -> tuple[pathlib.Path, ...]:
    """Return all guarded top-level study artifacts."""
    return tuple(
        output_directory / filename
        for filename in (
            POLYCONVEXITY_CSV,
            LH_SUMMARY_CSV,
            MEASURED_CSV,
            ENVELOPE_CSV,
            UMAT_CSV,
            SUMMARY_JSON,
            README_FILE,
        )
    )


def _study_provenance(
    case_results: Sequence[Mapping[str, Any]],
    umat_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable configuration and immutable-input provenance."""
    cases = []
    for result in case_results:
        case: FitCase = result["case"]
        cases.append(
            {
                "model_id": case.model_id,
                "knit": case.knit,
                "fit_code": case.fit_code,
                "case_id": case.payload["case_id"],
                "result_role": case.payload["result_role"],
                "parameters": case.parameters,
                "input_hashes": {
                    label: record["sha256"]
                    for label, record in case.payload["input_files"].items()
                },
                "lambda_1_bounds": [
                    float(result["surface_data"]["lambda_1"][0]),
                    float(result["surface_data"]["lambda_1"][-1]),
                ],
                "lambda_2_bounds": [
                    float(result["surface_data"]["lambda_2"][0]),
                    float(result["surface_data"]["lambda_2"][-1]),
                ],
                "stiffness_scale": result["stiffness_scale"],
                "sign_tolerance": result["sign_tolerance"],
            }
        )
    return {
        "schema_version": 1,
        "study_id": "raw_b09_polyconvexity_legendre_hadamard_v1",
        "configuration": {
            "fit_codes": FIT_CODES,
            "stretch_grid_points": STRETCH_GRID_POINTS,
            "angle_grid_points": ANGLE_GRID_POINTS,
            "local_refinement_starts": LOCAL_REFINEMENT_STARTS,
            "sign_absolute_tolerance": SIGN_ABSOLUTE_TOLERANCE,
            "sign_relative_tolerance": SIGN_RELATIVE_TOLERANCE,
            "activation_tolerance": ACTIVATION_TOLERANCE,
            "activation_offset": ACTIVATION_OFFSET,
            "umat_step_sizes": UMAT_STEP_SIZES,
            "compiled_umat_validated": bool(umat_rows),
        },
        "cases": cases,
        "polyconvexity": [result["polyconvexity"] for result in case_results],
        "legendre_hadamard_summary": [
            row for result in case_results for row in result["summary_rows"]
        ],
        "umat_validation": list(umat_rows),
    }


def execute_study(
    *,
    overwrite: bool,
    validate_with_umat: bool,
    output_directory: pathlib.Path = OUTPUT_DIRECTORY,
    points: int = STRETCH_GRID_POINTS,
    angle_points: int = ANGLE_GRID_POINTS,
    refinement_starts: int = LOCAL_REFINEMENT_STARTS,
) -> pathlib.Path:
    """Run the complete fixed-fit study and return the summary CSV path."""
    guarded = list(output_paths(output_directory))
    figure_directory = output_directory / FIGURE_DIRECTORY
    guarded.extend(
        figure_directory / f"{model.lower()}_{knit.lower().replace(' ', '_')}.png"
        for model in MODEL_ORDER
        for knit in KNIT_ORDER
    )
    existing = [path for path in guarded if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Stability-study output exists; set OVERWRITE_EXISTING=True or use "
            f"--overwrite: {existing}"
        )

    cases = load_fit_cases()
    angles = propagation_angles(angle_points)
    case_results = []
    for case in cases:
        print(
            f"Analyzing {case.model_id} {case.knit} fit {case.fit_code}: "
            "measured paths and incompressible envelope"
        )
        case_results.append(
            analyze_case(
                case,
                angles,
                points=points,
                refinement_starts=refinement_starts,
            )
        )

    umat_rows = (
        run_isolated_umat_validations(case_results) if validate_with_umat else []
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    poly_rows = [result["polyconvexity"] for result in case_results]
    summary_rows = [
        row for result in case_results for row in result["summary_rows"]
    ]
    measured_rows = [
        row for result in case_results for row in result["measured_rows"]
    ]
    envelope_rows = [
        row for result in case_results for row in result["envelope_rows"]
    ]
    write_csv_atomic(output_directory / POLYCONVEXITY_CSV, poly_rows)
    write_csv_atomic(output_directory / LH_SUMMARY_CSV, summary_rows)
    write_csv_atomic(output_directory / MEASURED_CSV, measured_rows)
    write_csv_atomic(output_directory / ENVELOPE_CSV, envelope_rows)
    if umat_rows:
        write_csv_atomic(output_directory / UMAT_CSV, umat_rows)
    else:
        write_csv_atomic(
            output_directory / UMAT_CSV,
            [
                {
                    "request_id": "validation_skipped",
                    "model_id": "",
                    "fit_code": "",
                    "knit": "",
                    "method": "",
                    "scope": "",
                    "analytic_value": "",
                    "finite_difference_value": "",
                    "absolute_difference": "",
                    "relative_difference": "",
                    "selected_step": "",
                    "step_spread": "",
                    "tolerance": "",
                    "status": "skipped",
                    "details": "Explicit debug control disabled compiled validation",
                }
            ],
        )
    for result in case_results:
        case: FitCase = result["case"]
        figure_path = figure_directory / (
            f"{case.model_id.lower()}_{case.knit.lower().replace(' ', '_')}.png"
        )
        write_case_heatmap(figure_path, result)
    write_json_atomic(
        output_directory / SUMMARY_JSON,
        _study_provenance(case_results, umat_rows),
    )
    write_text_atomic(
        output_directory / README_FILE,
        readme_text(
            case_results,
            umat_rows,
            compiled_umat_validated=validate_with_umat,
        ),
    )
    print(f"Wrote {len(summary_rows)} LH summary rows to {output_directory}")
    return output_directory / LH_SUMMARY_CSV


def parse_args() -> argparse.Namespace:
    """Parse normal user controls and hidden isolated-worker controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this study's existing outputs.",
    )
    parser.add_argument(
        "--skip-umat-validation",
        action="store_true",
        help="Debug only: skip compiled-Fortran validation.",
    )
    parser.add_argument("--worker-model", choices=MODEL_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--worker-request", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Run either one isolated UMAT worker or the complete user-facing study."""
    args = parse_args()
    if args.worker_model is not None:
        if args.worker_request is None or args.worker_output is None:
            raise ValueError(
                "--worker-request and --worker-output are required with --worker-model"
            )
        run_umat_validation_worker(
            args.worker_model,
            args.worker_request,
            args.worker_output,
        )
        return
    execute_study(
        overwrite=OVERWRITE_EXISTING or args.overwrite,
        validate_with_umat=(
            VALIDATE_WITH_COMPILED_UMAT and not args.skip_umat_validation
        ),
    )


if __name__ == "__main__":
    main()
