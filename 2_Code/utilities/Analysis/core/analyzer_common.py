"""Shared mechanics, data loading, and plotting for family analyzers.

The implementation follows the proposed paper model

    W = W_NL(I4, I5, I6, I7) + W_L(I4, I6)

and computes the stress contributions in second Piola-Kirchhoff form, then
converts each contribution directly to first Piola-Kirchhoff stress for plotting.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (UTILITIES_ROOT as RELEASE_UTILITIES_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    EXPERIMENT_ROOT as RELEASE_EXPERIMENT_ROOT)

import argparse
from collections.abc import Mapping, Sequence
import os
import pathlib as pl
import sys
from typing import Callable, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

curr_file = os.fspath(
    next(parent for parent in pl.Path(__file__).resolve().parents if parent.name == "Analysis")
)
parent_dir = os.path.dirname(curr_file)
sys.path.insert(0, parent_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from Python_Wrapper import mech, umat
from Python_Wrapper.model_registry import discover_model_specs, load_model_spec
from Analysis.core.fit_results import DEFAULT_RESULTS_CSV, FitResult, load_fit
import uml_plot_style as uml_style


ParameterMap = Mapping[str, float | int]

# ---------------------------------------------------------------------------
# Path settings
# ---------------------------------------------------------------------------
# Edit these paths when you move the analysis files, material models, input
# data, or desired output location. Keeping them here makes the script easier
# to run from VS Code without passing command-line arguments.
PROJECT_DIR = RELEASE_UTILITIES_ROOT
EXPERIMENTAL_DATA_ROOT = RELEASE_EXPERIMENT_ROOT
DEFAULT_SAVE_DIR = pl.Path(__file__).resolve().parents[3] / "generated/figures/model_analyzer"

DEFAULT_MODEL = "SV_exp_polycon_mo"
# Set this to an integer to use a fit when pressing Play in VS Code.
DEFAULT_FIT_CODE: int | None = 2166
DEFAULT_KNIT = "SingleJersey"
DEFAULT_RATIO = "0.75 to 1"
DEFAULT_DATA_COLLECTION = "Raw"  # Use "Raw" for the shipped compiled curves.
DEFAULT_YARN_TYPE = "Tripoly"  # Options in the data folder: "Tripoly".
DEFAULT_RAW_SAMPLE_FOLDER = "BRT_15mm_Tri_SJ_S02"
DEFAULT_RAW_DATA_SUBDIR = "Compiled_Data"
DEFAULT_COMPONENTS = ("xx", "yy")
EXP_TYPE = ["H", "P"]
UMAT_TYPE = ["F", "Sigma"]
P_STRESS_LABEL = r"First PK stress $P$"

# These are the constitutive decompositions implemented in this script. Models
# with other analysis families remain available to generic tools such as
# plotter.py, but require a dedicated analyzer adapter before use here.
SUPPORTED_ANALYSIS_FAMILIES = frozenset(
    {
        "stapleton_viaux",
        "stapleton_viaux_logarithmic",
        "stapleton_viaux_exponential",
        "stapleton_viaux_exponential_second_order",
        "stapleton_viaux_tanh",
        "faseranisotropie_sv",
    }
)

# Turn figures on/off here before pressing Play in VS Code.
PLOTS_TO_SHOW = {
    "linear_vs_nonlinear": True,
    "linear_terms": True,
    "nonlinear_terms": True,
    "linear_parameters": True,
    "nonlinear_parameters": True,
    "invariant_diagnostics_xx": True,
    "invariant_diagnostics_yy": True,
}

# Standalone parameter bar plot sizing. Edit these if you want larger/smaller
# PowerPoint-ready parameter figures without touching the plotting logic.
LINEAR_PARAM_FIGSIZE = (2.0, 1.35)
NONLINEAR_PARAM_FIGSIZE = (3, 2.05)
LINEAR_PARAM_GROUP_SPACING = 0.48
NONLINEAR_PARAM_GROUP_SPACING = 0.8
PARAM_BAR_WIDTH = 0.16
PARAM_VALUE_FONTSIZE = 10
PARAM_LABEL_FONTSIZE = 9
PARAM_TITLE_FONTSIZE = 10

COMPONENTS = {
    "xx": ((0, 0), 0),
    "yy": ((1, 1), 1),
    "zz": ((2, 2), 2),
    "yz": ((1, 2), 3),
    "xz": ((0, 2), 4),
    "xy": ((0, 1), 5),
}

DATA_COLLECTIONS = ("Raw",)
YARN_FOLDERS = {
    "Tripoly": "Tripoly",
}
KNIT_FILE_SUFFIXES = {
    "Interlock": "interlock",
    "Rib": "rib",
    "Purl": "purl",
    "SingleJersey": "singlejersey",
}

LINEAR_TERMS = (
    ("mu_1", r"Course linear ($\mu_1$)"),
    ("mu_2", r"Wales linear ($\mu_2$)"),
    ("mu_3", r"Coupling linear ($\mu_3$)"),
)

# The paper model has five nonlinear stress terms: two tied to M1
# (course direction), two tied to M2 (wales direction), and one coupling term.
NONLINEAR_TERMS = (
    ("K_1ani1", r"Course $I_4$ ($K^{ani}_{1,1}$, $\beta_1$)"),
    ("K_2ani1", r"Course $I_5$ ($K^{ani}_{2,1}$, $\beta_2$)"),
    ("K_1ani2", r"Wales $I_6$ ($K^{ani}_{1,2}$, $\gamma_1$)"),
    ("K_2ani2", r"Wales $I_7$ ($K^{ani}_{2,2}$, $\gamma_2$)"),
    ("K_coup", r"Coupling ($K_{coup}$, $\zeta$)"),
)

LINEAR_PARAMETER_BARS = (
    ("mu_1", r"$\mu_1$", ("mu",)),
    ("mu_2", r"$\mu_2$", ("mu2",)),
    ("mu_3", r"$\mu_3$", ("mu3",)),
)

NONLINEAR_PARAMETER_BARS = (
    ("K_1ani1", r"$K^{ani}_{1,1}$, $\beta_1$", ("K_1ani1", "beta_1")),
    ("K_2ani1", r"$K^{ani}_{2,1}$, $\beta_2$", ("K_2ani1", "beta_2")),
    ("K_1ani2", r"$K^{ani}_{1,2}$, $\gamma_1$", ("K_1ani2", "gamma_1")),
    ("K_2ani2", r"$K^{ani}_{2,2}$, $\gamma_2$", ("K_2ani2", "gamma_2")),
    ("K_coup", r"$K_{coup}$, $\zeta$", ("K_coup", "zeta")),
)

TERM_COLORS = {
    "linear": "#0072B2",
    "nonlinear": "#D55E00",
    "mu_1": "#0072B2",
    "mu_2": "#009E73",
    "mu_3": "#D55E00",
    "K_1ani1": "#0072B2",
    "K_2ani1": "#56B4E9",
    "K_1ani2": "#009E73",
    "K_2ani2": "#6CC24A",
    "K_coup": "#D55E00",
}

INVARIANT_COLORS = {
    "I4": "#0072B2",
    "I5": "#56B4E9",
    "I6": "#009E73",
    "I7": "#6CC24A",
    "coupling_M1": "#D55E00",
    "coupling_M2": "#E69F00",
}

DecomposeFunction = Callable[
    [np.ndarray, ParameterMap, str], dict[str, np.ndarray]
]
DiagnosticsFunction = Callable[
    [np.ndarray, ParameterMap, str], dict[str, np.ndarray]
]
ConvertFunction = Callable[[ParameterMap, str], dict[str, float | int]]


def main(
    *,
    supported_analysis_families: frozenset[str] = SUPPORTED_ANALYSIS_FAMILIES,
    default_model: str = DEFAULT_MODEL,
    default_fit_code: int | None = DEFAULT_FIT_CODE,
    decompose_fn: DecomposeFunction | None = None,
    diagnostics_fn: DiagnosticsFunction | None = None,
    convert_fn: ConvertFunction | None = None,
) -> None:
    """Run the shared analyzer workflow for one constitutive-model family."""
    model_choices = analyzable_model_ids(supported_analysis_families)
    args = parse_args(model_choices, default_model, default_fit_code)
    uml_style.apply()
    decompose_fn = decompose_fn or decompose_history
    diagnostics_fn = diagnostics_fn or invariant_diagnostics_history
    convert_fn = convert_fn or convert_parameters

    selected_fit: FitResult | None = None
    if args.fit_code is not None:
        selected_fit = load_fit(args.fit_code, args.results_csv)
    if args.model is None:
        args.model = selected_fit.model if selected_fit is not None else default_model
    elif selected_fit is not None and args.model != selected_fit.model:
        raise ValueError(
            f"Fit {selected_fit.code} belongs to {selected_fit.model}, not {args.model}"
        )
    model_spec = load_model_spec(args.model)
    if model_spec.analysis_family not in supported_analysis_families:
        supported = ", ".join(model_choices)
        raise ValueError(
            f"This analyzer has no decomposition for {args.model!r} "
            f"(analysis family {model_spec.analysis_family!r}); supported models: "
            f"{supported}"
        )

    components = tuple(normalize_components(args.components))
    if selected_fit is None:
        param_all = material_parameters()
        parameters = param_all[args.knit]

        # The built-in dictionaries store the larger faseranisotropie_SV set.
        # Convert it so the same defaults can drive either supported model.
        model_parameters = convert_fn(parameters, args.model)
    else:
        # Historical SV_log fits may contain the removed K2 parameters.
        model_parameters = {
            name: selected_fit.parameters[name]
            for name in model_spec.parameter_names
        }
        print(
            f"Using fit {selected_fit.code} ({args.model}) from {selected_fit.source}"
        )
    sim_par = np.asarray(model_spec.pack_parameters(model_parameters), dtype=float)

    exp_path = experimental_path(
        args.knit,
        args.ratio,
        args.data_dir,
        args.data_collection,
        args.yarn_type,
        args.raw_sample_folder,
    )
    exp_data = np.loadtxt(exp_path)
    exp_strains = exp_data[:, :6]
    dfgrd_voigt = umat.strains_to_dfgrd(exp_strains, EXP_TYPE[0])

    print(f"Plotting {args.model} for {args.knit}, ratio '{args.ratio}'")
    print(f"Data selection: {args.data_collection}, {args.yarn_type}")
    print(f"Experimental data: {exp_path}")

    decomposed = decompose_fn(dfgrd_voigt, model_parameters, args.model)

    # The UMAT returns Cauchy stress. Convert it once to first PK stress so the
    # validation matches the stress measure used in the plots.
    if not args.no_umat:
        try:
            umat.compile_umat_meson(model_spec.fortran_source)
            umat_stress = call_umat_history(dfgrd_voigt, sim_par)
            umat_p_stress = cauchy_history_to_pk1(dfgrd_voigt, umat_stress)
            max_err = np.max(np.abs(umat_p_stress - decomposed["total"]))
            print(f"Max |UMAT - decomposed total| in first PK stress: {max_err:.6e}")
        except Exception as exc:
            if args.strict_umat:
                raise
            print(f"UMAT validation skipped: {exc}")

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if PLOTS_TO_SHOW["linear_vs_nonlinear"]:
        plot_linear_vs_nonlinear(exp_strains, decomposed, components, args, save_dir)
    if PLOTS_TO_SHOW["linear_terms"]:
        plot_linear_terms(exp_strains, decomposed, components, args, save_dir)
    if PLOTS_TO_SHOW["nonlinear_terms"]:
        plot_nonlinear_terms(exp_strains, decomposed, components, args, save_dir)
    if PLOTS_TO_SHOW["linear_parameters"]:
        plot_linear_parameters(model_parameters, args, save_dir)
    if PLOTS_TO_SHOW["nonlinear_parameters"]:
        plot_nonlinear_parameters(model_parameters, args, save_dir)
    if PLOTS_TO_SHOW["invariant_diagnostics_xx"]:
        plot_invariant_diagnostics(
            exp_strains, dfgrd_voigt, model_parameters, args, save_dir, "xx",
            diagnostics_fn,
        )
    if PLOTS_TO_SHOW["invariant_diagnostics_yy"]:
        plot_invariant_diagnostics(
            exp_strains, dfgrd_voigt, model_parameters, args, save_dir, "yy",
            diagnostics_fn,
        )
    plt.show()


def parse_args(
    model_choices: Sequence[str],
    default_model: str,
    default_fit_code: int | None,
) -> argparse.Namespace:
    """Parse plot configuration."""
    parser = argparse.ArgumentParser(
        description="Plot stress contribution components for the SSMCCF material model."
    )
    parser.add_argument(
        "--model",
        choices=tuple(model_choices),
        default=None,
        help=(
            f"Fortran UMAT/model to analyze. Default: model stored by --fit, "
            f"otherwise {default_model}."
        ),
    )
    parser.add_argument(
        "--fit-code",
        "--fit",
        dest="fit_code",
        type=int,
        default=default_fit_code,
        help="Fit code from results_final.csv. If omitted, use built-in parameters.",
    )
    parser.add_argument(
        "--results-csv",
        type=pl.Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Fit-results CSV. Default: {DEFAULT_RESULTS_CSV}",
    )
    parser.add_argument(
        "--knit",
        choices=tuple(material_parameters().keys()),
        default=DEFAULT_KNIT,
        help="Knit type parameter set to use.",
    )
    parser.add_argument(
        "--ratio",
        default=DEFAULT_RATIO,
        help="Experimental loading ratio label, e.g. '1 to 0' or '1 to 1'.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        help="Stress components to plot: xx yy zz xy xz yz.",
    )
    parser.add_argument(
        "--data-collection",
        choices=DATA_COLLECTIONS,
        default=DEFAULT_DATA_COLLECTION,
        help=(
            "Experimental-data collection to use. "
            "'Raw' uses a shipped specimen folder."
        ),
    )
    parser.add_argument(
        "--yarn-type",
        choices=tuple(YARN_FOLDERS.keys()),
        default=DEFAULT_YARN_TYPE,
        help="Yarn folder to use inside the experimental-data directory.",
    )
    parser.add_argument(
        "--raw-sample-folder",
        default=DEFAULT_RAW_SAMPLE_FOLDER,
        help="Sample folder used only when --data-collection Raw is selected.",
    )
    parser.add_argument(
        "--data-dir",
        type=pl.Path,
        default=None,
        help=(
            "Optional manual directory override. If set, this directory is searched "
            "before the hardcoded experimental-data folders."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=pl.Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory where SVG figures will be saved. Default: {DEFAULT_SAVE_DIR}",
    )
    parser.add_argument(
        "--no-umat",
        action="store_true",
        help="Skip compiling/calling the Fortran UMAT validation.",
    )
    parser.add_argument(
        "--strict-umat",
        action="store_true",
        help="Raise UMAT compile/call errors instead of warning and continuing.",
    )
    return parser.parse_args()


def analyzable_model_ids(
    supported_analysis_families: frozenset[str] = SUPPORTED_ANALYSIS_FAMILIES,
) -> tuple[str, ...]:
    """Return discovered models supported by this analyzer's equations."""
    return tuple(
        model_id
        for model_id, model in discover_model_specs().items()
        if model.analysis_family in supported_analysis_families
    )


def material_parameters() -> dict[str, dict[str, float | int]]:
    """Named built-in material parameters copied from plotter.py."""
    return {
        # Optimum parameters for clean data Tri_S06 training with clean 1st badge
        # 'Interlock': {
        #     'mu':       0.017211392,      # Linear
        #     'mu2':      0.003616,         # Linear
        #     'K_iso1':   0,                # Isotropic
        #     'K_iso2':   0,                # Isotropic
        #     'alpha_1':  3,                # Isotropic
        #     'alpha_2':  3,                # Isotropic
        #     'K_1ani1':  0.101560115,      # Anisotropic
        #     'beta_1':   20,               # Anisotropic
        #     'K_2ani1':  0,                # Anisotropic
        #     'beta_2':   3,              # Anisotropic
        #     'K_1ani2':  1.637823996,      # Anisotropic
        #     'gamma_1':  71,               # Anisotropic
        #     'K_2ani2':  0,                # Anisotropic
        #     'gamma_2':  9,                # Anisotropic
        #     'K_kop1':   0,                # Coupling
        #     'delta_1':  3,                # Coupling
        #     'K_kop2':   0,                # Coupling
        #     'delta_2':  3,                # Coupling
        #     'K_kop12':  0.540305347,      # Coupling
        #     'zeta':     3,                # Coupling
        #     'C1':       1,
        #     'C2':       1,
        #     'theta1':   0,
        #     'theta2':   90,
        #     'mu3':      0.011665408
        # },
        # Raw data optimum
        'Interlock': {
            'mu':       0.025316909,      # Linear
            'mu2':      0.003265676,      # Linear
            'K_iso1':   0,                # Isotropic
            'K_iso2':   0,                # Isotropic
            'alpha_1':  3,                # Isotropic
            'alpha_2':  3,                # Isotropic
            'K_1ani1':  31.89096877,      # Anisotropic
            'beta_1':   68,               # Anisotropic
            'K_2ani1':  0,                # Anisotropic
            'beta_2':   32,               # Anisotropic
            'K_1ani2':  1.833672105,      # Anisotropic
            'gamma_1':  15,               # Anisotropic
            'K_2ani2':  0,                # Anisotropic
            'gamma_2':  50,               # Anisotropic
            'K_kop1':   0,                # Coupling
            'delta_1':  3,                # Coupling
            'K_kop2':   0,                # Coupling
            'delta_2':  3,                # Coupling
            'K_kop12':  4.070181355,      # Coupling
            'zeta':     4,                # Coupling
            'C1':       1,
            'C2':       1,
            'theta1':   14.95597629,
            'theta2':   89.03751361,
            'mu3':      0.012060468
        },
        "Rib": {
            "mu": 0.001905792,
            "mu2": 0.003253939,
            "K_iso1": 0,
            "K_iso2": 0,
            "alpha_1": 3,
            "alpha_2": 3,
            "K_1ani1": 0.000678486,
            "beta_1": 20,
            "K_2ani1": 0.346985549,
            "beta_2": 607,
            "K_1ani2": 0.91156323,
            "gamma_1": 103,
            "K_2ani2": 0,
            "gamma_2": 7,
            "K_kop1": 0,
            "delta_1": 3,
            "K_kop2": 0,
            "delta_2": 3,
            "K_kop12": 7.901339204,
            "zeta": 8,
            "C1": 1,
            "C2": 1,
            "theta1": 0,
            "theta2": 90,
            "mu3": 0.001443221,
        },
        "Purl": {
            "mu": 0.015217527,
            "mu2": 0.005943088,
            "K_iso1": 0,
            "K_iso2": 0,
            "alpha_1": 3,
            "alpha_2": 3,
            "K_1ani1": 3.445545642,
            "beta_1": 11,
            "K_2ani1": 0.220635979,
            "beta_2": 588,
            "K_1ani2": 0.214419753,
            "gamma_1": 109,
            "K_2ani2": 0,
            "gamma_2": 7,
            "K_kop1": 0,
            "delta_1": 3,
            "K_kop2": 0,
            "delta_2": 3,
            "K_kop12": 4.907433976,
            "zeta": 4,
            "C1": 1,
            "C2": 1,
            "theta1": 0,
            "theta2": 90,
            "mu3": 0.0088,
        },
        "SingleJersey": {
            "mu": 0.0157652378524096,
            "mu2": 0.0184662792724592,
            "K_iso1": 0,
            "K_iso2": 0,
            "alpha_1": 3,
            "alpha_2": 3,
            "K_1ani1": 2.19573871314823,
            "beta_1": 75,
            "K_2ani1": 0,
            "beta_2": 513,
            "K_1ani2": 0.264034705035234,
            "gamma_1": 82,
            "K_2ani2": 3.16116991988597e-07,
            "gamma_2": 11,
            "K_kop1": 0,
            "delta_1": 3,
            "K_kop2": 0,
            "delta_2": 3,
            "K_kop12": 3.9350731910975,
            "zeta": 3,
            "C1": 1,
            "C2": 1,
            "theta1": 0,
            "theta2": 90,
            "mu3": 0.00999999205595173,
        },
    }


def convert_parameters(params: ParameterMap, model: str) -> dict[str, float | int]:
    """Convert the full built-in parameter set to the selected model's names."""
    if model == "faseranisotropie_SV":
        return dict(params)
    if model in {"StapletonViaux", "SV_exp_2O"}:
        # StapletonViaux removes the SV isotropic terms and the two isotropic
        # coupling terms. The remaining nonlinear coupling is K_coup, which is
        # the same quantity stored as K_kop12 in the SV parameter dictionary.
        return {
            "mu": params["mu"],
            "mu2": params["mu2"],
            "K_1ani1": params["K_1ani1"],
            "beta_1": params["beta_1"],
            "K_2ani1": params["K_2ani1"],
            "beta_2": params["beta_2"],
            "K_1ani2": params["K_1ani2"],
            "gamma_1": params["gamma_1"],
            "K_2ani2": params["K_2ani2"],
            "gamma_2": params["gamma_2"],
            "K_coup": params["K_kop12"],
            "zeta": params["zeta"],
            "C1": params["C1"],
            "C2": params["C2"],
            "theta1": params["theta1"],
            "theta2": params["theta2"],
            "mu3": params["mu3"],
        }
    if model in {"SV_log", "SV_exp", "SV_tanh"}:
        return {
            "mu": params["mu"],
            "mu2": params["mu2"],
            "K_1ani1": params["K_1ani1"],
            "beta_1": params["beta_1"],
            "K_1ani2": params["K_1ani2"],
            "gamma_1": params["gamma_1"],
            "K_coup": params["K_kop12"],
            "zeta": params["zeta"],
            "C1": params["C1"],
            "C2": params["C2"],
            "theta1": params["theta1"],
            "theta2": params["theta2"],
            "mu3": params["mu3"],
        }
    raise ValueError(f"Unsupported model: {model}")


def normalize_components(components: Sequence[str]) -> list[str]:
    """Validate component names."""
    normalized = []
    for component in components:
        key = component.lower()
        if key not in COMPONENTS:
            allowed = ", ".join(COMPONENTS)
            raise ValueError(f"Unknown component '{component}'. Use one of: {allowed}.")
        normalized.append(key)
    return normalized


def experimental_path(
    knit: str,
    ratio: str,
    data_dir: pl.Path | None,
    data_collection: str,
    yarn_type: str,
    raw_sample_folder: str | None,
) -> pl.Path:
    """Locate the selected experimental data file."""
    candidates = experimental_path_candidates(
        knit, ratio, data_dir, data_collection, yarn_type, raw_sample_folder
    )

    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find experiment file. Searched:\n{searched}")


def experimental_path_candidates(knit, ratio, data_dir, data_collection, yarn_type, raw_sample_folder):
    """Locate raw compiled curves; example: Interlock/S06/Compiled_Data."""
    if data_collection != "Raw":
        raise ValueError("Only the Raw collection is shipped")
    if not raw_sample_folder:
        raise ValueError("Set DEFAULT_RAW_SAMPLE_FOLDER, for example BRT_15mm_Tri_Int_S06")
    raw_name = raw_experiment_filename(ratio)
    candidates = [] if data_dir is None else [data_dir / raw_name]
    candidates.append(EXPERIMENTAL_DATA_ROOT / knit / raw_sample_folder / DEFAULT_RAW_DATA_SUBDIR / raw_name)
    return deduplicate_paths(candidates)





def raw_experiment_filename(ratio: str) -> str:
    """Return the ratio filename used in raw sample Compiled_Data folders."""
    return f"{ratio.replace('.', 'p')}.dat"


def deduplicate_paths(paths: Sequence[pl.Path]) -> list[pl.Path]:
    """Remove duplicate search candidates while preserving order."""
    unique_paths = []
    seen = set()
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        unique_paths.append(path)
        seen.add(key)
    return unique_paths


def decompose_history(
    dfgrd_voigt: np.ndarray,
    params: ParameterMap,
    model_id: str = "StapletonViaux",
) -> dict[str, np.ndarray]:
    """Evaluate all stress contribution histories in first PK Voigt form."""
    history: dict[str, list[np.ndarray]] = {}
    # These keys are intentionally the complete decomposition used by the
    # figures: 3 linear terms, 5 nonlinear terms, and their totals.
    keys = (
        "linear",
        "nonlinear",
        "total",
        "mu_1",
        "mu_2",
        "mu_3",
        "K_1ani1",
        "K_2ani1",
        "K_1ani2",
        "K_2ani2",
        "K_coup",
    )
    for key in keys:
        history[key] = []

    for f_voigt in dfgrd_voigt:
        F = mech.voigt_to_matrix(f_voigt)
        pieces = stress_decomposition(F, params, model_id)
        for key in keys:
            history[key].append(mech.matrix_to_voigt(pieces[key]))

    return {key: np.array(value) for key, value in history.items()}


def stress_decomposition(
    F: np.ndarray,
    params: ParameterMap,
    model_id: str = "StapletonViaux",
) -> dict[str, np.ndarray]:
    """Compute first Piola-Kirchhoff stress pieces for one deformation gradient."""
    spk = stress_decomposition_pk2(F, params, model_id)
    # The material equations are easiest to write in PK2 stress. Plotter.py
    # visualizes first PK stress, so convert directly with P = F S.
    return {name: mech.pk2_to_pk1(piece, F) for name, piece in spk.items()}


def stress_decomposition_pk2(
    F: np.ndarray,
    params: ParameterMap,
    model_id: str = "StapletonViaux",
) -> dict[str, np.ndarray]:
    """Compute model stress pieces in second Piola-Kirchhoff form."""
    # theta1/theta2 orient the structural tensors M1 and M2. With the current
    # parameter sets they are course = 0 deg and wales = 90 deg.
    theta1 = np.deg2rad(float(params["theta1"]))
    theta2 = np.deg2rad(float(params["theta2"]))
    n1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
    n2 = np.array([np.cos(theta2), np.sin(theta2), 0.0])
    M1 = np.outer(n1, n1)
    M2 = np.outer(n2, n2)

    C = F.T @ F
    C2 = C @ C

    # Fiber invariants from the draft manuscript. I4/I5 belong to course M1;
    # I6/I7 belong to wales M2.
    I4 = float(n1 @ C @ n1)
    I5 = float(n1 @ C2 @ n1)
    I6 = float(n2 @ C @ n2)
    I7 = float(n2 @ C2 @ n2)

    cn1 = C @ n1
    cn2 = C @ n2
    dI5dC = np.outer(n1, cn1) + np.outer(cn1, n1)
    dI7dC = np.outer(n2, cn2) + np.outer(cn2, n2)

    mu1 = float(params["mu"])
    mu2 = float(params["mu2"])
    mu3 = float(params["mu3"])
    mu_1 = 4.0 * mu1 * (I4 - 1.0) * M1
    mu_2 = 4.0 * mu2 * (I6 - 1.0) * M2
    mu_3 = 2.0 * mu3 * ((I6 - 1.0) * M1 + (I4 - 1.0) * M2)
    linear = mu_1 + mu_2 + mu_3

    # Delay parameters turn nonlinear fiber/coupling terms off until their
    # associated invariant is above its threshold.
    K_1ani1 = active(params["K_1ani1"], I4, params["C1"])
    K_1ani2 = active(params["K_1ani2"], I6, params["C2"])
    K_coup = float(params.get("K_coup", params.get("K_kop12", 0.0)))
    # The combined-max polyconvex variant has one shared activation argument;
    # unlike SV_exp, it must not be disabled by either individual threshold.
    if (
        model_id != "SV_exp_polycon_mo"
        and (I4 <= float(params["C1"]) or I6 <= float(params["C2"]))
    ):
        K_coup = 0.0

    beta_1 = float(params["beta_1"])
    gamma_1 = float(params["gamma_1"])
    zeta = float(params["zeta"])

    # Nonlinear stress terms from S_NL: two course terms, two wales terms,
    # and the course-wales coupling term.
    if model_id == "SV_log":
        lambda4 = np.sqrt(I4)
        lambda6 = np.sqrt(I6)
        response_I4 = np.log1p((lambda4 - 1.0) / beta_1) / lambda4
        response_I6 = np.log1p((lambda6 - 1.0) / gamma_1) / lambda6
        term_K_1ani1 = K_1ani1 * response_I4 * M1
        term_K_1ani2 = K_1ani2 * response_I6 * M2
        term_K_2ani1 = np.zeros_like(M1)
        term_K_2ani2 = np.zeros_like(M2)
    elif model_id in {"SV_exp", "SV_exp_polycon_mo"}:
        response_I4 = -np.expm1(-beta_1 * (I4 - 1.0))
        response_I6 = -np.expm1(-gamma_1 * (I6 - 1.0))
        term_K_1ani1 = 2.0 * K_1ani1 * response_I4 * M1
        term_K_1ani2 = 2.0 * K_1ani2 * response_I6 * M2
        term_K_2ani1 = np.zeros_like(M1)
        term_K_2ani2 = np.zeros_like(M2)
    elif model_id == "SV_exp_2O":
        K_2ani1 = active(params["K_2ani1"], I5, 1.0)
        K_2ani2 = active(params["K_2ani2"], I7, 1.0)
        beta_2 = float(params["beta_2"])
        gamma_2 = float(params["gamma_2"])
        response_I4 = -np.expm1(-beta_1 * (I4 - 1.0))
        response_I5 = (
            -np.expm1(-beta_2 * (I5 - 1.0)) if I5 > 1.0 else 0.0
        )
        response_I6 = -np.expm1(-gamma_1 * (I6 - 1.0))
        response_I7 = (
            -np.expm1(-gamma_2 * (I7 - 1.0)) if I7 > 1.0 else 0.0
        )
        term_K_1ani1 = 2.0 * K_1ani1 * response_I4 * M1
        term_K_2ani1 = 2.0 * K_2ani1 * response_I5 * dI5dC
        term_K_1ani2 = 2.0 * K_1ani2 * response_I6 * M2
        term_K_2ani2 = 2.0 * K_2ani2 * response_I7 * dI7dC
    elif model_id == "SV_tanh":
        response_I4 = np.tanh(beta_1 * (I4 - 1.0))
        response_I6 = np.tanh(gamma_1 * (I6 - 1.0))
        term_K_1ani1 = 2.0 * K_1ani1 * response_I4 * M1
        term_K_1ani2 = 2.0 * K_1ani2 * response_I6 * M2
        term_K_2ani1 = np.zeros_like(M1)
        term_K_2ani2 = np.zeros_like(M2)
    else:
        K_2ani1 = active(params["K_2ani1"], I5, 1.0)
        K_2ani2 = active(params["K_2ani2"], I7, 1.0)
        beta_2 = float(params["beta_2"])
        gamma_2 = float(params["gamma_2"])
        term_K_1ani1 = (
            2.0 * beta_1 * K_1ani1 * (I4 - 1.0) ** (beta_1 - 1.0) * M1
        )
        term_K_1ani2 = (
            2.0 * gamma_1 * K_1ani2 * (I6 - 1.0) ** (gamma_1 - 1.0) * M2
        )
        term_K_2ani1 = 2.0 * beta_2 * K_2ani1 * (I5 - 1.0) ** (beta_2 - 1.0) * dI5dC
        term_K_2ani2 = 2.0 * gamma_2 * K_2ani2 * (I7 - 1.0) ** (gamma_2 - 1.0) * dI7dC
    if model_id == "SV_exp_polycon_mo":
        # Combined-max coupling; example: a=0.6 and b=0.4 with a+b=1.
        a_coup = float(params["a"])
        b_coup = float(params["b"])
        coupling_argument = max(
            a_coup * (I4 - float(params["C1"]))
            + b_coup * (I6 - float(params["C2"])),
            0.0,
        )
        term_K_coup = np.zeros_like(M1)
        if coupling_argument > 0.0:
            term_K_coup = (
                2.0
                * K_coup
                * zeta
                * coupling_argument ** (zeta - 1.0)
                * (a_coup * M1 + b_coup * M2)
            )
    else:
        term_K_coup = 2.0 * K_coup * zeta * (
            (I6 - 1.0) ** zeta * (I4 - 1.0) ** (zeta - 1.0) * M1
            + (I4 - 1.0) ** zeta * (I6 - 1.0) ** (zeta - 1.0) * M2
        )
    nonlinear = (
        term_K_1ani1
        + term_K_2ani1
        + term_K_1ani2
        + term_K_2ani2
        + term_K_coup
    )

    return {
        "linear": linear,
        "nonlinear": nonlinear,
        "total": linear + nonlinear,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "mu_3": mu_3,
        "K_1ani1": term_K_1ani1,
        "K_2ani1": term_K_2ani1,
        "K_1ani2": term_K_1ani2,
        "K_2ani2": term_K_2ani2,
        "K_coup": term_K_coup,
    }


def active(value: float | int, invariant: float, threshold: float | int) -> float:
    """Return a parameter value only when its invariant is above threshold."""
    if invariant <= float(threshold):
        return 0.0
    return float(value)


def call_umat_history(dfgrd_voigt: np.ndarray, props: np.ndarray) -> np.ndarray:
    """Call the compiled Fortran UMAT for each deformation gradient."""
    stresses = []
    for F in dfgrd_voigt:
        stress, _statev = umat.call_umat_Abqs_sta(F, props)
        stresses.append(stress)
    return np.array(stresses)


def cauchy_history_to_pk1(dfgrd_voigt: np.ndarray, cauchy_stress: np.ndarray) -> np.ndarray:
    """Convert UMAT Cauchy stress history to first PK stress history."""
    p_stress = []
    for F_voigt, sigma_voigt in zip(dfgrd_voigt, cauchy_stress):
        F = mech.voigt_to_matrix(F_voigt)
        sigma = mech.voigt_to_matrix(sigma_voigt)
        p_stress.append(mech.matrix_to_voigt(mech.cauchystress_to_pk1(sigma, F)))
    return np.array(p_stress)


def invariant_diagnostics_history(
    dfgrd_voigt: np.ndarray,
    params: ParameterMap,
    model_id: str = "StapletonViaux",
) -> dict[str, np.ndarray]:
    """Evaluate scalar invariant diagnostics along the experiment history."""
    keys = (
        "I4_delta",
        "I5_delta",
        "I6_delta",
        "I7_delta",
        "I4_power",
        "I5_power",
        "I6_power",
        "I7_power",
        "coupling_M1_power",
        "coupling_M2_power",
        "course_I4_prefactor",
        "course_I5_prefactor",
        "wales_I6_prefactor",
        "wales_I7_prefactor",
        "coupling_M1_prefactor",
        "coupling_M2_prefactor",
    )
    history: dict[str, list[float]] = {key: [] for key in keys}

    for f_voigt in dfgrd_voigt:
        F = mech.voigt_to_matrix(f_voigt)
        values = invariant_diagnostics(F, params, model_id)
        for key in keys:
            history[key].append(values[key])

    return {key: np.array(value) for key, value in history.items()}


def invariant_diagnostics(
    F: np.ndarray,
    params: ParameterMap,
    model_id: str = "StapletonViaux",
) -> dict[str, float]:
    """Compute scalar invariant and nonlinear prefactor diagnostics."""
    theta1 = np.deg2rad(float(params["theta1"]))
    theta2 = np.deg2rad(float(params["theta2"]))
    n1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
    n2 = np.array([np.cos(theta2), np.sin(theta2), 0.0])
    C = F.T @ F
    C2 = C @ C

    I4 = float(n1 @ C @ n1)
    I5 = float(n1 @ C2 @ n1)
    I6 = float(n2 @ C @ n2)
    I7 = float(n2 @ C2 @ n2)
    d4 = I4 - 1.0
    d5 = I5 - 1.0
    d6 = I6 - 1.0
    d7 = I7 - 1.0

    beta_1 = float(params["beta_1"])
    gamma_1 = float(params["gamma_1"])
    zeta = float(params["zeta"])

    K_1ani1 = active(params["K_1ani1"], I4, params["C1"])
    K_1ani2 = active(params["K_1ani2"], I6, params["C2"])
    K_coup = float(params.get("K_coup", params.get("K_kop12", 0.0)))
    if I4 <= float(params["C1"]) or I6 <= float(params["C2"]):
        K_coup = 0.0

    if model_id == "SV_log":
        lambda4 = np.sqrt(I4)
        lambda6 = np.sqrt(I6)
        I4_power = float(np.log1p((lambda4 - 1.0) / beta_1) / lambda4)
        I6_power = float(np.log1p((lambda6 - 1.0) / gamma_1) / lambda6)
        course_I4_prefactor = 2.0 * K_1ani1 * I4_power
        wales_I6_prefactor = 2.0 * K_1ani2 * I6_power
        I5_power = 0.0
        I7_power = 0.0
        course_I5_prefactor = 0.0
        wales_I7_prefactor = 0.0
    elif model_id == "SV_exp":
        I4_power = float(-np.expm1(-beta_1 * d4))
        I6_power = float(-np.expm1(-gamma_1 * d6))
        course_I4_prefactor = 2.0 * K_1ani1 * I4_power
        wales_I6_prefactor = 2.0 * K_1ani2 * I6_power
        I5_power = 0.0
        I7_power = 0.0
        course_I5_prefactor = 0.0
        wales_I7_prefactor = 0.0
    elif model_id == "SV_exp_2O":
        beta_2 = float(params["beta_2"])
        gamma_2 = float(params["gamma_2"])
        K_2ani1 = active(params["K_2ani1"], I5, 1.0)
        K_2ani2 = active(params["K_2ani2"], I7, 1.0)
        I4_power = float(-np.expm1(-beta_1 * d4))
        I5_power = float(-np.expm1(-beta_2 * d5)) if I5 > 1.0 else 0.0
        I6_power = float(-np.expm1(-gamma_1 * d6))
        I7_power = float(-np.expm1(-gamma_2 * d7)) if I7 > 1.0 else 0.0
        course_I4_prefactor = 2.0 * K_1ani1 * I4_power
        course_I5_prefactor = 2.0 * K_2ani1 * I5_power
        wales_I6_prefactor = 2.0 * K_1ani2 * I6_power
        wales_I7_prefactor = 2.0 * K_2ani2 * I7_power
    elif model_id == "SV_tanh":
        I4_power = float(np.tanh(beta_1 * d4))
        I6_power = float(np.tanh(gamma_1 * d6))
        course_I4_prefactor = 2.0 * K_1ani1 * I4_power
        wales_I6_prefactor = 2.0 * K_1ani2 * I6_power
        I5_power = 0.0
        I7_power = 0.0
        course_I5_prefactor = 0.0
        wales_I7_prefactor = 0.0
    else:
        beta_2 = float(params["beta_2"])
        gamma_2 = float(params["gamma_2"])
        K_2ani1 = active(params["K_2ani1"], I5, 1.0)
        K_2ani2 = active(params["K_2ani2"], I7, 1.0)
        I4_power = positive_power(d4, beta_1 - 1.0)
        I6_power = positive_power(d6, gamma_1 - 1.0)
        course_I4_prefactor = 2.0 * beta_1 * K_1ani1 * I4_power
        wales_I6_prefactor = 2.0 * gamma_1 * K_1ani2 * I6_power
        I5_power = positive_power(d5, beta_2 - 1.0)
        I7_power = positive_power(d7, gamma_2 - 1.0)
        course_I5_prefactor = 2.0 * beta_2 * K_2ani1 * I5_power
        wales_I7_prefactor = 2.0 * gamma_2 * K_2ani2 * I7_power
    coupling_M1_power = positive_power(d6, zeta) * positive_power(d4, zeta - 1.0)
    coupling_M2_power = positive_power(d4, zeta) * positive_power(d6, zeta - 1.0)

    return {
        "I4_delta": d4,
        "I5_delta": d5,
        "I6_delta": d6,
        "I7_delta": d7,
        "I4_power": I4_power,
        "I5_power": I5_power,
        "I6_power": I6_power,
        "I7_power": I7_power,
        "coupling_M1_power": coupling_M1_power,
        "coupling_M2_power": coupling_M2_power,
        "course_I4_prefactor": course_I4_prefactor,
        "course_I5_prefactor": course_I5_prefactor,
        "wales_I6_prefactor": wales_I6_prefactor,
        "wales_I7_prefactor": wales_I7_prefactor,
        "coupling_M1_prefactor": 2.0
        * K_coup
        * zeta
        * coupling_M1_power,
        "coupling_M2_prefactor": 2.0
        * K_coup
        * zeta
        * coupling_M2_power,
    }


def positive_power(base: float, exponent: float) -> float:
    """Power helper for thresholded invariant terms."""
    if base <= 0.0:
        return 0.0
    return float(base**exponent)


def plot_invariant_diagnostics(
    strains: np.ndarray,
    dfgrd_voigt: np.ndarray,
    params: ParameterMap,
    args: argparse.Namespace,
    save_dir: pl.Path | None,
    x_component: str,
    diagnostics_fn: DiagnosticsFunction = invariant_diagnostics_history,
) -> None:
    """Plot invariant growth, powered factors, and nonlinear prefactors."""
    diagnostics = diagnostics_fn(dfgrd_voigt, params, args.model)
    x_component = normalize_components((x_component,))[0]
    x = component_x(strains, x_component)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.8), sharex=True, tight_layout=True)
    axes_list = cast(list[Axes], list(np.atleast_1d(axes).ravel()))

    invariant_lines = (
        ("I4_delta", r"$I_4 - 1$", INVARIANT_COLORS["I4"]),
        ("I5_delta", r"$I_5 - 1$", INVARIANT_COLORS["I5"]),
        ("I6_delta", r"$I_6 - 1$", INVARIANT_COLORS["I6"]),
        ("I7_delta", r"$I_7 - 1$", INVARIANT_COLORS["I7"]),
    )
    if args.model == "SV_log":
        i4_factor_label = r"$\log(1+(\sqrt{I_4}-1)/\beta_1)/\sqrt{I_4}$"
        i6_factor_label = r"$\log(1+(\sqrt{I_6}-1)/\gamma_1)/\sqrt{I_6}$"
    elif args.model == "SV_exp":
        i4_factor_label = r"$1-\exp[-\beta_1(I_4-1)]$"
        i6_factor_label = r"$1-\exp[-\gamma_1(I_6-1)]$"
    elif args.model == "SV_exp_2O":
        i4_factor_label = r"$1-\exp[-\beta_1(I_4-1)]$"
        i6_factor_label = r"$1-\exp[-\gamma_1(I_6-1)]$"
    elif args.model == "SV_tanh":
        i4_factor_label = r"$\tanh[\beta_1(I_4-1)]$"
        i6_factor_label = r"$\tanh[\gamma_1(I_6-1)]$"
    else:
        i4_factor_label = r"$(I_4 - 1)^{\beta_1 - 1}$"
        i6_factor_label = r"$(I_6 - 1)^{\gamma_1 - 1}$"

    i5_factor_label = r"$(I_5 - 1)^{\beta_2 - 1}$"
    i7_factor_label = r"$(I_7 - 1)^{\gamma_2 - 1}$"
    if args.model == "SV_exp_2O":
        i5_factor_label = r"$1-\exp[-\beta_2(I_5-1)]$"
        i7_factor_label = r"$1-\exp[-\gamma_2(I_7-1)]$"

    power_lines = (
        ("I4_power", i4_factor_label, INVARIANT_COLORS["I4"]),
        ("I5_power", i5_factor_label, INVARIANT_COLORS["I5"]),
        ("I6_power", i6_factor_label, INVARIANT_COLORS["I6"]),
        ("I7_power", i7_factor_label, INVARIANT_COLORS["I7"]),
        (
            "coupling_M1_power",
            r"$(I_6 - 1)^\zeta (I_4 - 1)^{\zeta - 1}$",
            INVARIANT_COLORS["coupling_M1"],
        ),
        (
            "coupling_M2_power",
            r"$(I_4 - 1)^\zeta (I_6 - 1)^{\zeta - 1}$",
            INVARIANT_COLORS["coupling_M2"],
        ),
    )
    prefactor_lines = (
        ("course_I4_prefactor", r"Course $I_4$", INVARIANT_COLORS["I4"]),
        ("course_I5_prefactor", r"Course $I_5$", INVARIANT_COLORS["I5"]),
        ("wales_I6_prefactor", r"Wales $I_6$", INVARIANT_COLORS["I6"]),
        ("wales_I7_prefactor", r"Wales $I_7$", INVARIANT_COLORS["I7"]),
        ("coupling_M1_prefactor", "Coupling M1", INVARIANT_COLORS["coupling_M1"]),
        ("coupling_M2_prefactor", "Coupling M2", INVARIANT_COLORS["coupling_M2"]),
    )

    plot_diagnostic_lines(axes_list[0], x, diagnostics, invariant_lines)
    axes_list[0].set_ylabel("Invariant change")
    axes_list[0].set_title("Invariant growth")

    plot_diagnostic_lines(axes_list[1], x, diagnostics, power_lines, log_scale=True)
    axes_list[1].set_ylabel("Powered factor")
    axes_list[1].set_title("Power suppression")

    plot_diagnostic_lines(axes_list[2], x, diagnostics, prefactor_lines, log_scale=True)
    axes_list[2].set_ylabel("Scalar prefactor")
    axes_list[2].set_title("Weighted nonlinear prefactors")

    for ax in axes_list:
        uml_style.style_axes(ax)
        ax.grid(False)
        ax.legend(fontsize="x-small", ncols=2, loc="best")

    axes_list[-1].set_xlabel(f"{x_component} stretch")
    fig.suptitle(
        f"{args.model} {args.knit}: invariant diagnostics vs {x_component} stretch",
        fontweight="bold",
    )
    save_figure(fig, save_dir, args, f"invariant_diagnostics_{x_component}")


def plot_diagnostic_lines(
    ax: Axes,
    x: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    lines: Sequence[tuple[str, str, str]],
    log_scale: bool = False,
) -> None:
    """Plot diagnostic series, hiding nonpositive values on log axes."""
    for key, label, color in lines:
        y = diagnostics[key]
        if log_scale:
            y = np.where(y > 0.0, y, np.nan)
        ax.plot(x, y, color=color, linewidth=1.6, label=label)
    if log_scale:
        ax.set_yscale("log")


def plot_linear_vs_nonlinear(
    strains: np.ndarray,
    decomp: Mapping[str, np.ndarray],
    components: Sequence[str],
    args: argparse.Namespace,
    save_dir: pl.Path | None,
) -> None:
    """Plot total linear and nonlinear stress groups."""
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    colors = uml_style.axis_colors(max(len(components), 2))
    styles = {"linear": "-", "nonlinear": "--"}

    for cidx, component in enumerate(components):
        x = component_x(strains, component)
        _, voigt_idx = COMPONENTS[component]
        for group in ("linear", "nonlinear"):
            ax.plot(
                x,
                decomp[group][:, voigt_idx],
                color=colors[cidx],
                linestyle=styles[group],
                label=f"{component} {group}",
            )

    ax.set_xlabel("Stretch")
    ax.set_ylabel(P_STRESS_LABEL)
    ax.set_title(f"{args.model} {args.knit}: linear vs nonlinear")
    ax.legend(fontsize="small")
    uml_style.style_axes(ax)
    ax.grid(False)
    save_figure(fig, save_dir, args, "linear_vs_nonlinear")


def plot_linear_terms(
    strains: np.ndarray,
    decomp: Mapping[str, np.ndarray],
    components: Sequence[str],
    args: argparse.Namespace,
    save_dir: pl.Path | None,
) -> None:
    """Plot the three linear stress terms as stacked areas."""
    if len(LINEAR_TERMS) != 3:
        raise ValueError("The linear stress split must contain exactly three terms.")

    # Stacked areas make the contribution of each term visible without relying
    # on line style differences.
    fig, axes = make_component_axes(len(components), figsize=(8, 3.2 * len(components)))
    colors = [TERM_COLORS[term] for term, _label in LINEAR_TERMS]

    for ax, component in zip(axes, components):
        x = component_x(strains, component)
        _, voigt_idx = COMPONENTS[component]
        values = [decomp[term][:, voigt_idx] for term, _label in LINEAR_TERMS]
        labels = [label for _term, label in LINEAR_TERMS]
        plot_signed_stack(ax, x, values, colors, labels)
        ax.plot(x, np.sum(np.vstack(values), axis=0), color="#222222", linewidth=1.2, label="Linear total")
        ax.set_ylabel(f"{component} {P_STRESS_LABEL}")
        ax.set_title(f"{component} component")
        uml_style.style_axes(ax)
        ax.grid(False)

    axes[-1].set_xlabel("Stretch")
    axes[0].legend(fontsize="small", loc="best")
    fig.suptitle(f"{args.model} {args.knit}: linear term contributions", fontweight="bold")
    save_figure(fig, save_dir, args, "linear_terms")


def plot_nonlinear_terms(
    strains: np.ndarray,
    decomp: Mapping[str, np.ndarray],
    components: Sequence[str],
    args: argparse.Namespace,
    save_dir: pl.Path | None,
) -> None:
    """Plot the active nonlinear stress terms as stacked areas."""
    terms = NONLINEAR_TERMS
    if args.model in {"SV_log", "SV_exp", "SV_tanh"}:
        terms = tuple(
            item for item in terms if item[0] not in {"K_2ani1", "K_2ani2"}
        )

    # The nonlinear split is visually crowded as line plots, so each stress
    # component gets its own stacked area panel.
    fig, axes = make_component_axes(len(components), figsize=(9, 3.4 * len(components)))
    colors = [TERM_COLORS[term] for term, _label in terms]

    for ax, component in zip(axes, components):
        x = component_x(strains, component)
        _, voigt_idx = COMPONENTS[component]
        values = [decomp[term][:, voigt_idx] for term, _label in terms]
        labels = [label for _term, label in terms]
        plot_signed_stack(ax, x, values, colors, labels)
        ax.plot(x, np.sum(np.vstack(values), axis=0), color="#222222", linewidth=1.2, label="Nonlinear total")
        ax.set_ylabel(f"{component} {P_STRESS_LABEL}")
        ax.set_title(f"{component} component")
        uml_style.style_axes(ax)
        ax.grid(False)

    axes[-1].set_xlabel("Stretch")
    axes[0].legend(fontsize="small", loc="best", ncols=2)
    fig.suptitle(f"{args.model} {args.knit}: nonlinear term contributions", fontweight="bold")
    save_figure(fig, save_dir, args, "nonlinear_terms")


def plot_linear_parameters(
    params: ParameterMap,
    args: argparse.Namespace,
    save_dir: pl.Path | None,
) -> None:
    """Plot a standalone bar chart of the linear parameters."""
    fig, ax = plt.subplots(figsize=LINEAR_PARAM_FIGSIZE, tight_layout=True)
    plot_parameter_bars(ax, params, LINEAR_PARAMETER_BARS, LINEAR_PARAM_GROUP_SPACING)
    # ax.set_title(
    #     f"{args.model} {args.knit}: linear parameters",
    #     fontweight="bold",
    #     fontsize=PARAM_TITLE_FONTSIZE,
    # )
    save_figure(fig, save_dir, args, "linear_parameters")


def plot_nonlinear_parameters(
    params: ParameterMap,
    args: argparse.Namespace,
    save_dir: pl.Path | None,
) -> None:
    """Plot standalone grouped bars for the nonlinear parameter pairs."""
    bars = NONLINEAR_PARAMETER_BARS
    if args.model in {"SV_log", "SV_exp", "SV_tanh"}:
        bars = tuple(
            item for item in bars if item[0] not in {"K_2ani1", "K_2ani2"}
        )
    fig, ax = plt.subplots(figsize=NONLINEAR_PARAM_FIGSIZE, tight_layout=True)
    plot_parameter_bars(ax, params, bars, NONLINEAR_PARAM_GROUP_SPACING)
    # ax.set_title(
    #     f"{args.model} {args.knit}: nonlinear parameters",
    #     fontweight="bold",
    #     fontsize=PARAM_TITLE_FONTSIZE,
    # )
    save_figure(fig, save_dir, args, "nonlinear_parameters")


def component_x(strains: np.ndarray, component: str) -> np.ndarray:
    """Use the matching normal stretch as x axis, or point number for shear."""
    _, voigt_idx = COMPONENTS[component]
    if voigt_idx in (0, 1, 2):
        return strains[:, voigt_idx] + 1.0
    return np.arange(strains.shape[0])


def make_component_axes(n_components: int, figsize: tuple[float, float]) -> tuple[Figure, list[Axes]]:
    """Create one vertical subplot per stress component."""
    fig, axes_raw = plt.subplots(n_components, 1, figsize=figsize, tight_layout=True)
    axes = cast(list[Axes], list(np.atleast_1d(axes_raw).ravel()))
    return fig, axes


def plot_parameter_bars(
    ax: Axes,
    params: ParameterMap,
    bars: Sequence[tuple[str, str, Sequence[str]]],
    group_spacing: float,
) -> None:
    """Draw normalized parameter bars with actual values annotated."""
    raw_positions: list[float] = []
    raw_values: list[float] = []
    raw_labels: list[str] = []
    raw_colors: list[str] = []
    tick_labels: list[str] = []
    heights: list[float] = []

    bar_width = PARAM_BAR_WIDTH
    for group_idx, (term, label, param_names) in enumerate(bars):
        center = -group_idx * group_spacing
        if len(param_names) == 1:
            offsets = np.array([0.0])
            labels = [label]
        else:
            offsets = np.linspace(-bar_width, bar_width, len(param_names))
            labels = parameter_labels(param_names)
        for offset, param_name, tick_label in zip(offsets, param_names, labels):
            value = parameter_value(params, param_name)
            raw_positions.append(center + float(offset))
            raw_values.append(float(value))
            raw_labels.append(format_parameter_value(value))
            raw_colors.append(TERM_COLORS[term])
            tick_labels.append(tick_label)
            heights.append(bar_width)

    max_value = max(abs(value) for value in raw_values) or 1.0
    display_values = [value / max_value for value in raw_values]
    ax.barh(raw_positions, display_values, color=raw_colors, height=heights, alpha=0.82)

    for y_pos, display_value, label in zip(raw_positions, display_values, raw_labels):
        x_pos = display_value + 0.035 if display_value >= 0 else display_value - 0.035
        ha = "left" if display_value >= 0 else "right"
        ax.text(x_pos, y_pos, label, ha=ha, va="center", fontsize=PARAM_VALUE_FONTSIZE)

    min_display = min(display_values)
    max_display = max(display_values)
    left_lim = min(-0.12, min_display - 0.22) if min_display < 0 else -0.06
    right_lim = max(1.16, max_display + 0.22)
    ax.set_xlim(left_lim, right_lim)
    ax.set_ylim(min(raw_positions) - 0.28, max(raw_positions) + 0.28)
    ax.set_yticks(raw_positions)
    ax.set_yticklabels(tick_labels, fontsize=PARAM_LABEL_FONTSIZE)
    ax.set_xticks([])
    ax.grid(False)
    uml_style.style_axes(ax)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.tick_params(axis="y", length=0, pad=1)


def parameter_value(params: ParameterMap, name: str) -> float | int:
    """Return a plotting parameter, including established family aliases."""
    if name in params:
        return params[name]
    if name == "K_coup" and "K_kop12" in params:
        return params["K_kop12"]
    raise KeyError(name)


def format_parameter_value(value: float | int) -> str:
    """Format parameter values compactly for the small bar strip."""
    val = float(value)
    if val == 0.0:
        return "0"
    if abs(val) >= 100 or abs(val) < 0.01:
        return f"{val:.2e}"
    return f"{val:.3g}"


def parameter_labels(param_names: Sequence[str]) -> list[str]:
    """Return pretty per-bar labels for material parameter names."""
    labels = {
        "mu": r"$\mu_1$",
        "mu2": r"$\mu_2$",
        "mu3": r"$\mu_3$",
        "K_1ani1": r"$K^{ani}_{1,1}$",
        "beta_1": r"$\beta_1$",
        "K_2ani1": r"$K^{ani}_{2,1}$",
        "beta_2": r"$\beta_2$",
        "K_1ani2": r"$K^{ani}_{1,2}$",
        "gamma_1": r"$\gamma_1$",
        "K_2ani2": r"$K^{ani}_{2,2}$",
        "gamma_2": r"$\gamma_2$",
        "K_coup": r"$K_{coup}$",
        "K_kop12": r"$K_{coup}$",
        "zeta": r"$\zeta$",
    }
    return [labels.get(param_name, param_name) for param_name in param_names]


def plot_signed_stack(
    ax: Axes,
    x: np.ndarray,
    values: Sequence[np.ndarray],
    colors: Sequence[str],
    labels: Sequence[str],
) -> None:
    """Plot stacked areas while preserving positive and negative contributions."""
    y = np.vstack(values)
    # stackplot assumes one-sided areas. Split positive and negative portions so
    # a compressive/negative term is still visible instead of being hidden.
    positive = np.clip(y, 0.0, None)
    negative = np.clip(y, None, 0.0)

    if np.any(positive):
        ax.stackplot(x, positive, colors=colors, labels=labels, alpha=0.82)
    if np.any(negative):
        ax.stackplot(x, negative, colors=colors, alpha=0.55)
    ax.axhline(0.0, color="#444444", linewidth=0.8)


def save_figure(
    fig: Figure,
    save_dir: pl.Path | None,
    args: argparse.Namespace,
    label: str,
) -> None:
    """Save a figure when requested."""
    if save_dir is None:
        return
    # Relative save directories are interpreted relative to the current shell
    # directory, matching normal Python/pathlib behavior.
    ratio = args.ratio.replace(".", "p").replace(" ", "_")
    data_label = filename_token(f"{args.data_collection}_{args.yarn_type}")
    if args.data_collection == "Raw" and args.raw_sample_folder:
        data_label = filename_token(f"{data_label}_{args.raw_sample_folder}")
    fit_label = f"_fit_{args.fit_code:04d}" if args.fit_code is not None else ""
    path = save_dir / (
        f"{args.model}_{args.knit}_{data_label}_{ratio}{fit_label}_{label}.svg"
    )
    fig.savefig(path, transparent=True)
    print(f"Saved {path}")


def filename_token(text: str) -> str:
    """Convert a short label to a filesystem-friendly filename token."""
    return text.replace(".", "p").replace(" ", "_").replace("\\", "_").replace("/", "_")


if __name__ == "__main__":
    main()
