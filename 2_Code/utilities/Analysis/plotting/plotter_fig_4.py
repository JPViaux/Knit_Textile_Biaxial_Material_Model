
# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (CODE_ROOT as RELEASE_CODE_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT,
    FIGURE_ROOT as RELEASE_FIGURE_ROOT)
import argparse
import ast
import pathlib as pl
import sys
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
curr_file = os.fspath(
    next(parent for parent in pl.Path(__file__).resolve().parents if parent.name == "Analysis")
)
parent_dir = os.path.dirname(curr_file)
sys.path.insert(0, parent_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
from Python_Wrapper import umat
from Python_Wrapper.model_registry import load_model_spec
from Analysis.core import experimental_datasets as exp_datasets
from Analysis.core.fit_results import DEFAULT_RESULTS_CSV, FitResult, load_fit
import uml_plot_style as uml_style

uml_style.apply()

PathMap = Mapping[str, pl.Path]
ParameterMap = Mapping[str, float | int]
AxisLimits = tuple[float, float] | None

RATIO_LABELS = {
    '0 to 1': '0:1',
    '0.25 to 1': '0.25:1',
    '0.5 to 1': '0.5:1',
    '0.75 to 1': '0.75:1',
    '1 to 0': '1:0',
    '1 to 0.25': '1:0.25',
    '1 to 0.5': '1:0.5',
    '1 to 0.75': '1:0.75',
    '1 to 1': '1:1',
}
ALL_PLOT_RATIOS = (
    ('0 to 1', '0.25 to 1', '0.5 to 1', '0.75 to 1'),
    ('1 to 0', '1 to 0.25', '1 to 0.5', '1 to 0.75'),
)
ALL_RATIO_ORDER = tuple(
    ratio for ratio_row in ALL_PLOT_RATIOS for ratio in ratio_row
) + ('1 to 1',)

# Ratios plotted when pressing Play in VS Code. Examples: ('1 to 1',) or
# ('0.5 to 1', '1 to 0.5', '1 to 1'). The complete tuple keeps the original
# 2x4 summary plus the independent 1:1 figure.
SELECTED_RATIOS: tuple[str, ...] = (
    "0.5 to 1",
    "1 to 0.5",
    "1 to 1",
    # "0 to 1",
    # "0.25 to 1",
    # "0.75 to 1",
    # "1 to 0",
    # "1 to 0.25",
    # "1 to 0.75",
)

# Legacy single-knit fit retained for the copied helper functions below.
DEFAULT_FIT_CODE: int | None = 2166

# Saved fits plotted from top to bottom when pressing Play in VS Code. Examples:
# Interlock 2166, Rib 2167, Purl 2165, and Single Jersey 2164 are the C063 fits.
KNIT_ROW_ORDER: tuple[str, ...] = (
    "Interlock",
    "Rib",
    "Purl",
    "SingleJersey",
)
FIT_CODES_BY_KNIT: dict[str, int] = {
    "Interlock": 2166,
    "Rib": 2167,
    "Purl": 2165,
    "SingleJersey": 2164,
}
# SVG filename used by the one-click run; for example, "figure_4_four_knits.svg".
OUTPUT_FILE_NAME: str = "figure_4_four_knits.svg"
# Folder used by the one-click run. For example, this manuscript directory
# keeps the generated Figure 4 beside the accepted manually assembled version.
OUTPUT_DIRECTORY: pl.Path = RELEASE_GENERATED_ROOT / 'figures/calibration'

# Experimental dataset used when pressing Play in VS Code. Examples include
# "tripoly_interlock_s06_raw" and "tripoly_purl_s02_raw". Keep None to
# reproduce the dataset stored with the selected fit.
EXPERIMENTAL_DATASET_KEYS_BY_KNIT: dict[str, str | None] = {
    "Interlock": None,
    "Rib": None,
    "Purl": None,
    "SingleJersey": None,
}

# Interlock display axes are intentionally reversed from the metadata mapping.
# Keep True to plot data axis 1 as Course and data axis 2 as Wales; use False
# to display the metadata directions instead (for example, Wales then Course).
INVERT_INTERLOCK_PLOT_AXES: bool = True

# Axis limits used for each knit row when pressing Play in VS Code. Each value
# can be (min, max), such as (0.98, 1.25), or None for automatic limits.
X_LIMITS_BY_KNIT: dict[str, AxisLimits] = {
    "Interlock": (0.98, 1.25),
    "Rib": (0.98, 1.25),
    "Purl": (0.98, 1.25),
    "SingleJersey": (0.98, 1.25),
}
# Y values use the units selected by Y_AXIS_QUANTITY. Examples: (0, 80) for
# P stress in kPa or (0, 0.1) for Force/reference width in N/mm.
Y_LIMITS_BY_KNIT: dict[str, AxisLimits] = {
    "Interlock": (0, 0.125),
    "Rib": (0, 0.03),
    "Purl": (0, 0.15),
    "SingleJersey": (0, 0.16),
}

# Legacy defaults retained for the copied single-knit plotting helpers.
DEFAULT_X_LIMITS: AxisLimits = (0.98, 1.25)
DEFAULT_Y_LIMITS: AxisLimits = (0, 0.1)

# Y-axis quantity shown when pressing Play in VS Code. Options are "p_stress"
# for the current first Piola-Kirchhoff stress or "force_per_width" for the
# membrane force divided by reference width.
Y_AXIS_QUANTITY: str = "force_per_width"
# Reference textile thickness in mm; for example, 1.59 mm converts MPa to N/mm.
REFERENCE_THICKNESS_MM: float = 1.59
# Unit conversion applied only when displaying P stress; for example, 1 MPa = 1000 kPa.
KPA_PER_MPA: float = 1000.0
Y_AXIS_OPTIONS: tuple[str, str] = ("p_stress", "force_per_width")

# Number of interpolated strain-path points used for the smooth model curve.
MODEL_CURVE_POINTS: int = 400

# Raw model curves follow the measured path; example: all experimental rows.

# Plot mode options: "Individual" or "All".
PLOT_MODE: str = "All"

# Legacy nominal subplot width used only by the individual/all-ratios helper
# functions below. For example, 4 makes an individual plot 4 inches wide.
# The one-click 4x3 Figure 4 instead uses the measured FIGURE_4_* settings.
PLOT_WIDTH_INCHES: float = 4
# Maximum number of ratio subplots per row. Examples: 4 preserves the current
# layout, while 6 places up to six selected ratios across one row.
MAX_SUBPLOT_COLUMNS: int = 6
# Legacy combined-grid width cap. For example, 12 permits three nominally
# 4-inch columns in the copied all-ratios helper; it does not size Figure 4.
MAX_FIGURE_WIDTH_INCHES: float = 12
PLOT_ASPECT_RATIO: float = 6 / 5.5
SHOW_PLOT_TITLE: bool = False
# True shows Exp./Model Course/Wales labels in the first plot; False shows only
# its two R-squared labels, matching the remaining plots in the figure.
SHOW_CURVE_LEGEND: bool = False
# Choose which axis titles are shown as (x, y). Examples: (True, False) shows
# only "Stretch", while (False, True) shows only the selected y-axis title.
SHOW_AXIS_TITLE: tuple[bool, bool] = (True, True)
# True keeps y-axis numeric tick labels on every subplot. Set False to retain
# them only on the first subplot of each row in a combined-ratio figure.
SHOW_Y_TICK_LABELS_ON_EACH_SUBPLOT: bool = False
PLOT_TITLE_FONT_SIZE: float = 8
PLOT_TITLE_PAD: float = 2
PLOT_LABEL_FONT_SIZE: float = 10
PLOT_TICK_FONT_SIZE: float = 8
PLOT_LEGEND_FONT_SIZE: float = 10
# Thickness of model-line samples inside legends. Examples: 1.2 matches the
# model curves, while 2.0 makes the legend lines easier to see.
LEGEND_LINE_WIDTH: float = 1.2
# Horizontal length of line samples inside legends, in legend-font-size units.
# Examples: 2.0 is Matplotlib's standard length, while 3.0 makes lines longer.
LEGEND_LINE_LENGTH: float = 1.0
# Vertical gap between legend entries, in legend-font-size units. For example,
# 0.15 keeps two entries compact, while 0.5 restores Matplotlib's default gap.
LEGEND_LABEL_SPACING: float = 0.5
# Vertical anchor above an axes, in axes-height units. For example, 1.15 places
# the legend above the plot, while 1.0 lets the legend extend into the plot.
LEGEND_VERTICAL_ANCHOR: float = 1.0
SCATTER_POINT_SIZE: float = 6
POINT_TRANS = 0.2
AXIS_LINE_WIDTH: float = 0.8
GRID_LINE_WIDTH: float = 0.5
MODEL_LINE_WIDTH: float = 1.2

# One-click Figure 4 geometry measured from the accepted row-by-row reference
# SVG (SV_Exp_3_fit.svg). Width and height are physical inches; for example,
# 6.75 by 8.0 reproduces the reference manuscript footprint.
FIGURE_4_WIDTH_INCHES: float = 6.75
FIGURE_4_HEIGHT_INCHES: float = 8.0
# Margins are fractions of the full Figure 4 canvas. For example, left=0.094
# reserves 9.4% for y ticks/labels and right=0.975 uses space through 97.5%.
FIGURE_4_LEFT_MARGIN: float = 0.094
FIGURE_4_RIGHT_MARGIN: float = 0.975
FIGURE_4_BOTTOM_MARGIN: float = 0.069
FIGURE_4_TOP_MARGIN: float = 0.976
# Spacing is a fraction of the average axes size. These values reproduce the
# reference: for example, column spacing 0.20 gives a gap 20% of axes width.
FIGURE_4_ROW_SPACING: float = 0.175
FIGURE_4_COLUMN_SPACING: float = 0.20

# Legacy spacing retained for the copied single-knit all-ratios helper below.
# Examples: 0.25 separates rows and 0.01 leaves a very small column gap.
ALL_PLOT_ROW_SPACING: float = 0.25
ALL_PLOT_COLUMN_SPACING: float = 0.01


@dataclass
class RatioPlotData:
    ratio: str
    path: pl.Path
    axis_directions: tuple[str, str]
    exp_strains: np.ndarray
    exp_stress: np.ndarray
    model_strains: np.ndarray
    sim_stress_curve: np.ndarray
    r_squared: list[float]


@dataclass(frozen=True)
class FitPlotContext:
    """Experimental folders and measures associated with one saved fit."""

    yarn_type: str
    knit_type: tuple[str, str]
    exp_data_type: str
    sample_folder: str
    exp_type: list[str]
    umat_type: list[str]


@dataclass(frozen=True)
class YAxisDisplay:
    """Describe one plotted y-axis quantity, scale, and unit-bearing label."""

    quantity: str
    scale_from_p_stress: float
    axis_label: str
    figure_label: str


DEFAULT_SAMPLE_FOLDERS = {
    "Interlock": "BRT_15mm_Tri_Int_S06",
    "SingleJersey": "BRT_15mm_Tri_SJ_S02",
    "Rib": "BRT_15mm_Tri_Rib_S03",
    "Purl": "BRT_15mm_Tri_Purl_S02",
}
DEFAULT_KNIT_TOKENS = {
    "Interlock": "interlock",
    "SingleJersey": "singlejersey",
    "Rib": "rib",
    "Purl": "purl",
}


def _metadata_pair(value: object, fallback: list[str]) -> list[str]:
    """Read a saved two-item measure pair, for example ``['H', 'P']``."""
    parsed = value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return [str(parsed[0]), str(parsed[1])]
    return list(fallback)


def resolve_y_axis_display(
    quantity: str,
    reference_thickness_mm: float = REFERENCE_THICKNESS_MM,
) -> YAxisDisplay:
    """Resolve P stress or Force/reference width display settings."""
    normalized = str(quantity).strip().lower()
    if normalized == "p_stress":
        return YAxisDisplay(
            quantity=normalized,
            scale_from_p_stress=KPA_PER_MPA,
            axis_label=r"P stress (kPa)",
            figure_label="P stress",
        )
    if normalized == "force_per_width":
        thickness = float(reference_thickness_mm)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "REFERENCE_THICKNESS_MM must be finite and positive for "
                "Force/reference width."
            )
        return YAxisDisplay(
            quantity=normalized,
            scale_from_p_stress=thickness,
            axis_label="Force/width (N/mm)",
            figure_label="Force/reference width",
        )
    options = ", ".join(Y_AXIS_OPTIONS)
    raise ValueError(f"Unknown Y_AXIS_QUANTITY {quantity!r}; choose from: {options}")


def resolve_y_limits(
    configured_limits: AxisLimits,
    command_line_limits: AxisLimits,
) -> AxisLimits:
    """Return limits already expressed in the active y-axis display units."""
    selected_limits = (
        command_line_limits
        if command_line_limits is not None
        else configured_limits
    )
    if selected_limits is None:
        return None
    return (float(selected_limits[0]), float(selected_limits[1]))


def fit_plot_context(
    selected_fit: FitResult | None,
    *,
    default_yarn: str,
    default_knit: tuple[str, str],
    default_data_type: str,
) -> FitPlotContext:
    """Infer plot inputs from a fit artifact, retaining manual defaults if absent."""
    default_sample = DEFAULT_SAMPLE_FOLDERS[default_knit[0]]
    if selected_fit is None:
        return FitPlotContext(
            default_yarn,
            default_knit,
            default_data_type,
            default_sample,
            ["H", "P"],
            ["F", "Sigma"],
        )

    metadata = selected_fit.metadata
    nested = metadata.get("experiment_context", {})
    context = nested if isinstance(nested, Mapping) else {}
    knit = str(context.get("knit", metadata.get("Knit", default_knit[0])))
    if knit not in DEFAULT_SAMPLE_FOLDERS:
        options = ", ".join(DEFAULT_SAMPLE_FOLDERS)
        raise ValueError(f"Fit {selected_fit.code} has unsupported knit {knit!r}; {options}")
    token = str(
        context.get(
            "knit_token",
            metadata.get("Knit_Token", DEFAULT_KNIT_TOKENS[knit]),
        )
    )
    yarn = str(context.get("yarn_type", metadata.get("Yarn_Type", default_yarn)))
    data_type = str(
        context.get("exp_data_type", metadata.get("Exp_Data_Type", default_data_type))
    ).lower()
    if data_type not in {"raw"}:
        raise ValueError(
            f"Fit {selected_fit.code} has unsupported data type {data_type!r}"
        )
    sample = str(
        context.get(
            "sample_folder",
            metadata.get("Sample_Folder", DEFAULT_SAMPLE_FOLDERS[knit]),
        )
    )
    exp_type = _metadata_pair(
        context.get("exp_type", metadata.get("Exp_Type")),
        ["H", "P"],
    )
    umat_type = [
        str(context.get("umat_strain", metadata.get("UMAT_Strain", "F"))),
        str(context.get("umat_stress", metadata.get("UMAT_Stress", "Sigma"))),
    ]
    return FitPlotContext(yarn, (knit, token), data_type, sample, exp_type, umat_type)


def resolve_plot_dataset(
    context: FitPlotContext,
    dataset_key: str | None,
    *,
    confirm_override: bool,
    input_func=input,
) -> tuple[FitPlotContext, exp_datasets.ExperimentalDataset]:
    """Resolve a catalog dataset and confirm any saved-fit override."""
    original = exp_datasets.find_dataset(
        context.yarn_type,
        context.knit_type[0],
        context.sample_folder,
        context.exp_data_type,
    )
    if dataset_key is None:
        return context, original

    selected = exp_datasets.get_dataset(dataset_key)
    if confirm_override:
        exp_datasets.confirm_dataset_override(
            original,
            selected,
            input_func=input_func,
        )
    selected_context = FitPlotContext(
        yarn_type=selected.yarn_type,
        knit_type=(selected.knit, selected.knit_token),
        exp_data_type=selected.data_type,
        sample_folder=selected.sample_folder,
        exp_type=context.exp_type,
        umat_type=context.umat_type,
    )
    return selected_context, selected


def plot_axis_directions(
    dataset: exp_datasets.ExperimentalDataset,
    ratio: str,
) -> tuple[str, str]:
    """Return physical plot labels, including the Interlock display correction."""
    directions = exp_datasets.axis_directions(dataset, ratio)
    if dataset.knit == "Interlock" and INVERT_INTERLOCK_PLOT_AXES:
        return directions[::-1]
    return directions

def _legacy_single_knit_main() -> None:
    '''Main part of the code'''
    args = parse_args()
    selected_fit: FitResult | None = None
    if args.fit_code is not None:
        selected_fit = load_fit(args.fit_code, args.results_csv)

    # Manual fallbacks used only with no fit; examples are Tripoly/Purl/raw.
    plot_context = fit_plot_context(
        selected_fit,
        default_yarn="Tripoly",
        default_knit=("Purl", "purl"),
        default_data_type="raw",
    )
    plot_context, selected_dataset = resolve_plot_dataset(
        plot_context,
        EXPERIMENTAL_DATASET_KEY,
        confirm_override=selected_fit is not None,
    )
    yarn_type = plot_context.yarn_type
    knit_type = plot_context.knit_type
    exp_data_type = plot_context.exp_data_type
    sample_folder = plot_context.sample_folder
    # Material Model
    if selected_fit is None:
        mat_model = "SV_Exp"
    else:
        mat_model = selected_fit.model
    model_spec = load_model_spec(mat_model)
    
    # File paths
    umass_dir = RELEASE_CODE_ROOT
    ssmccf_dir = RELEASE_GENERATED_ROOT

    umat_path = model_spec.fortran_source
    figure_directory = (
        RELEASE_FIGURE_ROOT
        / mat_model
        / "Plots_from_plotter"
        / yarn_type
        / knit_type[0]
    )
    if selected_fit is not None:
        figure_directory = figure_directory / f"fit_{selected_fit.code:04d}"
    figure_directory.mkdir(parents=True, exist_ok=True)
    # Material parameters
    param_all = {
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
        'Rib': {
            'mu':       0.0002648277694443776,      # Linear
            'mu2':      0.006623901895993168,      # Linear
            'K_iso1':   0,      # Isotropic
            'K_iso2':   0,      # Isotropic  
            'alpha_1':  3,      # Isotropic 
            'alpha_2':  3,      # Isotropic 
            'K_1ani1':  0.2202313531754161,     # Anisotropic 
            'beta_1':   9,     # Anisotropic   
            'K_2ani1':  0,      # Anisotropic  
            'beta_2':   53,      # Anisotropic   
            'K_1ani2':  4.584377556433681,     # Anisotropic  
            'gamma_1':  56,      # Anisotropic  
            'K_2ani2':  0,      # Anisotropic  
            'gamma_2':  56,      # Anisotropic  
            'K_kop1':   0,      # Coupling  
            'delta_1':  3,      # Coupling 
            'K_kop2':   0,      # Coupling  
            'delta_2':  3,      # Coupling
            'K_kop12':  4.2652758243275795,      # Coupling 
            'zeta':     10,      # Coupling    
            'C1':       1,         
            'C2':       1,      
            'theta1':   0.4898102423622088,  
            'theta2':   119.59406141646295,
            'mu3':      0.001443221           
        },
        'Purl': {
            'mu':       0.015217527,      # Linear
            'mu2':      0.005943088,      # Linear
            'K_iso1':   0,      # Isotropic
            'K_iso2':   0,      # Isotropic  
            'alpha_1':  3,      # Isotropic 
            'alpha_2':  3,      # Isotropic 
            'K_1ani1':  3.445545642,     # Anisotropic 
            'beta_1':   11,     # Anisotropic   
            'K_2ani1':  0.220635979,      # Anisotropic  
            'beta_2':   588,      # Anisotropic   
            'K_1ani2':  0.214419753,     # Anisotropic  
            'gamma_1':  109,      # Anisotropic  
            'K_2ani2':  0,      # Anisotropic  
            'gamma_2':  7,      # Anisotropic  
            'K_kop1':   0,      # Coupling  
            'delta_1':  3,      # Coupling 
            'K_kop2':   0,      # Coupling  
            'delta_2':  3,      # Coupling
            'K_kop12':  4.907433976,      # Coupling 
            'zeta':     4,      # Coupling    
            'C1':       1,         
            'C2':       1,      
            'theta1':   0,  
            'theta2':   90,
            'mu3':      0.0088            
        },
        'SingleJersey': {
            'mu':       0.0157652378524096,      # Linear
            'mu2':      0.0184662792724592,      # Linear
            'K_iso1':   0,      # Isotropic
            'K_iso2':   0,      # Isotropic  
            'alpha_1':  3,      # Isotropic 
            'alpha_2':  3,      # Isotropic 
            'K_1ani1':  2.19573871314823,     # Anisotropic 
            'beta_1':   75,     # Anisotropic   
            'K_2ani1':  0,      # Anisotropic  
            'beta_2':   513,      # Anisotropic   
            'K_1ani2':  0.264034705035234,     # Anisotropic  
            'gamma_1':  82,      # Anisotropic  
            'K_2ani2':  3.16116991988597E-07,      # Anisotropic   
            'gamma_2':  11,      # Anisotropic  
            'K_kop1':   0,      # Coupling  
            'delta_1':  3,      # Coupling 
            'K_kop2':   0,      # Coupling  
            'delta_2':  3,      # Coupling
            'K_kop12':  3.9350731910975,      # Coupling 
            'zeta':     3,      # Coupling    
            'C1':       1,         
            'C2':       1,      
            'theta1':   0,  
            'theta2':   90,
            'mu3':      0.00999999205595173            
        }                
    }
    if selected_fit is None:
        parameters_full = param_all[knit_type[0]]
        parameters = convert_parameters(parameters_full, mat_model)
    else:
        # Historical SV_log fits may contain the removed K2 parameters.
        parameters = {
            name: selected_fit.parameters[name]
            for name in model_spec.parameter_names
        }
        print(
            f"Using fit {selected_fit.code} ({mat_model}) from {selected_fit.source}"
        )
    
    # Convert the named parameters to the selected UMAT's expected PROPS order.
    sim_par = model_spec.pack_parameters(parameters)

    # Compile UMAT only after its parameter interface has been validated.
    umat.compile_umat_meson(umat_path)
    
    # UMAT and experiment strain and stress types
    exp_type = plot_context.exp_type
    umat_type = plot_context.umat_type

    selected_ratios = args.ratios if args.ratios is not None else SELECTED_RATIOS
    selected_paths = select_experimental_paths(
        selected_dataset.ratio_paths(), selected_ratios
    )
    axis_directions_by_ratio = {
        ratio: plot_axis_directions(selected_dataset, ratio)
        for ratio in selected_paths
    }
    y_axis = resolve_y_axis_display(
        args.y_axis_quantity,
        REFERENCE_THICKNESS_MM,
    )
    y_limits = resolve_y_limits(DEFAULT_Y_LIMITS, args.y_limits)

    plot_fit_vs_exp(
        selected_paths,
        sim_par,
        exp_type,
        umat_type,
        knit_type,
        figure_directory,
        x_limits=args.x_limits,
        y_limits=y_limits,
        y_axis=y_axis,
        # curve receives an identity anchor (for example, H11 = H22 = 0).
        start_model_curve_at_identity=False,
        # Raw data keeps its existing measured-row interpolation regardless of
        model_curve_path_mode=(
            "measured_path"
        ),
        axis_directions_by_ratio=axis_directions_by_ratio,
    )


def validate_four_knit_configuration() -> None:
    """Validate the editable four-row fit and dataset settings."""
    expected_knits = {"Interlock", "Rib", "Purl", "SingleJersey"}
    if len(KNIT_ROW_ORDER) != 4 or set(KNIT_ROW_ORDER) != expected_knits:
        raise ValueError(
            "KNIT_ROW_ORDER must contain Interlock, Rib, Purl, and "
            "SingleJersey exactly once."
        )
    if set(FIT_CODES_BY_KNIT) != expected_knits:
        raise ValueError("FIT_CODES_BY_KNIT must define exactly the four knit rows.")
    if set(EXPERIMENTAL_DATASET_KEYS_BY_KNIT) != expected_knits:
        raise ValueError(
            "EXPERIMENTAL_DATASET_KEYS_BY_KNIT must define exactly the four knit rows."
        )
    if set(X_LIMITS_BY_KNIT) != expected_knits:
        raise ValueError("X_LIMITS_BY_KNIT must define exactly the four knit rows.")
    if set(Y_LIMITS_BY_KNIT) != expected_knits:
        raise ValueError("Y_LIMITS_BY_KNIT must define exactly the four knit rows.")
    if len(set(FIT_CODES_BY_KNIT.values())) != 4:
        raise ValueError("FIT_CODES_BY_KNIT must contain four distinct fit codes.")


def create_four_knit_figure(
    data_by_knit: Mapping[str, Mapping[str, RatioPlotData]],
    ratios: Sequence[str],
    *,
    x_limits: AxisLimits,
    y_limits: AxisLimits,
    y_axis: YAxisDisplay,
) -> Figure:
    """Create the requested four-knit by three-ratio manuscript figure."""
    selected_ratios = tuple(ratios)
    if len(selected_ratios) != 3:
        raise ValueError("Figure 4 requires exactly three selected ratios.")
    missing_knits = [knit for knit in KNIT_ROW_ORDER if knit not in data_by_knit]
    if missing_knits:
        raise ValueError(f"Missing plotted knit rows: {missing_knits}")

    row_count = len(KNIT_ROW_ORDER)
    column_count = len(selected_ratios)
    # Use the accepted manuscript canvas directly instead of scaling a nominal
    # per-panel width; this preserves the row-by-row reference proportions.
    figdim = (FIGURE_4_WIDTH_INCHES, FIGURE_4_HEIGHT_INCHES)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=figdim,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=FIGURE_4_LEFT_MARGIN,
        right=FIGURE_4_RIGHT_MARGIN,
        bottom=FIGURE_4_BOTTOM_MARGIN,
        top=FIGURE_4_TOP_MARGIN,
        hspace=FIGURE_4_ROW_SPACING,
        wspace=FIGURE_4_COLUMN_SPACING,
    )

    for row_index, knit in enumerate(KNIT_ROW_ORDER):
        row_data = data_by_knit[knit]
        for column_index, ratio in enumerate(selected_ratios):
            if ratio not in row_data:
                raise ValueError(f"Missing ratio {ratio!r} for {knit}.")
            axis = axes[row_index, column_index]
            plot_data = row_data[ratio]
            legend_mode = (
                "full"
                if row_index == 0 and column_index == 0 and SHOW_CURVE_LEGEND
                else "r2_only"
            )
            draw_ratio_on_axes(
                plot_data,
                axis,
                legend_mode=legend_mode,
                y_axis=y_axis,
            )
            format_ratio_axes(
                axis,
                show_xlabel=row_index == row_count - 1,
                show_ylabel=column_index == 0,
                show_y_tick_labels=(
                    SHOW_Y_TICK_LABELS_ON_EACH_SUBPLOT or column_index == 0
                ),
                # Command-line limits override all rows; otherwise use the
                # editable limit assigned to this knit (for example, Purl).
                x_limits=x_limits if x_limits is not None else X_LIMITS_BY_KNIT[knit],
                y_limits=y_limits if y_limits is not None else Y_LIMITS_BY_KNIT[knit],
                y_axis=y_axis,
            )
            if SHOW_PLOT_TITLE:
                axis.set_title(
                    "Course:Wales "
                    + exp_datasets.physical_ratio_label(
                        ratio,
                        plot_data.axis_directions,
                    ),
                    fontweight="bold",
                    fontsize=PLOT_TITLE_FONT_SIZE,
                    pad=PLOT_TITLE_PAD,
                )
    return figure


def main() -> None:
    """Generate the complete four-knit by three-ratio figure in one run."""
    args = parse_args()
    validate_four_knit_configuration()
    selected_ratios = validate_selected_ratios(
        args.ratios if args.ratios is not None else SELECTED_RATIOS
    )
    if len(selected_ratios) != 3:
        raise ValueError("plotter_fig_4.py requires exactly three ratios.")

    active_fit_codes = (
        FIT_CODES_BY_KNIT
        if args.fit_codes is None
        else dict(zip(KNIT_ROW_ORDER, args.fit_codes, strict=True))
    )
    fits = {
        knit: load_fit(active_fit_codes[knit], args.results_csv)
        for knit in KNIT_ROW_ORDER
    }
    for knit, selected_fit in fits.items():
        fit_knit = fit_plot_context(
            selected_fit,
            default_yarn="Tripoly",
            default_knit=(knit, DEFAULT_KNIT_TOKENS[knit]),
            default_data_type="raw",
        ).knit_type[0]
        if fit_knit != knit:
            raise ValueError(
                f"Fit {selected_fit.code} belongs to {fit_knit}, not configured row {knit}."
            )

    model_ids = {selected_fit.model for selected_fit in fits.values()}
    if len(model_ids) != 1:
        raise ValueError(f"All four fits must use one material model; found {model_ids}.")
    model_id = next(iter(model_ids))
    model_spec = load_model_spec(model_id)
    umat.compile_umat_meson(model_spec.fortran_source)

    data_by_knit: dict[str, dict[str, RatioPlotData]] = {}
    for knit in KNIT_ROW_ORDER:
        selected_fit = fits[knit]
        context = fit_plot_context(
            selected_fit,
            default_yarn="Tripoly",
            default_knit=(knit, DEFAULT_KNIT_TOKENS[knit]),
            default_data_type="raw",
        )
        context, dataset = resolve_plot_dataset(
            context,
            EXPERIMENTAL_DATASET_KEYS_BY_KNIT[knit],
            confirm_override=EXPERIMENTAL_DATASET_KEYS_BY_KNIT[knit] is not None,
        )
        paths = select_experimental_paths(dataset.ratio_paths(), selected_ratios)
        parameters = {
            name: selected_fit.parameters[name]
            for name in model_spec.parameter_names
        }
        sim_parameters = model_spec.pack_parameters(parameters)
        data_by_knit[knit] = {
            ratio: build_ratio_plot_data(
                ratio,
                paths[ratio],
                sim_parameters,
                context.exp_type,
                context.umat_type,
                start_model_curve_at_identity=False,
                model_curve_path_mode=(
                    "measured_path"
                ),
                axis_directions=plot_axis_directions(dataset, ratio),
            )
            for ratio in selected_ratios
        }
        print(
            f"Prepared {knit} row from fit {selected_fit.code} "
            f"({selected_fit.model})."
        )

    display = resolve_y_axis_display(
        args.y_axis_quantity,
        REFERENCE_THICKNESS_MM,
    )
    figure = create_four_knit_figure(
        data_by_knit,
        selected_ratios,
        x_limits=args.x_limits,
        y_limits=(tuple(args.y_limits) if args.y_limits is not None else None),
        y_axis=display,
    )
    figure_directory = OUTPUT_DIRECTORY
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure_path = figure_directory / OUTPUT_FILE_NAME
    # Override the shared tight-bbox rcParam so the saved SVG keeps the measured
    # 6.75 x 8.0-inch Figure 4 canvas instead of being recropped around content.
    with plt.rc_context({"savefig.bbox": None}):
        figure.savefig(figure_path, transparent=True)
    print(f"The complete 4x3 figure was saved to {figure_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse optional fit selection while preserving the script's current defaults."""
    parser = argparse.ArgumentParser(
        description="Plot material-model fits against experimental data."
    )
    parser.set_defaults(fit_code=DEFAULT_FIT_CODE)
    parser.add_argument(
        "--fit-codes",
        dest="fit_codes",
        type=int,
        nargs=4,
        default=None,
        metavar=("INTERLOCK", "RIB", "PURL", "SINGLE_JERSEY"),
        help=(
            "Optional four-fit override in row order. If omitted, use "
            "FIT_CODES_BY_KNIT from the script."
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=pl.Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Fit-results CSV. Default: {DEFAULT_RESULTS_CSV}",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=None,
        metavar="RATIO",
        help=(
            'Ratios to plot, for example --ratios "0.5 to 1" '
            '"1 to 0.5" "1 to 1". If omitted, use SELECTED_RATIOS.'
        ),
    )
    parser.add_argument(
        "--x-limits",
        "--xlim",
        dest="x_limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Global x-axis limits as MIN MAX. If omitted, use "
            "X_LIMITS_BY_KNIT for each row."
        ),
    )
    parser.add_argument(
        "--y-limits",
        "--ylim",
        dest="y_limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Y-axis limits as MIN MAX in the selected display units. If omitted, "
            "use Y_LIMITS_BY_KNIT for each row."
        ),
    )
    parser.add_argument(
        "--y-axis",
        dest="y_axis_quantity",
        choices=Y_AXIS_OPTIONS,
        default=Y_AXIS_QUANTITY,
        help=(
            "Y-axis quantity: p_stress in kPa or force_per_width in N/mm. "
            f"Default: {Y_AXIS_QUANTITY}"
        ),
    )
    return parser.parse_args()


def validate_selected_ratios(ratios: Sequence[str]) -> tuple[str, ...]:
    """Return a valid, duplicate-free ratio selection in the requested order."""
    if isinstance(ratios, str):
        ratios = (ratios,)
    if not ratios:
        raise ValueError("SELECTED_RATIOS must contain at least one ratio.")

    canonical = {ratio.lower(): ratio for ratio in ALL_RATIO_ORDER}
    selected: list[str] = []
    for ratio in ratios:
        normalized = str(ratio).strip().replace('_', ' ').replace(':', ' to ')
        normalized = ' '.join(normalized.split()).lower()
        if normalized not in canonical:
            options = ", ".join(ALL_RATIO_ORDER)
            raise ValueError(f"Unknown ratio {ratio!r}; choose from: {options}")
        selected.append(canonical[normalized])

    if len(selected) != len(set(selected)):
        raise ValueError(f"SELECTED_RATIOS contains duplicates: {selected}")
    return tuple(selected)


def select_experimental_paths(
    experimental_paths: PathMap,
    ratios: Sequence[str],
) -> dict[str, pl.Path]:
    """Filter the available paths while preserving the selected ratio order."""
    selected = validate_selected_ratios(ratios)
    return {ratio: experimental_paths[ratio] for ratio in selected}


def ratio_grid_shape(
    ratio_count: int,
    max_columns: int = MAX_SUBPLOT_COLUMNS,
) -> tuple[int, int]:
    """Return the grid shape for up to eight selected non-equibiaxial ratios."""
    if ratio_count < 1 or ratio_count > 8:
        raise ValueError("A combined ratio grid requires between 1 and 8 ratios.")
    if not isinstance(max_columns, int) or isinstance(max_columns, bool):
        raise ValueError("MAX_SUBPLOT_COLUMNS must be an integer between 1 and 8.")
    if max_columns < 1 or max_columns > 8:
        raise ValueError("MAX_SUBPLOT_COLUMNS must be an integer between 1 and 8.")

    columns = min(ratio_count, max_columns)
    rows = int(np.ceil(ratio_count / columns))
    return rows, columns


def ratio_selection_stem(ratios: Sequence[str]) -> str:
    """Build a deterministic filename stem for a partial ratio selection."""
    tokens = [ratio.replace('.', 'p').replace(' ', '_') for ratio in ratios]
    return "selected_" + "__".join(tokens)

def plot_fit_vs_exp(
    experimental_paths: PathMap,
    sim_par: Sequence[float | int],
    exp_type: list[str],
    umat_type: list[str],
    knit_type: tuple[str, str],
    fig_path: pl.Path,
    x_limits: AxisLimits = None,
    y_limits: AxisLimits = None,
    start_model_curve_at_identity: bool = False,
    model_curve_path_mode: str = "measured_path",
    y_axis: YAxisDisplay | None = None,
    axis_directions_by_ratio: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    '''Plots a strain-stress curve of experimental data against and 
    the curve produced by the UMAT and the material parameters'''
    display = y_axis or resolve_y_axis_display(
        Y_AXIS_QUANTITY, REFERENCE_THICKNESS_MM
    )
    plot_mode = PLOT_MODE.strip().lower()
    if plot_mode == "individual":
        for ratio, path in experimental_paths.items():
            plot_data = build_ratio_plot_data(
                ratio,
                path,
                sim_par,
                exp_type,
                umat_type,
                start_model_curve_at_identity=start_model_curve_at_identity,
                model_curve_path_mode=model_curve_path_mode,
                axis_directions=(
                    axis_directions_by_ratio[ratio]
                    if axis_directions_by_ratio is not None
                    else ("course", "wales")
                ),
            )
            plot_individual_ratio(
                plot_data,
                knit_type,
                fig_path,
                x_limits,
                y_limits,
                legend_mode="full" if SHOW_CURVE_LEGEND else "r2_only",
                y_axis=display,
            )
    elif plot_mode == "all":
        plot_all_ratios(experimental_paths, sim_par, exp_type, umat_type, knit_type,
                        fig_path, x_limits, y_limits, start_model_curve_at_identity,
                        model_curve_path_mode, display, axis_directions_by_ratio)
    else:
        raise ValueError('PLOT_MODE must be either "Individual" or "All"')


def build_ratio_plot_data(
    ratio: str,
    path: pl.Path,
    sim_par: Sequence[float | int],
    exp_type: list[str],
    umat_type: list[str],
    start_model_curve_at_identity: bool = False,
    model_curve_path_mode: str = "measured_path",
    axis_directions: tuple[str, str] = ("course", "wales"),
) -> RatioPlotData:
    """Load one ratio and calculate model curves and R squared values."""
    exp_data = np.loadtxt(path)
    exp_strains = exp_data[:,:6]
    exp_stress = exp_data[:,6:12]

    sim_data_at_exp = umat.generate_exp_data(10, 0, 10, exp_type, np.array(sim_par), umat_type,
                                             strains = exp_strains)
    sim_stress_at_exp = sim_data_at_exp[:,6:12]

    model_strains = build_model_curve_strain_path(
        exp_strains,
        MODEL_CURVE_POINTS,
        model_curve_path_mode,
        start_at_identity=start_model_curve_at_identity,
    )
    sim_data_curve = umat.generate_exp_data(10, 0, 10, exp_type, np.array(sim_par), umat_type,
                                            strains = model_strains)
    sim_stress_curve = sim_data_curve[:,6:12]
    r_squared = [
        calc_r_squared(exp_stress[:, axis], sim_stress_at_exp[:, axis])
        for axis in range(2)
    ]
    return RatioPlotData(
        ratio=ratio,
        path=path,
        axis_directions=axis_directions,
        exp_strains=exp_strains,
        exp_stress=exp_stress,
        model_strains=model_strains,
        sim_stress_curve=sim_stress_curve,
        r_squared=r_squared,
    )


def plot_individual_ratio(
    plot_data: RatioPlotData,
    knit_type: tuple[str, str],
    fig_path: pl.Path,
    x_limits: AxisLimits,
    y_limits: AxisLimits,
    legend_mode: str = "full",
    save_stem: str | None = None,
    y_axis: YAxisDisplay | None = None,
) -> None:
    """Plot one ratio using the original individual-plot layout."""
    display = y_axis or resolve_y_axis_display(
        Y_AXIS_QUANTITY, REFERENCE_THICKNESS_MM
    )
    figdim = (PLOT_WIDTH_INCHES, PLOT_WIDTH_INCHES / PLOT_ASPECT_RATIO)
    fig, ax = plt.subplots(figsize=figdim, tight_layout=True)
    draw_ratio_on_axes(plot_data, ax, legend_mode=legend_mode, y_axis=display)
    format_ratio_axes(ax, show_xlabel=True, show_ylabel=True,
                      x_limits=x_limits, y_limits=y_limits, y_axis=display)
    if SHOW_PLOT_TITLE:
        ax.set_title(
            f"Experimental vs Model {display.figure_label} and Stretch - "
            f"{knit_type[0]} Course:Wales "
            f"{exp_datasets.physical_ratio_label(plot_data.ratio, plot_data.axis_directions)}",
            fontweight='bold',
            fontsize=PLOT_TITLE_FONT_SIZE,
            pad=PLOT_TITLE_PAD,
        )

    if fig_path is not None:
        fig_add = fig_path / f"{save_stem or plot_data.path.stem}.svg"
        plt.savefig(fig_add, transparent = True)
        print(f"The image {fig_add.stem} has been saved to {fig_add}")

    plt.show()


def plot_all_ratios(
    experimental_paths: PathMap,
    sim_par: Sequence[float | int],
    exp_type: list[str],
    umat_type: list[str],
    knit_type: tuple[str, str],
    fig_path: pl.Path,
    x_limits: AxisLimits,
    y_limits: AxisLimits,
    start_model_curve_at_identity: bool,
    model_curve_path_mode: str,
    y_axis: YAxisDisplay | None = None,
    axis_directions_by_ratio: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    """Plot selected ratios in an automatic grid, preserving the all-nine format."""
    display = y_axis or resolve_y_axis_display(
        Y_AXIS_QUANTITY, REFERENCE_THICKNESS_MM
    )
    selected_ratios = tuple(experimental_paths)
    all_nine_selected = (
        len(selected_ratios) == len(ALL_RATIO_ORDER)
        and set(selected_ratios) == set(ALL_RATIO_ORDER)
    )
    summary_ratios = (
        tuple(ratio for ratio_row in ALL_PLOT_RATIOS for ratio in ratio_row)
        if all_nine_selected
        else selected_ratios
    )
    rows, columns = ratio_grid_shape(len(summary_ratios))
    subplot_height = PLOT_WIDTH_INCHES / PLOT_ASPECT_RATIO
    figdim = (
        min(PLOT_WIDTH_INCHES * columns, MAX_FIGURE_WIDTH_INCHES),
        subplot_height * (rows + ALL_PLOT_ROW_SPACING * (rows - 1)),
    )
    fig, axes = plt.subplots(rows, columns, figsize=figdim, squeeze=False)
    fig.subplots_adjust(
        hspace=ALL_PLOT_ROW_SPACING,
        wspace=ALL_PLOT_COLUMN_SPACING,
    )

    for index, ratio in enumerate(summary_ratios):
        row_idx, col_idx = divmod(index, columns)
        ax = axes[row_idx, col_idx]
        plot_data = build_ratio_plot_data(
            ratio,
            experimental_paths[ratio],
            sim_par,
            exp_type,
            umat_type,
            start_model_curve_at_identity=start_model_curve_at_identity,
            model_curve_path_mode=model_curve_path_mode,
            axis_directions=(
                axis_directions_by_ratio[ratio]
                if axis_directions_by_ratio is not None
                else ("course", "wales")
            ),
        )
        legend_mode = (
            "full" if index == 0 and SHOW_CURVE_LEGEND else "r2_only"
        )
        draw_ratio_on_axes(
            plot_data,
            ax,
            legend_mode=legend_mode,
            y_axis=display,
        )
        format_ratio_axes(
            ax,
            show_xlabel=row_idx == rows - 1,
            show_ylabel=col_idx == 0,
            show_y_tick_labels=(
                SHOW_Y_TICK_LABELS_ON_EACH_SUBPLOT or col_idx == 0
            ),
            x_limits=x_limits,
            y_limits=y_limits,
            y_axis=display,
        )
        if SHOW_PLOT_TITLE:
            ax.set_title(
                "Course:Wales "
                + exp_datasets.physical_ratio_label(
                    ratio,
                    (
                        axis_directions_by_ratio[ratio]
                        if axis_directions_by_ratio is not None
                        else ("course", "wales")
                    ),
                ),
                fontweight='bold',
                fontsize=PLOT_TITLE_FONT_SIZE,
                pad=PLOT_TITLE_PAD,
            )

    for unused_index in range(len(summary_ratios), rows * columns):
        row_idx, col_idx = divmod(unused_index, columns)
        fig.delaxes(axes[row_idx, col_idx])

    if fig_path is not None:
        save_stem = (
            "all_ratios"
            if all_nine_selected
            else ratio_selection_stem(summary_ratios)
        )
        fig_add = fig_path / f"{save_stem}.svg"
        plt.savefig(fig_add, transparent = True)
        print(f"The image {fig_add.stem} has been saved to {fig_add}")

    plt.show()

    if all_nine_selected:
        one_to_one = build_ratio_plot_data(
            '1 to 1',
            experimental_paths['1 to 1'],
            sim_par,
            exp_type,
            umat_type,
            start_model_curve_at_identity=start_model_curve_at_identity,
            model_curve_path_mode=model_curve_path_mode,
            axis_directions=(
                axis_directions_by_ratio['1 to 1']
                if axis_directions_by_ratio is not None
                else ("course", "wales")
            ),
        )
        plot_individual_ratio(
            one_to_one,
            knit_type,
            fig_path,
            x_limits,
            y_limits,
            legend_mode="r2_only",
            save_stem="1 to 1 all ratios",
            y_axis=display,
        )


def draw_ratio_on_axes(
    plot_data: RatioPlotData,
    ax: Axes,
    legend_mode: str,
    y_axis: YAxisDisplay | None = None,
) -> None:
    """Draw one ratio on an Axes with either full or compact legend labels."""
    display = y_axis or resolve_y_axis_display(
        Y_AXIS_QUANTITY, REFERENCE_THICKNESS_MM
    )
    scale = display.scale_from_p_stress
    colors = uml_style.axis_colors(2)
    exp_labels = {"course": "Exp. Course", "wales": "Exp. Wales"}
    model_labels = {"course": "Model Course", "wales": "Model Wales"}

    for color_index, direction in enumerate(("course", "wales")):
        i = plot_data.axis_directions.index(direction)
        exp_label = exp_labels[direction] if legend_mode == "full" else "_nolegend_"
        if legend_mode == "full":
            model_label = (
                f"{model_labels[direction]} ($R^2$={format_r_squared(plot_data.r_squared[i])})"
            )
        else:
            model_label = f"$R^2$={format_r_squared(plot_data.r_squared[i])}"
        ax.scatter(
            plot_data.exp_strains[:, i] + 1,
            plot_data.exp_stress[:, i] * scale,
            color=colors[color_index],
            label=exp_label,
            marker='o',
            s=SCATTER_POINT_SIZE,
            alpha=POINT_TRANS,
        )
        ax.plot(
            plot_data.model_strains[:, i] + 1,
            plot_data.sim_stress_curve[:, i] * scale,
            color=colors[color_index],
            label=model_label,
            linewidth=MODEL_LINE_WIDTH,
        )
    legend = ax.legend(
        fontsize=PLOT_LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.0, LEGEND_VERTICAL_ANCHOR),
        borderaxespad=0.1,
        handlelength=LEGEND_LINE_LENGTH,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    for line in legend.get_lines():
        line.set_linewidth(LEGEND_LINE_WIDTH)


def format_ratio_axes(
    ax: Axes,
    show_xlabel: bool,
    show_ylabel: bool,
    x_limits: AxisLimits,
    y_limits: AxisLimits,
    show_y_tick_labels: bool = True,
    y_axis: YAxisDisplay | None = None,
) -> None:
    """Apply common axes formatting and the user-controlled line widths."""
    display = y_axis or resolve_y_axis_display(
        Y_AXIS_QUANTITY, REFERENCE_THICKNESS_MM
    )
    ax.set_box_aspect(1 / PLOT_ASPECT_RATIO)
    ax.set_xlabel(r"Stretch $\lambda$" if SHOW_AXIS_TITLE[0] and show_xlabel else "",
                  fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel(display.axis_label if SHOW_AXIS_TITLE[1] and show_ylabel else "",
                  fontsize=PLOT_LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_TICK_FONT_SIZE,
                   width=AXIS_LINE_WIDTH)
    ax.tick_params(axis="y", labelleft=show_y_tick_labels)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    uml_style.style_axes(ax)
    ax.grid(True, linewidth=GRID_LINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def interpolate_strain_path(
    strains: np.ndarray,
    n_points: int,
    start_at_identity: bool = False,
) -> np.ndarray:
    """Interpolate the measured load path, optionally beginning at stretch 1."""
    curve_strains = np.asarray(strains, dtype=float)
    if len(curve_strains) == 0:
        return curve_strains.copy()

    # H = 0 corresponds to F = I and stretch = 1 (for example, [0, 0, 0, 0, 0, 0]).
    identity_strain = np.zeros((1, curve_strains.shape[1]), dtype=float)
    if start_at_identity and not np.allclose(curve_strains[0], identity_strain[0]):
        curve_strains = np.vstack((identity_strain, curve_strains))

    if len(curve_strains) < 2:
        return curve_strains.copy()

    n_points = max(2, int(n_points))

    # Preserve the existing row-based interpolation for raw data (for example,
    # three raw rows remain equally spaced along the interpolation coordinate).
    if not start_at_identity:
        source_path = np.linspace(0.0, 1.0, len(curve_strains))
        target_path = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([
            np.interp(target_path, source_path, curve_strains[:, axis])
            for axis in range(curve_strains.shape[1])
        ])

    # Use cumulative strain-path distance so a large first loading interval
    # receives proportionally more calculated points (for example, 0.00 to 0.10).
    segment_lengths = np.linalg.norm(np.diff(curve_strains, axis=0), axis=1)
    source_path = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    unique_path_points = np.concatenate(([True], np.diff(source_path) > 0.0))
    source_path = source_path[unique_path_points]
    curve_strains = curve_strains[unique_path_points]

    if len(source_path) == 1:
        return np.repeat(curve_strains, n_points, axis=0)

    target_path = np.linspace(0.0, source_path[-1], n_points)
    return np.column_stack([
        np.interp(target_path, source_path, curve_strains[:, axis])
        for axis in range(curve_strains.shape[1])
    ])


def build_model_curve_strain_path(
    strains: np.ndarray,
    n_points: int,
    path_mode: str,
    start_at_identity: bool,
) -> np.ndarray:
    """Build only the dense strain path used to draw the material-model curve."""
    normalized_mode = path_mode.strip().lower()
    if normalized_mode == "measured_path":
        return interpolate_strain_path(
            strains,
            n_points,
            start_at_identity=start_at_identity,
        )
    if normalized_mode == "proportional_endpoint":
        return proportional_endpoint_strain_path(strains, n_points)
    raise ValueError(
        'MODEL_CURVE_PATH_MODE must be "measured_path" or '
        '"proportional_endpoint".'
    )


def proportional_endpoint_strain_path(
    strains: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Return a proportional visual path from identity to maximum deformation."""
    experimental_strains = np.asarray(strains, dtype=float)
    if experimental_strains.ndim != 2:
        raise ValueError("Experimental strains must be a two-dimensional array.")
    if len(experimental_strains) == 0:
        return experimental_strains.copy()
    if experimental_strains.shape[1] < 2:
        raise ValueError("At least two in-plane strain components are required.")

    n_points = max(2, int(n_points))

    # Select the in-plane direction with the largest tensile stretch (for
    # example, axis 1 for a 0.5:1 test), then keep its complete experimental row.
    maximum_in_plane_strains = np.max(experimental_strains[:, :2], axis=0)
    dominant_axis = int(np.argmax(maximum_in_plane_strains))
    endpoint_index = int(np.argmax(experimental_strains[:, dominant_axis]))
    endpoint_strain = experimental_strains[endpoint_index].copy()

    # Scale the complete biaxial endpoint with one loading fraction (for
    # example, alpha = [0.0, ..., 1.0] for 200 calculated model points).
    alpha = np.linspace(0.0, 1.0, n_points)
    model_strains = alpha[:, np.newaxis] * endpoint_strain[np.newaxis, :]

    # incompressible. Recalculate H33 so J = F11*F22*F33 remains one throughout.
    if model_strains.shape[1] >= 6 and np.allclose(endpoint_strain[3:6], 0.0):
        stretch_1 = 1.0 + model_strains[:, 0]
        stretch_2 = 1.0 + model_strains[:, 1]
        if np.any(stretch_1 <= 0.0) or np.any(stretch_2 <= 0.0):
            raise ValueError("The proportional path produced a nonpositive stretch.")
        model_strains[:, 2] = 1.0 / (stretch_1 * stretch_2) - 1.0

    return model_strains


def calc_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R squared between observed and predicted values."""
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    total_sum_squares = np.sum((observed - np.mean(observed)) ** 2)
    if np.isclose(total_sum_squares, 0.0):
        return np.nan
    return float(1.0 - residual_sum_squares / total_sum_squares)


def format_r_squared(value: float) -> str:
    """Format R squared values for plot annotations."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.3f}"

def convert_parameters(params: ParameterMap, model: str) -> dict[str, float | int]:
    """Convert the full SV-style parameter dictionary to the selected model."""
    if model == "faseranisotropie_SV":
        return dict(params)
    if model in {"StapletonViaux", "SV_exp_2O"}:
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
    raise ValueError(f"Unsupported material model: {model}")

if __name__ == '__main__':
    main()  
