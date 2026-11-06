"""Finalize the completed all-nine SV_exp_polycon_mo necessity calibrations."""

from __future__ import annotations

import csv
import hashlib
from release_paths import json
import math
import pathlib
import sys
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


ANALYSIS_DIR = next(p for p in pathlib.Path(__file__).resolve().parents if p.name == "Analysis")
SUPPORT_DIR = ANALYSIS_DIR.parent
sys.path[:0] = [str(SUPPORT_DIR), str(ANALYSIS_DIR)]

from Analysis.diagnostics import constitutive_stability_study as stability  # noqa: E402
from Analysis.diagnostics import model_component_contribution_study as components  # noqa: E402
from Analysis.studies.model_selection import sv_exp_component_necessity_study as old  # noqa: E402
from Analysis.studies.model_selection import sv_exp_polycon_mo_component_necessity_study as study  # noqa: E402
from Python_Wrapper import mech, umat  # noqa: E402


OUTPUT = study.OUTPUT_ROOT
FIGURES = OUTPUT / "figures"
RUN_STABILITY = True  # Example: False rebuilds calibration deliverables only.
STABILITY_POINTS = 121  # Knit-envelope samples per stretch axis.
STABILITY_ANGLES = 721  # Propagation directions over [0, pi).
STABILITY_REFINEMENTS = 32  # Local starts retained per stability scope.
KNITS = tuple(study.FULL_FIT_CODES)
STATUS_ORDER = ("Required", "Optional/substitutable", "Unsupported")
STATUS_COLORS = {"Required": "#B2182B", "Optional/substitutable": "#F4A261", "Unsupported": "#2166AC"}


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows() -> dict[str, list[dict[str, Any]]]:
    result = {}
    for knit in KNITS:
        token = knit.lower().replace(" ", "_")
        paths = sorted((study.FULL_CHECKPOINT_DIR / token).glob("M*.json"))
        if len(paths) != 39:
            raise ValueError(f"{knit}: expected 39 checkpoints, found {len(paths)}")
        result[knit] = [load_json(path) for path in paths]
    return result


def flat_tables(
    summaries: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, necessity, parameters, conditioning = [], [], [], []
    for knit in KNITS:
        summary = summaries[knit]
        adequate = set(summary["primary_adequate_candidate_ids"])
        selected = summary["selected_lowest_complexity_adequate"]["candidate_id"]
        for row in checkpoints[knit]:
            candidates.append({
                "knit": knit, "candidate_id": row["candidate_id"], "mask": row["mask"],
                "components": ";".join(row["components"]),
                "effective_parameter_count": row["effective_parameter_count"],
                "primary_cost": row["primary_cost"], "equal_ratio_cost": row["equal_ratio_cost"],
                "adequate_primary": int(row["candidate_id"] in adequate),
                "selected": int(row["candidate_id"] == selected),
                "profiled_evaluations": row["profiled_evaluations"], "elapsed_seconds": row["elapsed_seconds"],
            })
            parameters.append({
                "knit": knit, "candidate_id": row["candidate_id"], "selected": int(row["candidate_id"] == selected),
                **row["parameters"],
            })
            conditioning.append({
                "knit": knit, "candidate_id": row["candidate_id"],
                "design_rank": row["design_rank"], "design_condition_number": row["design_condition_number"],
                "active_bounds": json.dumps(row["active_bounds"], sort_keys=True),
                "minimum_constraint_margins": json.dumps(row["minimum_constraint_margins"], sort_keys=True),
                "direct_cost_difference": row["direct_cost_difference"],
            })
        necessity.extend(summary["component_classifications"])
    return candidates, necessity, parameters, conditioning


def selected_full_predictions(
    summaries: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, metrics = [], []
    for knit in KNITS:
        inputs = study.load_full_inputs(knit)
        by_id = {row["candidate_id"]: row for row in checkpoints[knit]}
        selected_id = summaries[knit]["selected_lowest_complexity_adequate"]["candidate_id"]
        selected = by_id[selected_id]
        full = by_id["M63"]
        compiled = np.atleast_2d(np.loadtxt(inputs["compiled_path"], dtype=float))
        dfgrds = np.asarray(umat.strains_to_dfgrd(compiled[:, :6], "H"), dtype=float)
        experimental_pk1 = compiled[:, 6:12]
        selected_pk2 = np.asarray(selected["predicted_pk2_axes"], dtype=float)
        full_pk2 = np.asarray(full["predicted_pk2_axes"], dtype=float)
        # Expand the two fitted axes into six-component histories for conversion.
        selected_six = np.zeros((len(dfgrds), 6)); full_six = np.zeros_like(selected_six)
        selected_six[:, :2] = selected_pk2; full_six[:, :2] = full_pk2
        selected_pk1 = old.pk2_to_pk1_history(dfgrds, selected_six)
        full_pk1 = old.pk2_to_pk1_history(dfgrds, full_six)
        direction_axis = {name: axis for axis, name in enumerate(inputs["axis_directions"])}
        for curve in inputs["curves"]:
            sl = slice(curve.start_row, curve.stop_row)
            for direction in ("course", "wale"):
                axis = direction_axis[direction]
                for label, values in (("selected", selected_pk1), ("full", full_pk1)):
                    r2, sse, sst = old.r2_parts(experimental_pk1[sl, axis], values[sl, axis])
                    metrics.append({"knit": knit, "ratio": curve.label, "direction": direction, "model": label, "scope": "all-nine_calibration", "R2": r2, "SSE": sse, "SST": sst, "observation_count": curve.stop_row-curve.start_row})
                for index in range(curve.start_row, curve.stop_row):
                    F = mech.voigt_to_matrix(dfgrds[index])
                    predictions.append({
                        "knit": knit, "ratio": curve.label, "row_index": index-curve.start_row,
                        "direction": direction, "axis": axis, "stretch_axis1": F[0, 0], "stretch_axis2": F[1, 1],
                        "experimental_pk1": experimental_pk1[index, axis], "selected_pk1": selected_pk1[index, axis], "full_pk1": full_pk1[index, axis],
                        "experimental_pk2": selected["experimental_pk2_axes"][index][axis], "selected_pk2": selected_pk2[index, axis], "full_pk2": full_pk2[index, axis],
                    })
        for label in ("selected", "full"):
            subset = [row for row in metrics if row["knit"] == knit and row["model"] == label]
            sse, sst = sum(row["SSE"] for row in subset), sum(row["SST"] for row in subset)
            metrics.append({"knit": knit, "ratio": "ALL", "direction": "pooled", "model": label, "scope": "all-nine_calibration", "R2": 1.0-sse/sst, "SSE": sse, "SST": sst, "observation_count": sum(row["observation_count"] for row in subset)})
    return predictions, metrics


def save_pair(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def calibration_figures(necessity: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(row["knit"], row["component"]): row for row in necessity}
    values = np.asarray([[STATUS_ORDER.index(lookup[(knit, component)]["primary_classification"]) for component in study.COMPONENT_ORDER] for knit in KNITS])
    cmap = mcolors.ListedColormap([STATUS_COLORS[name] for name in STATUS_ORDER])
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(6), [name.replace("_", " ").title() for name in study.COMPONENT_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(4), KNITS); ax.set_title("SV_exp_polycon_mo all-nine calibration component support")
    for i in range(4):
        for j in range(6): ax.text(j, i, STATUS_ORDER[values[i,j]].split("/")[0], ha="center", va="center", fontsize=8, color="white" if values[i,j] != 1 else "black")
    save_pair(fig, "component_status_matrix")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, knit in zip(axes.flat, KNITS):
        rows = [lookup[(knit, component)] for component in study.COMPONENT_ORDER]
        ax.bar(range(6), [row["best_without_penalty_percent"] for row in rows], color=[STATUS_COLORS[row["primary_classification"]] for row in rows])
        ax.axhline(2, color="black", linestyle="--"); ax.set_title(knit); ax.set_ylabel("Best-without penalty (%)")
        ax.set_xticks(range(6), [name.replace("_", "\n") for name in study.COMPONENT_ORDER], fontsize=8); ax.grid(axis="y", alpha=.25)
    save_pair(fig, "best_without_penalties")
    for knit in KNITS:
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 12))
        knit_rows = [row for row in predictions if row["knit"] == knit]
        ratio_order = list(dict.fromkeys(row["ratio"] for row in knit_rows))
        for ax, ratio in zip(axes.flat, ratio_order):
            for direction, color in (("course", "#1257D1"), ("wale", "#777777")):
                rows = [row for row in knit_rows if row["ratio"] == ratio and row["direction"] == direction]
                x = range(len(rows)); ax.scatter(x, [r["experimental_pk1"] for r in rows], s=8, color=color, alpha=.6)
                ax.plot(x, [r["full_pk1"] for r in rows], color=color, linestyle=":", label=f"{direction} full")
                ax.plot(x, [r["selected_pk1"] for r in rows], color=color, label=f"{direction} selected")
            ax.set_title(ratio); ax.grid(axis="y", alpha=.2); ax.set_xlabel("Observation"); ax.set_ylabel("First Piola stress")
        handles, labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(f"{knit}: full versus selected all-nine calibration"); fig.subplots_adjust(bottom=.1, top=.93, hspace=.35, wspace=.28)
        save_pair(fig, f"curves_{knit.lower().replace(' ', '_')}")


def run_stability(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    angles = stability.propagation_angles(STABILITY_ANGLES)
    for knit in KNITS:
        inputs = study.load_full_inputs(knit)
        # Dataset lookup uses the compact registry token ``SingleJersey``.
        config = components.FitConfiguration("SV_exp_polycon_mo", knit.replace(" ", ""), inputs["fit_code"], inputs["artifact"]["experiment_context"]["sample_folder"])
        datasets = components.load_ratio_datasets(config)
        parameters = {name: float(value) for name, value in summaries[knit]["selected_lowest_complexity_adequate"]["parameters"].items()}
        case = stability.FitCase("SV_exp_polycon_mo", knit, inputs["fit_code"], parameters, inputs["artifact"], datasets)
        result = stability.analyze_case(case, angles, points=STABILITY_POINTS, refinement_starts=STABILITY_REFINEMENTS)
        selected_id = summaries[knit]["selected_lowest_complexity_adequate"]["candidate_id"]
        for key in ("summary_rows", "measured_rows", "envelope_rows"):
            for row in result[key]: row["selected_candidate_id"] = selected_id
        results.append(result)
        fig = old.create_stability_figure(result); fig.suptitle(f"SV_exp_polycon_mo selected-model stability - {knit}")
        save_pair(fig, f"stability_{knit.lower().replace(' ', '_')}")
    umat_rows = stability.run_isolated_umat_validations(results)
    return (
        [row for result in results for row in result["summary_rows"]],
        [row for result in results for row in result["measured_rows"]],
        [row for result in results for row in result["envelope_rows"]],
        umat_rows,
    )


def main() -> None:
    summary = load_json(OUTPUT / "calibration_summary.json")
    summaries = {row["knit"]: row for row in summary["knits"]}
    checkpoints = checkpoint_rows()
    candidates, necessity, parameters, conditioning = flat_tables(summaries, checkpoints)
    predictions, metrics = selected_full_predictions(summaries, checkpoints)
    for name, rows in (("candidate_scores.csv", candidates), ("component_necessity.csv", necessity), ("parameters.csv", parameters), ("curve_metrics.csv", metrics), ("predictions.csv", predictions), ("conditioning.csv", conditioning)):
        write_csv(OUTPUT / name, rows)
    validation = [{"scope": "profiled_vs_compiled_umat", "knit": row["knit"], "candidate_id": row["candidate_id"], "difference": row["direct_cost_difference"], "tolerance": 1e-10, "passed": row["direct_cost_difference"] < 1e-10} for knit_rows in checkpoints.values() for row in knit_rows]
    calibration_figures(necessity, predictions)
    if RUN_STABILITY:
        stability_summary, measured, envelope, umat_rows = run_stability(summaries)
        write_csv(OUTPUT / "stability.csv", stability_summary)
        write_csv(OUTPUT / "stability_measured.csv", measured)
        write_csv(OUTPUT / "stability_envelope.csv", envelope)
        validation.extend({"scope": "stability_critical_state", **row} for row in umat_rows)
    write_csv(OUTPUT / "umat_validation.csv", validation)
    summary["deliverable_counts"] = {"candidate_scores": len(candidates), "component_necessity": len(necessity), "parameters": len(parameters), "curve_metrics": len(metrics), "predictions": len(predictions), "conditioning": len(conditioning), "umat_validation": len(validation)}
    study.atomic_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
