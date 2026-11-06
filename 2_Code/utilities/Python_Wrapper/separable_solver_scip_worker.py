"""Isolated PySCIPOpt worker used by the certified separable solver.

The worker intentionally avoids NumPy/SciPy so SCIP cannot collide with the
Conda environment's already initialized Intel OpenMP runtime on Windows.
"""

from __future__ import annotations

from release_paths import json
import pathlib
import sys

from pyscipopt import Model, quicksum


def solve_problem(payload: dict[str, object]) -> dict[str, object]:
    """Build and solve the serialized SOS1 least-squares formulation."""
    model = Model("certified_separable_material_fit")
    if not bool(payload.get("display", False)):
        model.hideOutput()
    model.setRealParam("limits/time", float(payload["time_limit_seconds"]))
    model.setRealParam("limits/gap", float(payload["relative_gap"]))
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)

    constants = dict(payload["constants"])
    families = dict(payload["families"])
    ceiling = float(payload["amplitude_ceiling"])
    constant_variables = {
        name: model.addVar(
            name=f"coef_{name}",
            lb=float(settings["lower"]),
            ub=float(settings["upper"]),
            vtype="C",
        )
        for name, settings in constants.items()
    }

    amplitude_variables: dict[tuple[str, int], object] = {}
    for linear_name, family in families.items():
        group = []
        weights = []
        for candidate in family["candidates"]:
            exponent = int(candidate["exponent"])
            variable = model.addVar(
                name=(
                    f"amp_{linear_name}_{family['shape_parameter']}_{exponent}"
                ),
                lb=0.0,
                ub=ceiling,
                vtype="C",
            )
            amplitude_variables[(linear_name, exponent)] = variable
            group.append(variable)
            weights.append(float(exponent))
        model.addConsSOS1(
            group,
            weights=weights,
            name=f"select_{linear_name}_{family['shape_parameter']}",
        )

    target = list(payload["target"])
    residual_variables = []
    infinity = model.infinity()
    for row_index, target_value in enumerate(target):
        residual = model.addVar(
            name=f"residual_{row_index}",
            lb=-infinity,
            ub=infinity,
            vtype="C",
        )
        terms = [
            float(constants[name]["column"][row_index]) * variable
            for name, variable in constant_variables.items()
        ]
        for linear_name, family in families.items():
            for candidate in family["candidates"]:
                exponent = int(candidate["exponent"])
                terms.append(
                    float(candidate["column"][row_index])
                    * amplitude_variables[(linear_name, exponent)]
                )
        model.addCons(
            residual == quicksum(terms) - float(target_value),
            name=f"prediction_{row_index}",
        )
        residual_variables.append(residual)

    epigraph = model.addVar(name="squared_error", lb=0.0, vtype="C")
    # Scaling improves the numerical conditioning of the quadratic epigraph.
    # The parent independently recomputes the physical SSE and direct response.
    epigraph_scale = 1.0e6
    model.addCons(
        epigraph_scale * quicksum(value * value for value in residual_variables)
        <= epigraph_scale * epigraph,
        name="least_squares_epigraph",
    )
    model.setObjective(epigraph, "minimize")

    problem_path = payload.get("problem_path")
    if problem_path:
        path = pathlib.Path(str(problem_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        model.writeProblem(str(path))
    model.optimize()
    best = model.getBestSol()
    if best is not None and payload.get("solution_path"):
        path = pathlib.Path(str(payload["solution_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        model.writeBestSol(str(path), write_zeros=True)

    result: dict[str, object] = {
        "status": str(model.getStatus()),
        "has_solution": best is not None,
        "relative_gap": float(model.getGap()),
        "primal_bound": float(model.getPrimalbound()),
        "dual_bound": float(model.getDualbound()),
        "solve_seconds": float(model.getSolvingTime()),
        "constant_values": {},
        "amplitude_values": [],
    }
    if best is not None:
        result["constant_values"] = {
            name: float(model.getSolVal(best, variable))
            for name, variable in constant_variables.items()
        }
        result["amplitude_values"] = [
            [name, exponent, float(model.getSolVal(best, variable))]
            for (name, exponent), variable in amplitude_variables.items()
        ]
    return result


def main() -> None:
    """Read one JSON problem path and write one JSON solution path."""
    if len(sys.argv) != 3:
        raise SystemExit("Usage: separable_solver_scip_worker.py INPUT OUTPUT")
    input_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = solve_problem(payload)
    output_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
