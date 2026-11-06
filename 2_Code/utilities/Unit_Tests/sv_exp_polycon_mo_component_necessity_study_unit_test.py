"""Focused structural tests for the SV_exp_polycon_mo necessity study."""

from __future__ import annotations

import pathlib
import json
import sys
import tempfile
import unittest


SUPPORT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUPPORT_DIR))

from Analysis.studies.model_selection import sv_exp_polycon_mo_component_necessity_study as study


class ComponentNecessityStudyTests(unittest.TestCase):
    """Verify enumeration, physical mapping, complexity, and selection semantics."""

    def test_enumerates_39_unique_admissible_masks(self) -> None:
        candidates = study.enumerate_admissible_subsets(("wale", "course"))
        self.assertEqual(len(candidates), 39)
        self.assertEqual(len({candidate.mask for candidate in candidates}), 39)
        for candidate in candidates:
            if "linear_coupling" in candidate.components:
                self.assertIn("linear_course", candidate.components)
                self.assertIn("linear_wale", candidate.components)

    def test_interlock_reversed_axis_mapping(self) -> None:
        mapping = study.physical_parameter_map(("wale", "course"))
        self.assertEqual(mapping["linear_wale"], "mu")
        self.assertEqual(mapping["linear_course"], "mu2")
        self.assertEqual(mapping["nonlinear_wale"], "K_1ani1")
        self.assertEqual(mapping["nonlinear_course"], "K_1ani2")

    def test_full_model_has_ten_effective_parameters_and_a_is_derived(self) -> None:
        full = next(
            candidate
            for candidate in study.enumerate_admissible_subsets(("wale", "course"))
            if candidate.mask == 63
        )
        self.assertEqual(full.effective_parameter_count, 10)
        definition = study.reduced_definition(full)
        self.assertEqual(definition.derived_parameter_names, ("a",))
        self.assertEqual(definition.basis_dependencies()["K_coup"], ("b", "zeta"))
        self.assertAlmostEqual(definition.derived_parameters[0].evaluate({"b": 0.37}), 0.63)

    def test_rotated_cone_definition_and_feasibility(self) -> None:
        candidate = next(
            item
            for item in study.enumerate_admissible_subsets(("wale", "course"))
            if item.mask == 7
        )
        constraint = study.reduced_definition(candidate).rotated_cone_constraints[0]
        self.assertEqual((constraint.first, constraint.second, constraint.cross), ("mu", "mu2", "mu3"))
        self.assertGreaterEqual(4.0 * 2.0 * 0.5 - 2.0**2, 0.0)
        self.assertLess(4.0 * 2.0 * 0.5 - 2.01**2, 0.0)

    def test_full_resolution_zeta_is_strictly_greater_than_two(self) -> None:
        self.assertEqual(study.FULL_ZETA_VALUES, tuple(range(3, 16)))
        self.assertNotIn(2, study.FULL_ZETA_VALUES)

    def test_inactive_shapes_are_removed(self) -> None:
        candidate = next(
            item
            for item in study.enumerate_admissible_subsets(("wale", "course"))
            if item.components == ("linear_course",)
        )
        definition = study.reduced_definition(candidate)
        self.assertEqual(definition.shape_parameters, ())
        self.assertEqual(candidate.effective_parameter_count, 1)

    def test_two_percent_boundary_and_tie_breaking(self) -> None:
        rows = [
            {"mask": 1, "components": ("linear_course",), "effective_parameter_count": 2, "score": 1.02},
            {"mask": 2, "components": ("linear_wale",), "effective_parameter_count": 2, "score": 1.02},
            {"mask": 3, "components": ("linear_course", "linear_wale"), "effective_parameter_count": 3, "score": 1.0},
        ]
        adequate = study.adequate_rows(rows, "score")
        self.assertEqual({row["mask"] for row in adequate}, {1, 2, 3})
        self.assertEqual(study.select_reduced(rows, "score")["mask"], 1)

    def test_required_optional_and_unsupported_classification(self) -> None:
        rows = [
            {"mask": 1, "components": ("linear_course", "linear_wale"), "score": 1.0},
            {"mask": 2, "components": ("linear_course", "nonlinear_course"), "score": 1.01},
            {"mask": 3, "components": ("nonlinear_coupling",), "score": 2.0},
        ]
        result = study.classify(rows, "score")
        self.assertEqual(result["linear_course"], "Required")
        self.assertEqual(result["linear_wale"], "Optional/substitutable")
        self.assertEqual(result["nonlinear_coupling"], "Unsupported")

    def test_atomic_checkpoint_replacement_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "M39.json"
            study.atomic_json(checkpoint, {"candidate_id": "M39", "complete": False})
            study.atomic_json(checkpoint, {"candidate_id": "M39", "complete": True})
            self.assertEqual(json.loads(checkpoint.read_text(encoding="utf-8"))["complete"], True)
            self.assertEqual(list(checkpoint.parent.glob("*.tmp")), [])

    def test_weighting_schemes_reconcile_when_adequate_sets_match(self) -> None:
        rows = [
            {"mask": 39, "components": ("linear_course", "linear_wale", "linear_coupling", "nonlinear_coupling"), "primary": 1.0, "equal": 1.0},
            {"mask": 63, "components": study.COMPONENT_ORDER, "primary": 1.01, "equal": 1.01},
            {"mask": 3, "components": ("linear_course", "linear_wale"), "primary": 1.2, "equal": 1.2},
        ]
        self.assertEqual(study.classify(rows, "primary"), study.classify(rows, "equal"))

    def test_synthetic_known_reduced_structure_is_recovered(self) -> None:
        target_mask = 39
        rows = []
        for candidate in study.enumerate_admissible_subsets(("course", "wale")):
            distance = (candidate.mask ^ target_mask).bit_count()
            rows.append(
                {
                    "mask": candidate.mask,
                    "components": candidate.components,
                    "effective_parameter_count": candidate.effective_parameter_count,
                    "score": 1.0 + 0.05 * distance,
                }
            )
        recovered = study.select_reduced(rows, "score")
        self.assertEqual(recovered["mask"], target_mask)


if __name__ == "__main__":
    unittest.main()
