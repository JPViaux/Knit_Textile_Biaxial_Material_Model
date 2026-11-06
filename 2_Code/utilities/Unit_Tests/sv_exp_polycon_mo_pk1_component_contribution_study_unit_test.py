"""Focused tests for the SV_exp_polycon_mo PK1/PK2 contribution study."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


SUPPORT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

from Analysis.core import analyzer_common as analyzer  # noqa: E402
from Analysis.diagnostics import (  # noqa: E402
    sv_exp_polycon_mo_pk1_component_contribution_study as study,
)
from Python_Wrapper import mech  # noqa: E402


class ComponentContributionStudyTests(unittest.TestCase):
    def test_requested_fit_inventory_is_exact(self) -> None:
        configs = study.study_configurations()
        self.assertEqual(
            [(item.knit, item.fit_code) for item in configs],
            [
                ("Interlock", 1962),
                ("Rib", 1961),
                ("Purl", 1963),
                ("SingleJersey", 1964),
            ],
        )

    def test_normalization_scope_validation(self) -> None:
        self.assertEqual(study.validate_normalization_scope("per_ratio"), "per_ratio")
        self.assertEqual(study.validate_normalization_scope("all_ratios"), "all_ratios")
        with self.assertRaisesRegex(ValueError, "NORMALIZATION_SCOPE"):
            study.validate_normalization_scope("ratio")

    def test_stress_measure_validation(self) -> None:
        self.assertEqual(study.validate_stress_measure("PK1"), "PK1")
        self.assertEqual(study.validate_stress_measure("PK2"), "PK2")
        with self.assertRaisesRegex(ValueError, "stress_measure"):
            study.validate_stress_measure("Cauchy")

    def test_reference_peaks_support_both_scopes(self) -> None:
        identity = mech.matrix_to_voigt(np.eye(3))
        stretched = mech.matrix_to_voigt(np.diag([1.1, 1.2, 1.0]))
        datasets = {}
        for index, ratio in enumerate(study.RATIO_ORDER, start=1):
            pk1 = np.zeros((2, 6), dtype=float)
            pk2 = np.zeros((2, 6), dtype=float)
            pk1[1, 0] = float(index)
            pk1[1, 1] = float(2 * index)
            pk2[1, 0] = float(3 * index)
            pk2[1, 1] = float(4 * index)
            datasets[ratio] = study.RatioDataset(
                ratio=ratio,
                path=pathlib.Path(f"{ratio}.dat"),
                data=np.zeros((2, 12)),
                dfgrd_voigt=np.vstack((identity, stretched)),
                experimental_pk1=pk1,
                experimental_pk2=pk2,
                origin_mask=np.asarray([True, False]),
                axis_directions=("course", "wales"),
            )

        per_ratio = study.reference_peaks(datasets, "per_ratio", "PK1")
        all_ratios = study.reference_peaks(datasets, "all_ratios", "PK1")
        pk2_per_ratio = study.reference_peaks(datasets, "per_ratio", "PK2")

        self.assertEqual(per_ratio["0:1"], {"course": 1.0, "wales": 2.0})
        self.assertEqual(per_ratio["1:1"], {"course": 9.0, "wales": 18.0})
        self.assertTrue(
            all(value == {"course": 9.0, "wales": 18.0} for value in all_ratios.values())
        )
        self.assertEqual(pk2_per_ratio["0:1"], {"course": 3.0, "wales": 4.0})
        self.assertEqual(pk2_per_ratio["1:1"], {"course": 27.0, "wales": 36.0})

    def test_combined_max_coupling_matches_closed_form_and_pk1_transform(self) -> None:
        params = {
            "mu": 0.0,
            "mu2": 0.0,
            "mu3": 0.0,
            "K_1ani1": 0.0,
            "beta_1": 2.0,
            "K_1ani2": 0.0,
            "gamma_1": 3.0,
            "K_coup": 1.7,
            "zeta": 4,
            "C1": 1.0,
            "C2": 1.0,
            "theta1": 0.0,
            "theta2": 90.0,
            "a": 0.6,
            "b": 0.4,
        }
        F = np.diag([1.2, 1.1, 1.0])
        I4 = F[0, 0] ** 2
        I6 = F[1, 1] ** 2
        q = params["a"] * (I4 - 1.0) + params["b"] * (I6 - 1.0)
        expected_pk2 = (
            2.0
            * params["K_coup"]
            * params["zeta"]
            * q ** (params["zeta"] - 1)
            * np.diag([params["a"], params["b"], 0.0])
        )

        pk2 = analyzer.stress_decomposition_pk2(F, params, study.MODEL_ID)
        pk1 = analyzer.stress_decomposition(F, params, study.MODEL_ID)

        np.testing.assert_allclose(pk2["K_coup"], expected_pk2, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(pk1["K_coup"], F @ expected_pk2, rtol=0.0, atol=1.0e-14)

    def test_combined_max_coupling_uses_shared_activation(self) -> None:
        params = {
            "mu": 0.0,
            "mu2": 0.0,
            "mu3": 0.0,
            "K_1ani1": 0.0,
            "beta_1": 2.0,
            "K_1ani2": 0.0,
            "gamma_1": 2.0,
            "K_coup": 1.0,
            "zeta": 2,
            "C1": 1.0,
            "C2": 1.0,
            "theta1": 0.0,
            "theta2": 90.0,
            "a": 0.75,
            "b": 0.25,
        }
        family_one_only = np.diag([np.sqrt(1.2), 1.0, 1.0])
        compression = np.diag([0.9, 0.9, 1.0])

        active = analyzer.stress_decomposition_pk2(
            family_one_only, params, study.MODEL_ID
        )["K_coup"]
        inactive = analyzer.stress_decomposition_pk2(
            compression, params, study.MODEL_ID
        )["K_coup"]

        self.assertGreater(float(np.max(np.abs(active))), 0.0)
        np.testing.assert_array_equal(inactive, np.zeros((3, 3)))

    def test_zero_reference_requires_zero_experiment_and_model(self) -> None:
        zeros = study.normalized_sum(
            [0.0, 0.0], 0.0, np.zeros(3), label="synthetic course"
        )
        self.assertEqual(zeros, [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "zero experimental reference"):
            study.normalized_sum(
                [1.0, 0.0], 0.0, np.zeros(3), label="synthetic course"
            )

    def test_saved_fits_and_training_arrays_are_immutable_and_complete(self) -> None:
        registry = study.registry_rows()
        for config in study.study_configurations():
            _, payload = study.load_and_validate_fit(config, registry)
            datasets = study.load_ratio_datasets(config)
            self.assertEqual(tuple(datasets), study.RATIO_ORDER)
            study.verify_training_data(config, payload, datasets)


if __name__ == "__main__":
    unittest.main()
