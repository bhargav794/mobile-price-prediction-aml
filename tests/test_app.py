"""Meaningful inference and Streamlit interaction tests against real artifacts.

Run after the notebook: python -m unittest discover -s tests -p 'test_app.py' -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import compare_scenario, default_inputs, load_assets, score_csv, score_inputs


class AppInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.schema = load_assets(str(ROOT / "artifacts"))
        cls.defaults = default_inputs(cls.schema)

    def test_single_prediction_matches_persisted_pipeline(self):
        result = score_inputs(self.model, self.schema, self.defaults)
        expected = self.model.predict(pd.DataFrame([self.defaults], columns=self.schema["feature_order"]))
        self.assertEqual(result["price_range"].tolist(), expected.tolist())
        self.assertTrue(int(result.iloc[0]["price_range"]) in range(4))
        probabilities = result[[f"p_{i}" for i in range(4)]].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_whatif_uses_modified_values_without_mutating_baseline(self):
        baseline = self.defaults.copy()
        ram = int(self.schema["features"]["ram"]["max"])
        battery = int(self.schema["features"]["battery_power"]["min"])
        before, after = compare_scenario(self.model, self.schema, baseline, ram, battery)
        self.assertEqual(baseline, self.defaults)
        pd.testing.assert_frame_equal(before, score_inputs(self.model, self.schema, baseline))
        pd.testing.assert_frame_equal(after, score_inputs(
            self.model, self.schema, {**baseline, "ram": ram, "battery_power": battery}))

    def test_batch_preserves_nontrivial_order_and_ignores_export_columns(self):
        # Reversed real rows detect accidental sorting, and include actual CSV junk fields.
        frame = pd.read_csv(ROOT / "Mobile_Price_Prediction_test.csv").head(9).iloc[::-1]
        result = score_csv(self.model, self.schema, frame.to_csv(index=False).encode())
        self.assertEqual(result["id"].tolist(), frame["id"].astype(str).tolist())
        expected = self.model.predict(frame[self.schema["feature_order"]])
        self.assertEqual(result["price_range"].tolist(), expected.tolist())
        self.assertFalse(any(str(name).startswith("Unnamed:") for name in result.columns))

    def test_batch_preserves_opaque_ids_including_leading_zeros(self):
        frame = pd.DataFrame([self.defaults] * 3).assign(id=["001", "NA", "phone-z"])
        result = score_csv(self.model, self.schema, frame.to_csv(index=False).encode())
        self.assertEqual(result["id"].tolist(), ["001", "NA", "phone-z"])

    def test_batch_rejects_missing_features_target_and_invalid_values(self):
        valid = pd.DataFrame([self.defaults], columns=self.schema["feature_order"])
        invalids = {
            "missing predictor": valid.drop(columns=["ram"]),
            "target leakage": valid.assign(price_range=1),
            "unknown category": valid.assign(blue="Maybe"),
            "negative RAM": valid.assign(ram=-1),
            "nonfinite value": valid.assign(clock_speed=float("inf")),
            "fractional integer": valid.assign(n_cores=2.5),
        }
        for reason, frame in invalids.items():
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                score_csv(self.model, self.schema, frame.to_csv(index=False).encode())

    def test_batch_rejects_empty_and_unreadable_files(self):
        for content in (b"", b"ram,battery_power\n", b"\xff\xfe\x00"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                score_csv(self.model, self.schema, content)


class StreamlitInteractionTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))
        self.assertEqual(len(self.app.error), 0, str(self.app.error))

    def test_all_inputs_prediction_and_probability_chart_render(self):
        self.assertEqual(len(self.app.number_input), 13)
        self.assertEqual(len(self.app.selectbox), 7)
        self.app.number_input(key="spec_ram").set_value(3000)
        self.app.selectbox(key="spec_blue").select("Yes")
        self.app.button(key="predict_single").click().run()
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))
        self.assertEqual(len(self.app.error), 0, str(self.app.error))
        metrics = {metric.label: metric.value for metric in self.app.metric}
        self.assertIn("Predicted relative price category", metrics)
        self.assertIn(" · ", metrics["Predicted relative price category"])
        # Streamlit 1.66 renamed the chart protobuf element; either represents
        # the same successfully rendered Vega-Lite probability visualization.
        charts = [node for node in self.app if node.type in {"vega_lite_chart", "arrow_vega_lite_chart"}]
        self.assertGreaterEqual(len(charts), 1)
        self.assertEqual(self.app.session_state["baseline_inputs"]["ram"], 3000)
        self.assertEqual(self.app.slider(key="whatif_ram").value, 3000)

    def test_whatif_form_displays_both_scenarios(self):
        self.app.slider(key="whatif_ram").set_value(3500)
        self.app.slider(key="whatif_battery").set_value(1500)
        self.app.button(key="compare_scenarios").click().run()
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))
        self.assertEqual(len(self.app.error), 0, str(self.app.error))
        labels = {metric.label for metric in self.app.metric}
        self.assertIn("Baseline relative price category", labels)
        self.assertIn("Scenario relative price category", labels)
        self.assertEqual(self.app.session_state["scenario_values"], (3500, 1500))

    def test_missing_artifacts_show_recovery_instructions(self):
        probe = AppTest.from_string(
            "from unittest.mock import patch\n"
            "import app\n"
            "with patch.object(app, 'load_assets', side_effect=FileNotFoundError('Missing model.joblib')):\n"
            "    app.main()\n", default_timeout=30,
        ).run()
        self.assertEqual(len(probe.exception), 0, str(probe.exception))
        self.assertEqual(len(probe.error), 1)
        self.assertIn("Missing model.joblib", probe.error[0].value)
        self.assertIn("Run the submission notebook", probe.info[0].value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
