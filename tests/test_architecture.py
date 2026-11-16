import tempfile
import unittest
import ast
import contextlib
import io
import os
import subprocess
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from DL.architecture import ArchitectureClassifier, CLASSES, design_matrix
from DL.effects import paired_effect_table, read_pair_scores, split_aligned_matrices
from DL.FDR.FDR_control import FDR_control


class EffectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stat = self.root / "result_epoch20_featImport.csv"
        self.stat.write_text("10,2,8,1,5,3\n")
        self.prs = self.root / "prs.csv"
        self.prs.write_text("6,7,9,4,1,2\n")
        self.names = ["F1", "F2", "F3"]
        self.x = self.root / "X.csv"
        pd.DataFrame([[1, 2, 3]], columns=self.names).to_csv(self.x, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_blocked_pairing_and_both_exposure_estimates(self):
        t = paired_effect_table(self.stat, self.prs, self.names)
        np.testing.assert_array_equal(t.W_stat, [9, -3, 5])
        np.testing.assert_array_equal(t.W_prs, [2, 6, 7])
        np.testing.assert_array_equal(t.exposure_prs, [6, 7, 9])

    def test_appended_runs_require_explicit_index(self):
        with self.stat.open("a") as f:
            f.write("20,4,16,2,10,6\n")
        with self.assertRaises(ValueError):
            read_pair_scores(self.stat, self.names)
        np.testing.assert_array_equal(read_pair_scores(self.stat, self.names, 1).W, [18, -6, 10])
        result = FDR_control().calculate_multiple_W_for_selected(self.x, self.root, [("F1",)], 2)
        self.assertEqual(result, {"F1": [9, 18]})

    def test_invalid_width_and_nonfinite(self):
        for text in ("1,2,3\n", "1,2,nan,4,5,6\n"):
            self.stat.write_text(text)
            with self.assertRaises(ValueError):
                read_pair_scores(self.stat, self.names)

    def test_no_threshold_stays_unselected(self):
        control = FDR_control()
        self.assertTrue(np.isinf(control.kfilter([9, -3, 5], q=0.05)))
        self.assertEqual(control.controlFilter(self.x, self.root), [])
        np.testing.assert_array_equal(control.controlFilter_return_all(self.x, self.root)[0], [9, -3, 5])

    def test_threshold_boundary_and_zero(self):
        self.assertEqual(FDR_control().kfilter(np.ones(20), q=0.05), 1)
        self.assertTrue(np.isinf(FDR_control().kfilter(np.zeros(42))))
        self.assertEqual(len(FDR_control().controlFilter(self.x, self.root, q=0.5)), 2)

    def test_invalid_q_and_fake_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            FDR_control().kfilter([1], q=-100)
        with self.assertRaises(ValueError):
            FDR_control().controlFilter(self.x, self.root, bootstrap=True)
        with self.assertRaises(ValueError):
            FDR_control().calculate_multiple_W_for_selected(self.x, self.root, [("F1",)], 10)

    def test_matrix_alignment(self):
        stat = np.array([[1, 2, 10, 20], [3, 4, 30, 40]])
        prs = np.array([[1, 2, 50, 60], [3, 4, 70, 80]])
        x, k, p = split_aligned_matrices(stat, prs, ["F1", "F2"])
        np.testing.assert_array_equal(x, stat[:, :2])
        np.testing.assert_array_equal(p, prs[:, 2:])
        with self.assertRaises(ValueError):
            split_aligned_matrices(stat, prs[::-1], ["F1", "F2"])


class ClassifierTests(unittest.TestCase):
    def fixture(self):
        # Software test data only. This is not biological validation.
        rng = np.random.default_rng(12)
        rows, labels, groups = [], [], []
        centers = [[3, 0.1, 3, 2.9], [3, 0.1, 3, 0.1], [3, 2.9, 3, 0.1]]
        for group in range(12):
            for label, center in zip(CLASSES, centers):
                for repeat in range(3):
                    rows.append(np.asarray(center) + rng.normal(0, 0.03, 4))
                    labels.append(label)
                    groups.append("population_" + str(group))
        return pd.DataFrame(rows, columns=["exposure_stat", "stat_control", "exposure_prs", "prs_control"]), labels, groups

    def test_grouped_fit_prediction_and_serialization(self):
        import joblib
        x, y, g = self.fixture()
        m = ArchitectureClassifier(protocol_id="synthetic-software-test").fit(x, y, g, 3)
        self.assertEqual(m.oof_.groupby("group").fold.nunique().max(), 1)
        self.assertGreater(m.validation_["balanced_accuracy"], 0.95)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(m, path)
            restored = joblib.load(path)
            result = restored.predict(x, score_definition=m.score_definition, protocol_id=m.protocol_id)
        np.testing.assert_allclose(result.filter(regex="^p_").sum(axis=1), 1)
        self.assertEqual(set(result.assignment), set(CLASSES))
        rejected = m.predict(x, score_definition=m.score_definition, protocol_id=m.protocol_id, min_probability=1)
        self.assertTrue((rejected.assignment == "unresolved").all())

    def test_score_protocol_guard(self):
        x, y, g = self.fixture()
        m = ArchitectureClassifier(protocol_id="dnn-v1").fit(x, y, g, 3)
        with self.assertRaises(ValueError):
            m.predict(x, score_definition="ridge_coefficients", protocol_id="dnn-v1")
        with self.assertRaises(ValueError):
            m.predict(x, score_definition=m.score_definition, protocol_id="other")

    def test_missing_classes_groups_and_nan(self):
        x, y, g = self.fixture()
        for labels, groups in [(y, ["one"] * len(g)), ([CLASSES[0]] * len(y), g)]:
            with self.assertRaises(ValueError):
                ArchitectureClassifier(protocol_id="test").fit(x, labels, groups, 3)
        x.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            design_matrix(x, "paired")

    def test_joint_schema(self):
        x = pd.DataFrame({"exposure": [-3.], "prs_control": [-2.], "stat_control": [1.]})
        d = design_matrix(x, "joint")
        self.assertEqual(d.W_stat.iloc[0], 2)
        self.assertEqual(d.W_prs.iloc[0], 1)

    def test_command_line_round_trip(self):
        x, y, g = self.fixture()
        x["truth_class"], x["population_id"] = y, g
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores, model, output = root / "scores.csv", root / "model.joblib", root / "out.csv"
            x.to_csv(scores, index=False)
            common = ["--score-definition", "test_scores", "--protocol-id", "software_test"]
            subprocess.run([sys.executable, "-m", "DL.architecture", "train", str(scores), str(model),
                            "--folds", "3"] + common, check=True, capture_output=True)
            subprocess.run([sys.executable, "-m", "DL.architecture", "predict", str(model), str(scores),
                            str(output)] + common, check=True, capture_output=True)
            self.assertEqual(len(pd.read_csv(output)), len(x))
            self.assertTrue(Path(str(model) + ".validation.json").is_file())


class CallbackExportTests(unittest.TestCase):
    def test_callback_named_exports_match_actual_legacy_arrays(self):
        # Exercise the real callback body with deterministic layer weights.
        # Legacy Keras training itself is not available in this modern test env.
        path = Path(__file__).resolve().parents[1] / "DL/DNN/DNN.py"
        tree = ast.parse(path.read_text())
        dnn = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DNN")
        callback = next(n for n in dnn.body if isinstance(n, ast.ClassDef) and n.name == "Job_finish_Callback")
        module = ast.Module(body=[callback], type_ignores=[])
        ns = {"np": np, "pd": pd, "os": os, "uuid": uuid,
              "keras": types.SimpleNamespace(callbacks=types.SimpleNamespace(Callback=object))}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        weights = [None, np.array([[[2.], [1.]], [[3.], [4.]]]), np.ones((2, 1, 1)),
                   None, np.eye(2), np.eye(2), np.ones((2, 1))]
        class Layer:
            def __init__(self, w): self.w = w
            def get_weights(self): return [self.w]
        with tempfile.TemporaryDirectory() as tmp:
            cb = ns["Job_finish_Callback"](tmp, 2, ["A", "B"], "statKO", "fit1")
            cb.model = types.SimpleNamespace(layers=[Layer(w) for w in weights])
            with contextlib.redirect_stdout(io.StringIO()):
                cb.on_epoch_end(19)
            named = pd.read_csv(Path(tmp) / "result_epoch20_paired_scores.csv")
            np.testing.assert_array_equal(named.W, [3, -7])
            self.assertEqual(named.feature_id.tolist(), ["A", "B"])
            legacy = read_pair_scores(Path(tmp) / "result_epoch20_featImport.csv", ["A", "B"])
            np.testing.assert_array_equal(named.W, legacy.W)
            self.assertTrue((named.run_id == "fit1").all())


if __name__ == "__main__":
    unittest.main()
