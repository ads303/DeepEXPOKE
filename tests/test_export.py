import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from DL.export import combined_matrix


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.names = ["F1", "F2"]
        self.stat = pd.DataFrame([[1, 2, 10, 20], [3, 4, 30, 40]], columns=self.names + ["K1", "K2"])
        self.prs = pd.DataFrame([[1, 2, 50, 60], [3, 4, 70, 80]], columns=["a", "b", "K1", "K2"])
        self.xy = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=self.names + ["Y"])
        pd.DataFrame({"Feature": ["a", "b"], "Label": self.names}).to_csv(self.root / "features.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, command, inputs, destination, extra=()):
        return subprocess.run([sys.executable, "-m", "DL.export", command, *map(str, inputs),
                               str(destination), "--feature-map", str(self.root / "features.csv"),
                               "--cohort", "software_test", "--endpoint", "binary", *extra],
                              capture_output=True, text=True)

    def test_complete_matrix_and_cli_round_trip(self):
        paths = [self.root / f for f in ["stat.txt", "prs.csv", "xy.txt"]]
        for table, path, sep in zip([self.stat, self.prs, self.xy], paths, ["\t", ",", "\t"]):
            table.to_csv(path, sep=sep, index=False)
        output = self.root / "handoff"
        run = self.run_cli("matrices", paths, output)
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = pd.read_csv(output / "matrix.csv.gz")
        self.assertEqual(actual.columns.tolist(), ["source_row", "X_F1", "X_F2", "statKO_F1", "statKO_F2",
                                                  "prsCO_F1", "prsCO_F2", "Y"])
        np.testing.assert_array_equal(actual, [[1, 1, 2, 10, 20, 50, 60, 0], [2, 3, 4, 30, 40, 70, 80, 1]])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["outcome_counts"], {"0": 1, "1": 1})
        self.assertEqual(len(manifest["sources"]), 4)
        self.assertNotEqual(self.run_cli("matrices", paths, output).returncode, 0)

    def test_reordered_or_invalid_inputs_rejected(self):
        for stat, prs, xy in [(self.stat, self.prs.iloc[::-1], self.xy),
                              (self.stat, self.prs, self.xy.iloc[::-1]),
                              (self.stat, self.prs[["a", "b", "K2", "K1"]], self.xy),
                              (self.stat, self.prs, self.xy.assign(Y=[0, np.nan])),
                              (self.stat, self.prs, self.xy.assign(Y=[0, 2]))]:
            with self.assertRaises(ValueError):
                combined_matrix(stat, prs, xy, self.names)

    def test_score_export_preserves_negative_w_and_both_exposures(self):
        stat, prs = self.root / "stat.csv", self.root / "prs.csv"
        stat.write_text("2,3,1,4\n")
        prs.write_text("5,6,7,2\n")
        output = self.root / "effects"
        run = self.run_cli("scores", [stat, prs], output, ["--protocol-id", "test"])
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = pd.read_csv(output / "effects.csv")
        np.testing.assert_array_equal(actual.W_stat, [1, -1])
        np.testing.assert_array_equal(actual.W_prs, [-2, 4])
        np.testing.assert_array_equal(actual.exposure_prs, [5, 6])


if __name__ == "__main__":
    unittest.main()
