"""Export aligned participant matrices or complete saved DNN score tables.

Outputs go to a new directory chosen by the caller. Participant matrices belong
in a private analysis/handoff directory, outside the public code repository.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DL.effects import paired_effect_table, split_aligned_matrices


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_feature_map(path):
    """Read the repository's Feature,Label dictionary in training order."""
    mapping = pd.read_csv(path, dtype=str)
    if not {"Feature", "Label"}.issubset(mapping.columns):
        raise ValueError("Feature map must contain Feature and Label columns")
    mapping = mapping[["Feature", "Label"]]
    if (mapping.empty or mapping.isna().any().any()
            or mapping.Label.duplicated().any()
            or mapping.apply(lambda c: c.str.strip().eq("")).any().any()):
        raise ValueError("Feature map requires unique nonempty IDs and nonempty names")
    return mapping


def combined_matrix(stat, prs, xy, feature_ids):
    """Validate [X,K] / [X,K] / [X,Y] and retain the original row order.

The statistical and XY tables anchor feature order using dictionary IDs. The
PRS X headers may be descriptive, but their numeric block must match exactly.
Both control blocks must be K1..Kp in the same positional feature order.
"""
    names = list(feature_ids)
    controls = ["K" + str(i + 1) for i in range(len(names))]
    if stat.columns.tolist() != names + controls:
        raise ValueError("Statistical matrix columns must be feature IDs followed by K1..Kp")
    if prs.columns[len(names):].tolist() != controls:
        raise ValueError("PRS control columns must be K1..Kp in feature order")
    x, stat_control, prs_control = split_aligned_matrices(stat, prs, names)
    if xy.columns.tolist() != names + ["Y"] or not np.array_equal(xy.iloc[:, :-1].to_numpy(), x.to_numpy()):
        raise ValueError("XY exposure columns/values must exactly match the control matrices")
    if not xy.Y.isin([0, 1]).all():
        raise ValueError("Expected nonmissing binary outcomes coded 0/1")
    result = pd.concat([x.add_prefix("X_"), stat_control.add_prefix("statKO_"),
                        prs_control.add_prefix("prsCO_"), xy[["Y"]].reset_index(drop=True)], axis=1)
    result.insert(0, "source_row", np.arange(1, len(result) + 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    matrices = sub.add_parser("matrices", help="Export all X, statKO, prsCO and Y values")
    matrices.add_argument("stat_matrix")
    matrices.add_argument("prs_matrix")
    matrices.add_argument("xy_matrix")
    matrices.add_argument("--stat-separator", choices=["tab", "comma"], default="tab")
    matrices.add_argument("--prs-separator", choices=["tab", "comma"], default="comma")
    matrices.add_argument("--xy-separator", choices=["tab", "comma"], default="tab")
    scores = sub.add_parser("scores", help="Export all paired raw scores and W contrasts")
    scores.add_argument("stat_scores")
    scores.add_argument("prs_scores")
    scores.add_argument("--stat-run-index", type=int)
    scores.add_argument("--prs-run-index", type=int)
    scores.add_argument("--protocol-id", required=True)
    for command in (matrices, scores):
        command.add_argument("destination", type=Path)
        command.add_argument("--feature-map", required=True, type=Path)
        command.add_argument("--cohort", required=True)
        command.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Choose a new destination directory; existing exports are never overwritten")
    mapping = read_feature_map(args.feature_map)
    names = mapping.Label.tolist()
    manifest = {"cohort": args.cohort, "endpoint": args.endpoint, "n_exposures": len(names)}
    if args.command == "matrices":
        paths = [args.stat_matrix, args.prs_matrix, args.xy_matrix]
        separators = [args.stat_separator, args.prs_separator, args.xy_separator]
        tables = [pd.read_csv(path, sep="\t" if sep == "tab" else ",", float_precision="round_trip")
                  for path, sep in zip(paths, separators)]
        result = combined_matrix(*tables, names)
        filename = "matrix.csv.gz"
        manifest.update({"n_rows": len(result), "n_columns": len(result.columns),
                         "outcome_counts": {str(k): int(v) for k, v in result.Y.value_counts().items()},
                         "row_id": "source_row is a one-based row position, NOT a participant identifier",
                         "validation": "Finite matrices; exact X alignment across all three source tables; binary Y",
                         "scope": "Matrix export only; no model fit, effect estimation or final-run provenance inferred"})
    else:
        paths = [args.stat_scores, args.prs_scores]
        result = paired_effect_table(*paths, names, stat_run_index=args.stat_run_index,
                                     prs_run_index=args.prs_run_index)
        result.insert(1, "feature_name", mapping.Feature.to_numpy())
        filename = "effects.csv"
        manifest.update({"score_definition": "dnn_squared_path_importance", "protocol_id": args.protocol_id,
                         "stat_run_index": args.stat_run_index if args.stat_run_index is not None else 0,
                         "prs_run_index": args.prs_run_index if args.prs_run_index is not None else 0,
                         "scope": "All features from caller-paired blocked featImport outputs; no selection filtering",
                         "pairing": "Caller must verify matching cohort, outcome, preprocessing and split"})
    paths.append(args.feature_map)
    manifest["sources"] = [{"filename": Path(path).name, "sha256": digest(path)} for path in paths]
    # All input validation completes before any output directory is created.
    args.destination.mkdir(parents=True)
    result.to_csv(args.destination / filename, index=False, compression="infer")
    mapping.to_csv(args.destination / "feature_dictionary.csv", index=False)
    manifest["output"] = {"filename": filename, "sha256": digest(args.destination / filename)}
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"destination": str(args.destination), "rows": len(result), "file": filename}))


if __name__ == "__main__":
    main()
