"""Supervised classification of exposure-associated signal patterns.

Train on independent truth labels (e.g. simulation mechanisms), using the same
score definition as at prediction. Probabilities describe learned signal
patterns, not causal probabilities. No pretrained UKBB model is supplied.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("genetics-driven", "non-genetics-driven", "correlation-driven")
SCHEMAS = {
    "joint": ("exposure", "prs_control", "stat_control"),
    "paired": ("exposure_stat", "stat_control", "exposure_prs", "prs_control"),
}


def design_matrix(frame, schema):
    """Keep raw scores and add magnitude contrasts in their original units."""
    if schema not in SCHEMAS:
        raise ValueError("schema must be joint or paired")
    columns = list(SCHEMAS[schema])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("Missing score columns: " + ", ".join(sorted(missing)))
    x = frame[columns].apply(pd.to_numeric, errors="raise").copy()
    if x.empty or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Scores must be nonempty and finite; missing scores are not zero")
    stat_x = "exposure" if schema == "joint" else "exposure_stat"
    prs_x = "exposure" if schema == "joint" else "exposure_prs"
    x["W_stat"] = x[stat_x].abs() - x.stat_control.abs()
    x["W_prs"] = x[prs_x].abs() - x.prs_control.abs()
    return x


class ArchitectureClassifier:
    """A standardized multinomial logistic model with grouped validation."""

    def __init__(self, schema="paired", score_definition="dnn_squared_path_importance",
                 protocol_id=None, random_state=42):
        if schema not in SCHEMAS:
            raise ValueError("Unknown score schema")
        if not score_definition or not protocol_id:
            raise ValueError("Specify score_definition and a matching preprocessing/model protocol_id")
        self.schema = schema
        self.score_definition = score_definition
        self.protocol_id = protocol_id
        self.random_state = random_state

    def _new_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=self.random_state))

    def fit(self, frame, labels, groups, n_splits=5):
        """Validate by independent simulation population before fitting all rows.

        All outcomes/exposures/reference variants from one simulated population
        must share a group. This does not validate transfer to UKBB.
        """
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
        from sklearn.model_selection import StratifiedGroupKFold
        x = design_matrix(frame, self.schema)
        y, g = np.asarray(labels), np.asarray(groups)
        if y.ndim != 1 or g.ndim != 1 or len(y) != len(x) or len(g) != len(x):
            raise ValueError("Labels/groups must be one-dimensional and aligned to score rows")
        if pd.isna(y).any() or pd.isna(g).any() or set(y) != set(CLASSES):
            raise ValueError("Require nonmissing groups and all three exact class labels")
        if not isinstance(n_splits, int) or n_splits < 2 or len(set(g)) < n_splits:
            raise ValueError("Need at least n_splits independent groups and n_splits >= 2")
        for label in CLASSES:
            if len(set(g[y == label])) < n_splits:
                raise ValueError("Every class must occur in at least n_splits independent groups")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        probabilities = np.full((len(x), len(CLASSES)), np.nan)
        folds = np.full(len(x), -1, dtype=int)
        for fold, (train, test) in enumerate(cv.split(x, y, g)):
            if set(y[train]) != set(CLASSES):
                raise ValueError("Training fold is missing a class; revise independent groups")
            model = self._new_model().fit(x.iloc[train], y[train])
            order = [list(model.classes_).index(label) for label in CLASSES]
            probabilities[test] = model.predict_proba(x.iloc[test])[:, order]
            folds[test] = fold
        predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
        # sklearn's log_loss uses lexicographically ordered labels.
        sorted_order = [CLASSES.index(label) for label in sorted(CLASSES)]
        self.validation_ = {
            "n_rows": len(x), "n_groups": len(set(g)), "n_splits": n_splits,
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "log_loss": float(log_loss(y, probabilities[:, sorted_order], labels=sorted(CLASSES))),
            "class_order": list(CLASSES),
            "confusion_matrix": confusion_matrix(y, predicted, labels=list(CLASSES)).tolist(),
            "scope": "Grouped validation on supplied labels only; UKBB transfer is unvalidated",
        }
        self.oof_ = pd.DataFrame({"group": g, "fold": folds, "truth": y, "prediction": predicted})
        for i, label in enumerate(CLASSES):
            self.oof_["p_" + label] = probabilities[:, i]
        self.model_ = self._new_model().fit(x, y)
        return self

    def predict(self, frame, *, score_definition, protocol_id, min_probability=0.0):
        if not hasattr(self, "model_"):
            raise ValueError("Fit the classifier before prediction")
        if score_definition != self.score_definition or protocol_id != self.protocol_id:
            raise ValueError("Prediction score definition/protocol must match training")
        if not 0 <= min_probability <= 1:
            raise ValueError("min_probability must be between 0 and 1")
        x = design_matrix(frame, self.schema)
        order = [list(self.model_.classes_).index(label) for label in CLASSES]
        probabilities = self.model_.predict_proba(x)[:, order]
        result = frame.copy()
        for i, label in enumerate(CLASSES):
            result["p_" + label] = probabilities[:, i]
        result["predicted_class"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
        result["max_probability"] = probabilities.max(axis=1)
        result["assignment"] = result.predicted_class.where(
            result.max_probability >= min_probability, "unresolved")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("scores_csv")
    train.add_argument("model_file")
    train.add_argument("--schema", choices=SCHEMAS, default="paired")
    train.add_argument("--label-column", default="truth_class")
    train.add_argument("--group-column", default="population_id")
    train.add_argument("--folds", type=int, default=5)
    predict = sub.add_parser("predict")
    predict.add_argument("model_file")
    predict.add_argument("scores_csv")
    predict.add_argument("output_csv")
    predict.add_argument("--min-probability", type=float, default=0.0)
    for cmd in (train, predict):
        cmd.add_argument("--score-definition", required=True)
        cmd.add_argument("--protocol-id", required=True)
    args = parser.parse_args()
    import joblib
    frame = pd.read_csv(args.scores_csv)
    if args.command == "train":
        model = ArchitectureClassifier(args.schema, args.score_definition, args.protocol_id)
        model.fit(frame, frame[args.label_column], frame[args.group_column], args.folds)
        joblib.dump(model, args.model_file)
        Path(args.model_file + ".validation.json").write_text(json.dumps(model.validation_, indent=2))
        model.oof_.to_csv(args.model_file + ".oof.csv", index=False)
    else:
        # Load only artifacts created by a trusted training run: joblib uses pickle.
        model = joblib.load(args.model_file)
        model.predict(frame, score_definition=args.score_definition, protocol_id=args.protocol_id,
                      min_probability=args.min_probability).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    # Persist the importable class path when launched via python -m.
    from DL.architecture import main as imported_main
    imported_main()
