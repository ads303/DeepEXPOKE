"""Export all exposure/control scores without losing feature pairing.

The DNN callback writes blocked rows: X1, X2, ..., C1, C2, ... .
Importance scores and signed weight products are different estimands. These
helpers read featImport scores; they do not estimate causal coefficients.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def read_pair_scores(path, feature_ids, run_index=None, layout="blocked"):
    """Read one saved fit, requiring an explicit index for appended fits."""
    names = list(feature_ids)
    if not names or len(set(names)) != len(names) or any(pd.isna(x) for x in names):
        raise ValueError("feature_ids must be nonempty, unique and nonmissing")
    values = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if values.shape[1] != 2 * len(names) or not np.isfinite(values).all():
        raise ValueError("Expected 2*p finite scores per saved fit")
    if run_index is None:
        if len(values) != 1:
            raise ValueError("Multiple saved fits: specify run_index explicitly")
        run_index = 0
    if isinstance(run_index, bool) or not isinstance(run_index, (int, np.integer)) or not 0 <= run_index < len(values):
        raise ValueError("run_index is outside the saved fits")
    row = values[run_index]
    if layout == "interleaved":
        pair = row.reshape(len(names), 2)
    elif layout == "blocked":
        pair = row.reshape(2, len(names)).T
    else:
        raise ValueError("layout must be interleaved or blocked")
    return pd.DataFrame({
        "feature_id": names,
        "exposure_importance": pair[:, 0],
        "control_importance": pair[:, 1],
        "W": np.abs(pair[:, 0]) - np.abs(pair[:, 1]),
    })


def paired_effect_table(stat_path, prs_path, feature_ids, *,
                        stat_run_index=None, prs_run_index=None):
    """Retain both exposure estimates from the two separately fitted models.

The caller must supply the original training feature order and pair fits from
the same outcome, sample definition, split and preprocessing protocol.
"""
    names = list(feature_ids)
    stat = read_pair_scores(stat_path, names, stat_run_index)
    prs = read_pair_scores(prs_path, names, prs_run_index)
    return pd.DataFrame({
        "feature_id": names,
        "exposure_stat": stat.exposure_importance,
        "stat_control": stat.control_importance,
        "exposure_prs": prs.exposure_importance,
        "prs_control": prs.control_importance,
        "W_stat": stat.W,
        "W_prs": prs.W,
    })


def split_aligned_matrices(stat, prs, feature_ids):
    """Split [X, statKO] and [X, prsCO] only when X matches cell for cell.

This validates positional consistency, not participant identity. Preserve the
cohort's original participant manifest separately when it is available.
"""
    names = list(feature_ids)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_ids must be nonempty and unique")
    a, b = np.asarray(stat, dtype=float), np.asarray(prs, dtype=float)
    p = len(names)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 2 * p or a.shape[0] == 0:
        raise ValueError("Expected matching nonempty n by 2*p input matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing or nonfinite matrix values")
    if not np.array_equal(a[:, :p], b[:, :p]):
        raise ValueError("Exposure blocks differ: recover participant IDs/order before joining")
    return tuple(pd.DataFrame(v, columns=names) for v in (a[:, :p], a[:, p:], b[:, p:]))
