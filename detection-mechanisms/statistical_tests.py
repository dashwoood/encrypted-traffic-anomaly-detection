"""Statistical significance tests for comparing detection methods.

Provides Kruskal-Wallis (3-group), paired Wilcoxon signed-rank (post-hoc
with Bonferroni), and Mann-Whitney U tests for comparing DL vs classical ML
vs statistical/heuristic model performance, as required by the thesis
methodology (inferential statistics).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

LOG = logging.getLogger(__name__)

DEEP_LEARNING = {"autoencoder", "cnn", "lstm", "gru", "transformer"}
CLASSICAL_ML = {"isolation_forest", "pca", "kmeans"}
STATISTICAL = {"baseline", "threshold", "ensemble"}

DL_CV = ["autoencoder", "cnn", "lstm", "gru", "transformer"]
DL_SUPERVISED_CV = ["cnn", "lstm", "gru", "transformer"]
CLASSICAL_ML_CV = ["isolation_forest", "pca", "kmeans"]
STATISTICAL_CV = ["threshold", "ensemble"]


def _load_results(results_dir: Path) -> list[dict]:
    results = []
    for p in sorted(Path(results_dir).glob("experiment_*.json")):
        if p.name == "experiment_summary.json":
            continue
        with open(p) as f:
            data = json.load(f)
        if "error" not in data:
            results.append(data)
    return results


def _load_cv_f1(results_dir: Path, model_name: str) -> list[float] | None:
    """Load per-fold F1 scores for a model from cv_<model>.json, if present."""
    cv_path = Path(results_dir) / f"cv_{model_name}.json"
    if not cv_path.exists():
        return None
    try:
        with open(cv_path) as f:
            data = json.load(f)
    except Exception:
        LOG.warning("Failed to load CV file for model %s", model_name)
        return None

    folds = data.get("folds", [])
    scores: list[float] = []
    for fold in folds:
        if isinstance(fold, dict) and "f1" in fold:
            try:
                scores.append(float(fold["f1"]))
            except (TypeError, ValueError):
                continue
    return scores or None


def _mean_f1_per_fold_across_models(
    results_dir: Path,
    model_names: list[str],
) -> list[float] | None:
    """Same-length per-fold F1 lists required; returns mean F1 at each fold index."""
    series: list[list[float]] = []
    for m in model_names:
        s = _load_cv_f1(results_dir, m)
        if s is None:
            return None
        series.append(s)
    k = len(series[0])
    if k < 2 or any(len(s) != k for s in series):
        return None
    return [float(np.mean([s[i] for s in series])) for i in range(k)]


def _wilcoxon_paired_folds(
    a: list[float],
    b: list[float],
    *,
    alternative: str = "greater",
) -> dict:
    """Paired Wilcoxon on fold-level means; a vs b."""
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    if len(aa) != len(bb) or len(aa) < 2:
        return {"note": "insufficient paired folds"}
    diff = aa - bb
    if np.all(diff == 0):
        return {"note": "zero difference on all folds"}
    try:
        w_stat, p_val = stats.wilcoxon(aa, bb, alternative=alternative)
        return {
            "wilcoxon_stat": round(float(w_stat), 4),
            "p_value": round(float(p_val), 6),
            "n_folds": int(len(aa)),
            "mean_a": round(float(aa.mean()), 4),
            "mean_b": round(float(bb.mean()), 4),
        }
    except ValueError as e:
        return {"note": str(e)}


def compare_two_models(metrics_a: list[float], metrics_b: list[float]) -> dict:
    """Run paired t-test and Wilcoxon signed-rank on two matched metric lists."""
    a = np.array(metrics_a, dtype=float)
    b = np.array(metrics_b, dtype=float)
    diff = a - b

    out: dict = {
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4),
    }

    if len(a) >= 2 and not np.all(diff == 0):
        t_stat, t_pval = stats.ttest_rel(a, b)
        out["paired_ttest"] = {"t_stat": round(float(t_stat), 4), "p_value": round(float(t_pval), 6)}

        try:
            w_stat, w_pval = stats.wilcoxon(a, b)
            out["wilcoxon"] = {"w_stat": round(float(w_stat), 4), "p_value": round(float(w_pval), 6)}
        except ValueError:
            out["wilcoxon"] = {"note": "not enough variation for Wilcoxon test"}
    else:
        out["paired_ttest"] = {"note": "insufficient data for paired test"}

    if len(a) >= 2:
        pooled_std = np.sqrt((a.var() + b.var()) / 2)
        if pooled_std > 0:
            out["cohens_d"] = round(float(diff.mean() / pooled_std), 4)
        else:
            out["cohens_d"] = 0.0

    return out


def compare_groups(
    ai_metrics: list[float],
    traditional_metrics: list[float],
) -> dict:
    """Mann-Whitney U test for independent group comparison."""
    a = np.array(ai_metrics, dtype=float)
    b = np.array(traditional_metrics, dtype=float)

    out: dict = {
        "ai_mean": round(float(a.mean()), 4),
        "traditional_mean": round(float(b.mean()), 4),
    }

    if len(a) >= 1 and len(b) >= 1:
        try:
            u_stat, u_pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            out["mann_whitney_u"] = {
                "u_stat": round(float(u_stat), 4),
                "p_value": round(float(u_pval), 6),
            }
        except ValueError:
            out["mann_whitney_u"] = {"note": "insufficient variation"}

    return out


def kruskal_three_groups(results_dir: Path) -> dict:
    """Kruskal-Wallis test across DL / classical ML / statistical groups.

    Uses per-fold mean F1 from cross-validation files. Post-hoc pairwise
    Wilcoxon tests with Bonferroni correction (3 comparisons).
    """
    dl_folds = _mean_f1_per_fold_across_models(results_dir, DL_CV)
    ml_folds = _mean_f1_per_fold_across_models(results_dir, CLASSICAL_ML_CV)
    stat_folds = _mean_f1_per_fold_across_models(results_dir, STATISTICAL_CV)

    if not all([dl_folds, ml_folds, stat_folds]):
        return {"note": "CV fold data unavailable for one or more groups."}

    k = len(dl_folds)
    if len(ml_folds) != k or len(stat_folds) != k:
        return {"note": "Unequal fold counts across groups."}

    out: dict = {
        "groups": {
            "deep_learning": {"models": DL_CV, "fold_means": [round(x, 4) for x in dl_folds]},
            "classical_ml": {"models": CLASSICAL_ML_CV, "fold_means": [round(x, 4) for x in ml_folds]},
            "statistical": {"models": STATISTICAL_CV, "fold_means": [round(x, 4) for x in stat_folds]},
        },
        "n_folds": k,
    }

    try:
        h_stat, kw_p = stats.kruskal(dl_folds, ml_folds, stat_folds)
        out["kruskal_wallis"] = {
            "H_statistic": round(float(h_stat), 4),
            "p_value": round(float(kw_p), 6),
            "reject_h0_at_005": bool(kw_p < 0.05),
        }
    except ValueError as e:
        out["kruskal_wallis"] = {"note": str(e)}
        return out

    n_comparisons = 3
    pairs = [
        ("DL_vs_classical_ML", dl_folds, ml_folds),
        ("DL_vs_statistical", dl_folds, stat_folds),
        ("classical_ML_vs_statistical", ml_folds, stat_folds),
    ]
    posthoc: dict = {}
    for name, a, b in pairs:
        w = _wilcoxon_paired_folds(a, b, alternative="greater")
        raw_p = w.get("p_value")
        if raw_p is not None:
            corrected = min(raw_p * n_comparisons, 1.0)
            w["p_value_bonferroni"] = round(corrected, 6)
            w["reject_h0_at_005_bonferroni"] = bool(corrected < 0.05)
        posthoc[name] = w
    out["posthoc_wilcoxon_bonferroni"] = posthoc

    dl_sup = _mean_f1_per_fold_across_models(results_dir, DL_SUPERVISED_CV)
    if dl_sup and len(dl_sup) == k:
        sup_pairs = [
            ("DL_supervised_vs_classical_ML", dl_sup, ml_folds),
            ("DL_supervised_vs_statistical", dl_sup, stat_folds),
        ]
        sup_posthoc: dict = {}
        for name, a, b in sup_pairs:
            w = _wilcoxon_paired_folds(a, b, alternative="greater")
            raw_p = w.get("p_value")
            if raw_p is not None:
                corrected = min(raw_p * 2, 1.0)
                w["p_value_bonferroni"] = round(corrected, 6)
                w["reject_h0_at_005_bonferroni"] = bool(corrected < 0.05)
            sup_posthoc[name] = w
        out["supervised_only_posthoc"] = sup_posthoc

    return out


def run_comparison(results_dir: Path) -> dict:
    """Load experiment results and run statistical comparisons.

    Returns a dict with:
      - per_model: metrics per model
      - three_group_comparison: Kruskal-Wallis + post-hoc Wilcoxon (3 groups)
      - legacy ai_vs_traditional: group comparison on F1 (Mann-Whitney, kept for continuity)
      - hypothesis_test: primary paired Wilcoxon on fold-mean F1 (DL vs rest)
      - hypothesis_test_supervised_only: Wilcoxon without autoencoder in DL group
      - mann_whitney_model_level: legacy one-sided test on single F1 per model
    """
    # Legacy sets for backward compatibility
    TRADITIONAL = {"baseline", "isolation_forest", "pca", "kmeans", "threshold", "ensemble"}
    AI = DEEP_LEARNING
    TRAD_FOR_CV = ["isolation_forest", "pca", "kmeans", "threshold", "ensemble"]
    AI_ALL_CV = DL_CV

    results = _load_results(results_dir)
    if not results:
        return {"error": "No results found"}

    per_model = {r["model"]: r for r in results}
    ai_f1 = [r["f1"] for r in results if r["model"] in AI]
    trad_f1 = [r["f1"] for r in results if r["model"] in TRADITIONAL and r["model"] != "baseline"]

    out: dict = {"per_model": per_model}

    out["three_group_comparison"] = kruskal_three_groups(results_dir)

    if ai_f1 and trad_f1:
        out["ai_vs_traditional"] = compare_groups(ai_f1, trad_f1)

        ai_f1_arr = np.array(ai_f1)
        trad_f1_arr = np.array(trad_f1)
        try:
            u_stat, p_val = stats.mannwhitneyu(ai_f1_arr, trad_f1_arr, alternative="greater")
            out["mann_whitney_model_level"] = {
                "description": (
                    "One-sided Mann-Whitney U on single F1 score per model (not paired by fold).\n"
                    "H0: DL models do not achieve higher F1 than rest.\n"
                    "H1: DL models achieve higher F1."
                ),
                "mann_whitney_u": round(float(u_stat), 4),
                "p_value": round(float(p_val), 6),
                "dl_mean_f1": round(float(ai_f1_arr.mean()), 4),
                "rest_mean_f1": round(float(trad_f1_arr.mean()), 4),
                "reject_h0_at_005": bool(p_val < 0.05),
            }
        except ValueError:
            out["mann_whitney_model_level"] = {"note": "insufficient data"}

        trad_fold_means = _mean_f1_per_fold_across_models(results_dir, TRAD_FOR_CV)
        ai_fold_means = _mean_f1_per_fold_across_models(results_dir, AI_ALL_CV)
        if ai_fold_means and trad_fold_means and len(ai_fold_means) == len(trad_fold_means):
            w_primary = _wilcoxon_paired_folds(ai_fold_means, trad_fold_means, alternative="greater")
            out["hypothesis_test"] = {
                "description": (
                    "Primary: one-sided Wilcoxon signed-rank test on paired fold means.\n"
                    "For each fold k, mean F1 over DL group vs "
                    "mean F1 over classical ML + statistical (without baseline).\n"
                    "H0: mean DL fold-F1 <= mean rest fold-F1.\n"
                    "H1: mean DL fold-F1 > mean rest fold-F1."
                ),
                "dl_models": AI_ALL_CV,
                "rest_models": TRAD_FOR_CV,
                "fold_mean_f1_dl": [round(x, 4) for x in ai_fold_means],
                "fold_mean_f1_rest": [round(x, 4) for x in trad_fold_means],
                "test": w_primary,
                "reject_h0_at_005": bool(
                    w_primary.get("p_value", 1.0) < 0.05
                    if "p_value" in w_primary
                    else False
                ),
            }
        else:
            out["hypothesis_test"] = {
                "note": (
                    "Paired fold means unavailable: run experiments with --cv-folds K>1 so "
                    "cv_<model>.json exists for all models in both groups."
                )
            }

        sup_ai = _mean_f1_per_fold_across_models(results_dir, DL_SUPERVISED_CV)
        if (
            sup_ai
            and trad_fold_means
            and len(sup_ai) == len(trad_fold_means)
        ):
            w_sup = _wilcoxon_paired_folds(sup_ai, trad_fold_means, alternative="greater")
            out["hypothesis_test_supervised_only"] = {
                "description": (
                    "Secondary: same as primary but DL group excludes autoencoder "
                    "(supervised deep learning: cnn, lstm, gru, transformer).\n"
                    "Autoencoder is trained unsupervised; its F1 is closer to classical ML methods."
                ),
                "dl_supervised_models": DL_SUPERVISED_CV,
                "rest_models": TRAD_FOR_CV,
                "fold_mean_f1_dl_supervised": [round(x, 4) for x in sup_ai],
                "fold_mean_f1_rest": [round(x, 4) for x in trad_fold_means],
                "test": w_sup,
                "reject_h0_at_005": bool(
                    w_sup.get("p_value", 1.0) < 0.05 if "p_value" in w_sup else False
                ),
            }

        dl_models = [r["model"] for r in results if r["model"] in AI]
        rest_models = [r["model"] for r in results if r["model"] in TRADITIONAL and r["model"] != "baseline"]

        paired: dict = {}
        if dl_models and rest_models:
            best_dl = max(dl_models, key=lambda m: per_model[m]["f1"])
            best_rest = max(rest_models, key=lambda m: per_model[m]["f1"])

            dl_cv_f1 = _load_cv_f1(results_dir, best_dl)
            rest_cv_f1 = _load_cv_f1(results_dir, best_rest)

            if dl_cv_f1 and rest_cv_f1 and len(dl_cv_f1) == len(rest_cv_f1) and len(dl_cv_f1) >= 2:
                paired["best_dl_vs_best_rest"] = {
                    "models": {"dl": best_dl, "rest": best_rest},
                    "metric": "f1",
                    "cv_f1_dl": dl_cv_f1,
                    "cv_f1_rest": rest_cv_f1,
                    "tests": compare_two_models(dl_cv_f1, rest_cv_f1),
                }
            else:
                paired["note"] = (
                    "Cross-validation files (cv_<model>.json) missing or insufficient for paired tests."
                )

        if paired:
            out["paired_tests"] = paired

    return out


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results")
    comparison = run_comparison(results_dir)
    print(json.dumps(comparison, indent=2, default=str))
