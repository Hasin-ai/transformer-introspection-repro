"""Statistics and results tables.

Deliberately small: three estimators, all non-parametric, all reported with
uncertainty. With N = 30 concepts there is no case for anything fancier, and a
parametric test would be leaning on distributional assumptions that a logit
difference does not obviously satisfy.

* ``bootstrap_mean``  — percentile bootstrap CI on a condition's mean.
* ``paired_permutation_test`` — the confirmatory test. Items are *paired by
  concept*: the same concept appears in the injected and control arms, so the
  permutation shuffles the condition label within each pair rather than across
  the whole sample. This controls for the large item-to-item variance that
  concepts induce.
* ``cohens_dz`` — paired effect size, so results are comparable across measures.

Multiple comparisons
--------------------
``holm_bonferroni`` is provided and is applied to the layer sweep, which tests
one hypothesis per layer. The confirmatory test at the pre-registered layer is
a single test and is not corrected.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def bootstrap_mean(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(arr.size),
    }


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size: mean difference over the SD of the differences."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("inf")


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Sign-flipping permutation test on paired differences.

    Under the null that the condition label carries no information, the sign of
    each paired difference is exchangeable. Flipping signs at random gives the
    null distribution of the mean difference.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"observed": float("nan"), "p_value": float("nan"), "n": 0}

    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (signs * diff).mean(axis=1)

    if alternative == "two-sided":
        p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    elif alternative == "greater":
        p = float((null >= observed - 1e-12).mean())
    else:
        p = float((null <= observed + 1e-12).mean())
    # Add-one correction: a permutation p-value is never exactly zero.
    p = (p * n_perm + 1) / (n_perm + 1)
    return {"observed": observed, "p_value": p, "n": int(diff.size)}


def binomial_test_vs_chance(n_correct: int, n_total: int, chance: float) -> dict[str, float]:
    """Exact one-sided binomial test for forced-choice accuracy above chance."""
    from scipy import stats

    if n_total == 0:
        return {"accuracy": float("nan"), "p_value": float("nan"), "n": 0}
    result = stats.binomtest(n_correct, n_total, chance, alternative="greater")
    return {
        "accuracy": n_correct / n_total,
        "chance": chance,
        "p_value": float(result.pvalue),
        "n": n_total,
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns a reject/accept mask."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject.tolist()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def detection_table(
    df: pd.DataFrame,
    reference: str = "random_control",
    measure: str = "yes_minus_no",
    n_boot: int = 5000,
    n_perm: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-condition summary of the detection measure, with paired tests.

    Expects one row per (condition, concept). Conditions are paired on
    ``concept``; rows without a matching concept in the reference condition are
    dropped from the test (but still counted in the descriptive statistics).
    """
    # Only the detection question. The yes-bias control asks a *different*
    # question, so pairing it against this reference would confound question
    # identity with injection; it gets its own table below.
    detection = df[df["question_kind"] == "detection"]
    ref = detection[detection["condition"] == reference].set_index("concept")[measure]

    rows: list[dict[str, Any]] = []
    for condition, group in detection.groupby("condition"):
        values = group[measure].to_numpy()
        summary = bootstrap_mean(values, n_boot=n_boot, alpha=alpha, seed=seed)

        row: dict[str, Any] = {
            "condition": condition,
            "N": summary["n"],
            "mean": summary["mean"],
            "std": summary["std"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "frac_yes": float((values > 0).mean()) if values.size else float("nan"),
        }

        if condition == reference:
            row.update({"delta_vs_ref": 0.0, "dz": 0.0, "p_value": float("nan")})
        else:
            paired = group.set_index("concept")[measure]
            shared = paired.index.intersection(ref.index)
            a, b = paired.loc[shared].to_numpy(), ref.loc[shared].to_numpy()
            test = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)
            row.update(
                {
                    "delta_vs_ref": test["observed"],
                    "dz": cohens_dz(a, b),
                    "p_value": test["p_value"],
                }
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def yesbias_table(
    injected: pd.DataFrame,
    baseline: pd.DataFrame,
    measure: str = "yes_minus_no",
    n_perm: int = 10000,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Does injecting the concept shift an *unrelated* no-answer question?

    The comparison is within question type and paired by item: the same
    yes-bias question, with and without the concept injected. Comparing the
    yes-bias question against the detection question instead would confound
    "which question was asked" with "was anything injected", which is why this
    needs its own table rather than a row in ``detection_table``.

    A large positive shift here is the failure mode the control exists to catch:
    the injection making the model generally more willing to say "Yes".
    """
    inj = injected[injected["question_kind"] == "yesbias"].set_index("concept")[measure]
    base = baseline[baseline["question_kind"] == "yesbias"].set_index("concept")[measure]
    shared = inj.index.intersection(base.index)
    a, b = inj.loc[shared].to_numpy(), base.loc[shared].to_numpy()

    stats_inj = bootstrap_mean(a, n_boot=n_boot, seed=seed)
    stats_base = bootstrap_mean(b, n_boot=n_boot, seed=seed)
    test = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)
    return pd.DataFrame(
        [
            {
                "comparison": "yes-bias question: injected vs no injection",
                "N": test["n"],
                "mean_injected": stats_inj["mean"],
                "mean_no_injection": stats_base["mean"],
                "shift": test["observed"],
                "dz": cohens_dz(a, b),
                "p_value": test["p_value"],
            }
        ]
    )


def identification_table(df: pd.DataFrame, seed: int = 0, n_boot: int = 5000) -> pd.DataFrame:
    """Per-condition forced-choice accuracy with a binomial test against chance."""
    ident = df[df["question_kind"] == "identification"]
    rows: list[dict[str, Any]] = []
    for condition, group in ident.groupby("condition"):
        n_total = len(group)
        n_correct = int(group["id_correct"].sum())
        chance = float(group["chance"].iloc[0]) if n_total else float("nan")
        test = binomial_test_vs_chance(n_correct, n_total, chance)
        margin = bootstrap_mean(group["id_margin"].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append(
            {
                "condition": condition,
                "N": n_total,
                "accuracy": test["accuracy"],
                "chance": chance,
                "p_vs_chance": test["p_value"],
                "mean_rank": float(group["id_rank"].mean()) if n_total else float("nan"),
                "mean_margin": margin["mean"],
                "margin_ci_low": margin["ci_low"],
                "margin_ci_high": margin["ci_high"],
            }
        )
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def layer_sweep_table(
    df: pd.DataFrame,
    reference: str = "random_control",
    measure: str = "yes_minus_no",
    n_perm: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Injected-vs-control difference at every layer, Holm-corrected.

    This is the exploratory analysis. The correction is what stops "the effect
    was significant at layer 17 out of 24 layers tested" from meaning anything
    it should not.
    """
    detection = df[df["question_kind"] == "detection"]
    rows: list[dict[str, Any]] = []
    for layer, group in detection.groupby("layer"):
        inj = group[group["condition"] == "injected"].set_index("concept")[measure]
        ref = group[group["condition"] == reference].set_index("concept")[measure]
        shared = inj.index.intersection(ref.index)
        a, b = inj.loc[shared].to_numpy(), ref.loc[shared].to_numpy()
        test = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)
        rows.append(
            {
                "layer": int(layer),
                "N": test["n"],
                "mean_injected": float(a.mean()) if a.size else float("nan"),
                "mean_control": float(b.mean()) if b.size else float("nan"),
                "delta": test["observed"],
                "dz": cohens_dz(a, b),
                "p_value": test["p_value"],
            }
        )
    table = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    if len(table):
        table["significant_holm"] = holm_bonferroni(table["p_value"].tolist(), alpha=alpha)
    return table


def identification_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Forced-choice accuracy per layer per condition."""
    ident = df[df["question_kind"] == "identification"]
    if ident.empty:
        return pd.DataFrame(columns=["layer", "condition", "accuracy", "chance", "N"])
    grouped = (
        ident.groupby(["layer", "condition"])
        .agg(accuracy=("id_correct", "mean"), chance=("chance", "first"), N=("id_correct", "size"))
        .reset_index()
    )
    return grouped


def format_markdown(table: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Render a table as GitHub-flavoured markdown."""
    display = table.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "—")
    return display.to_markdown(index=False)
