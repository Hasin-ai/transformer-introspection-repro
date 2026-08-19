"""Figures.

Design rules followed here
--------------------------
* One y-axis per panel. Where two measures with different units belong in the
  same story (logit lens), they are stacked as small multiples rather than
  forced onto a twin axis.
* Categorical colour is assigned to a *condition* and never re-assigned, so the
  same condition is the same colour in every figure. The four-slot palette below
  is validated for colour-vision deficiency (worst adjacent CVD ΔE 9.1).
* Identity is never carried by colour alone: every multi-series figure has a
  legend, and bars carry direct value labels.
* The signed intervention heatmap uses a diverging blue-to-red ramp with a
  neutral grey midpoint, because zero is a meaningful centre there. The
  unsigned figures use a single-hue sequential ramp.
* Grid and axes are recessive; the data is the darkest thing on the page.

Every figure gets a title, axis labels, and a written interpretation, which is
collected into ``results/figures/CAPTIONS.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")                      # headless-safe; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .metrics import bootstrap_mean

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dc"

CONDITION_COLOR: dict[str, str] = {
    "injected": "#2a78d6",          # slot 1, blue
    "random_control": "#eb6834",    # slot 2, orange
    "no_injection": "#1baf7a",      # slot 3, aqua
    "yesbias_control": "#eda100",   # slot 4, yellow
}
CONDITION_LABEL: dict[str, str] = {
    "injected": "Injected concept",
    "random_control": "Random vector (norm-matched)",
    "no_injection": "No injection",
    "yesbias_control": "Yes-bias control question",
}
CONDITION_ORDER = ["injected", "random_control", "no_injection", "yesbias_control"]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2321"]
)

CAPTIONS: list[tuple[str, str]] = []


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)


def _save(fig: plt.Figure, path: Path, caption: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    CAPTIONS.append((path.name, caption))
    return path


def write_captions(figures_dir: str | Path) -> Path:
    """Write every figure's interpretation to CAPTIONS.md."""
    path = Path(figures_dir) / "CAPTIONS.md"
    lines = ["# Figure interpretations", ""]
    for name, caption in CAPTIONS:
        lines += [f"## `{name}`", "", caption.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Detection by condition
# ---------------------------------------------------------------------------
def plot_detection_by_condition(
    df: pd.DataFrame,
    out_dir: str | Path,
    measure: str = "yes_minus_no",
    seed: int = 0,
) -> Path:
    """Grouped bar: mean detection signal per condition with bootstrap CIs."""
    detection = df[df["question_kind"].isin(["detection", "yesbias"])]
    conditions = [c for c in CONDITION_ORDER if c in set(detection["condition"])]

    means, los, his, ns = [], [], [], []
    for condition in conditions:
        stats = bootstrap_mean(
            detection[detection["condition"] == condition][measure].to_numpy(), seed=seed
        )
        means.append(stats["mean"])
        los.append(stats["mean"] - stats["ci_low"])
        his.append(stats["ci_high"] - stats["mean"])
        ns.append(stats["n"])

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _style(ax)
    x = np.arange(len(conditions))
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=[CONDITION_COLOR[c] for c in conditions],
        zorder=3,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.errorbar(
        x, means, yerr=[los, his], fmt="none",
        ecolor=INK_SOFT, elinewidth=1.6, capsize=5, zorder=4,
    )
    ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=2)

    # Labels sit clear of the whiskers, not on top of them.
    tops = [m + h for m, h in zip(means, his)]
    bottoms = [m - l for m, l in zip(means, los)]
    span = max(max(tops), 0) - min(min(bottoms), 0) + 1e-6
    pad = 0.05 * span
    for xi, mean in enumerate(means):
        y = tops[xi] + pad if mean >= 0 else bottoms[xi] - pad
        ax.text(
            xi, y, f"{mean:+.2f}", ha="center",
            va="bottom" if mean >= 0 else "top",
            color=INK, fontsize=10, fontweight="medium", zorder=5,
        )
    ax.set_ylim(min(min(bottoms), 0) - 4 * pad, max(max(tops), 0) + 4 * pad)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{CONDITION_LABEL[c]}\n(N={n})" for c, n in zip(conditions, ns)], fontsize=9)
    ax.set_ylabel("logit(Yes) − logit(No)\nat the answer position")
    ax.set_title(
        "Does the model report detecting an injected thought?",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    caption = (
        "Mean detection signal per condition, with 95% percentile-bootstrap confidence "
        "intervals over concepts. Positive values mean the model favours 'Yes'. The "
        "introspection hypothesis predicts that **Injected concept** sits clearly above "
        "**Random vector**, which is matched on perturbation magnitude and differs only "
        "in content. If the two are level, the model is responding to *being perturbed*, "
        "not to *what* was injected. **Yes-bias control** asks an unrelated question whose "
        "answer is 'No' while the same vector is injected; a rise there means the "
        "injection is making the model generally more agreeable, which would invalidate "
        "the detection measure."
    )
    return _save(fig, Path(out_dir) / "01_detection_by_condition.png", caption)


# ---------------------------------------------------------------------------
# 2. Identification accuracy
# ---------------------------------------------------------------------------
def plot_identification_accuracy(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Forced-choice accuracy per condition against chance."""
    ident = df[df["question_kind"] == "identification"]
    conditions = [c for c in CONDITION_ORDER if c in set(ident["condition"])]
    accs, ns = [], []
    for condition in conditions:
        group = ident[ident["condition"] == condition]
        accs.append(float(group["id_correct"].mean()))
        ns.append(len(group))
    chance = float(ident["chance"].iloc[0]) if len(ident) else np.nan

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _style(ax)
    x = np.arange(len(conditions))
    ax.bar(
        x, accs, width=0.62,
        color=[CONDITION_COLOR[c] for c in conditions],
        zorder=3, edgecolor=SURFACE, linewidth=2,
    )
    ax.axhline(chance, color=INK_SOFT, linewidth=1.6, linestyle="--", zorder=4)
    ax.text(
        len(conditions) - 0.45, chance, f"  chance = {chance:.2f}",
        color=INK_SOFT, fontsize=9, va="bottom", ha="right",
    )
    for xi, (acc, n) in enumerate(zip(accs, ns)):
        ax.text(xi, acc + 0.015, f"{acc:.2f}", ha="center", va="bottom",
                color=INK, fontsize=10, fontweight="medium", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{CONDITION_LABEL[c]}\n(N={n})" for c, n in zip(conditions, ns)], fontsize=9)
    ax.set_ylim(0, max(max(accs, default=0.2), chance) * 1.35)
    ax.set_ylabel("forced-choice accuracy")
    ax.set_title(
        "Can the model name the injected concept?",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    caption = (
        "Accuracy at picking the injected concept over matched distractors drawn from the "
        "same word bank, scored by length-normalised log-probability rather than by "
        "grading free text. The dashed line is chance. Detection (figure 1) without "
        "above-chance identification would say the model notices *something* but has no "
        "access to its content. Above-chance identification in the **random vector** arm "
        "would be alarming: nothing there carries concept information, so any signal would "
        "indicate a leak in the scoring procedure."
    )
    return _save(fig, Path(out_dir) / "02_identification_accuracy.png", caption)


# ---------------------------------------------------------------------------
# 3. Layer sweep
# ---------------------------------------------------------------------------
def plot_layer_sweep(
    sweep: pd.DataFrame,
    out_dir: str | Path,
    primary_layer: int | None = None,
    measure: str = "yes_minus_no",
    seed: int = 0,
) -> Path:
    """Detection signal as a function of injection depth, per condition."""
    detection = sweep[sweep["question_kind"] == "detection"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(ax)

    for condition in CONDITION_ORDER:
        group = detection[detection["condition"] == condition]
        if group.empty:
            continue
        layers, means, lows, highs = [], [], [], []
        for layer, sub in group.groupby("layer"):
            stats = bootstrap_mean(sub[measure].to_numpy(), seed=seed)
            layers.append(int(layer))
            means.append(stats["mean"])
            lows.append(stats["ci_low"])
            highs.append(stats["ci_high"])
        order = np.argsort(layers)
        layers = np.asarray(layers)[order]
        means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]
        highs = np.asarray(highs)[order]

        color = CONDITION_COLOR[condition]
        ax.fill_between(layers, lows, highs, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(layers, means, color=color, linewidth=2.0,
                label=CONDITION_LABEL[condition], zorder=3)

    ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
    if primary_layer is not None:
        ax.axvline(primary_layer, color=INK_SOFT, linewidth=1.4, linestyle=":", zorder=1)
        ax.text(
            primary_layer, ax.get_ylim()[1], " pre-registered layer",
            color=INK_SOFT, fontsize=9, va="top", ha="left",
        )

    ax.set_xlabel("injection layer (block index)")
    ax.set_ylabel("logit(Yes) − logit(No)")
    ax.set_title(
        "Detection signal by injection depth  —  exploratory",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    caption = (
        "Mean detection signal (shaded band = 95% bootstrap CI) as the injection layer "
        "moves through the network. Lindsey (2025) reports the effect peaking around "
        "two-thirds of the way through the model, which is where the dotted line sits. "
        "**This panel is exploratory**: it tests one hypothesis per layer, so the "
        "confirmatory claim rests on the pre-registered layer alone and the per-layer "
        "p-values in `layer_sweep_table` are Holm-corrected. Reading off the best layer "
        "post hoc is exactly how a null result is turned into a false positive."
    )
    return _save(fig, Path(out_dir) / "03_layer_sweep.png", caption)


# ---------------------------------------------------------------------------
# 4. Strength sweep
# ---------------------------------------------------------------------------
def plot_strength_sweep(
    sweep: pd.DataFrame,
    out_dir: str | Path,
    measure: str = "yes_minus_no",
    seed: int = 0,
) -> Path:
    """Detection signal versus injection strength, per condition."""
    detection = sweep[sweep["question_kind"] == "detection"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    _style(ax)

    for condition in CONDITION_ORDER:
        group = detection[detection["condition"] == condition]
        if group.empty:
            continue
        strengths, means, lows, highs = [], [], [], []
        for strength, sub in group.groupby("strength"):
            stats = bootstrap_mean(sub[measure].to_numpy(), seed=seed)
            strengths.append(float(strength))
            means.append(stats["mean"])
            lows.append(stats["ci_low"])
            highs.append(stats["ci_high"])
        order = np.argsort(strengths)
        s = np.asarray(strengths)[order]
        color = CONDITION_COLOR[condition]
        ax.fill_between(s, np.asarray(lows)[order], np.asarray(highs)[order],
                        color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(s, np.asarray(means)[order], color=color, linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=CONDITION_LABEL[condition], zorder=3)

    ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
    ax.set_xlabel("injection strength (residual-norm units)")
    ax.set_ylabel("logit(Yes) − logit(No)")
    ax.set_title("Detection", fontsize=11, fontweight="semibold", loc="left", pad=8)
    legend = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    ax2 = axes[1]
    _style(ax2)
    ident = sweep[sweep["question_kind"] == "identification"]
    chance = float(ident["chance"].iloc[0]) if len(ident) else np.nan
    for condition in CONDITION_ORDER:
        group = ident[ident["condition"] == condition]
        if group.empty:
            continue
        agg = group.groupby("strength")["id_correct"].mean().sort_index()
        ax2.plot(agg.index.to_numpy(), agg.to_numpy(),
                 color=CONDITION_COLOR[condition], linewidth=2.0,
                 marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label=CONDITION_LABEL[condition], zorder=3)
    if np.isfinite(chance):
        ax2.axhline(chance, color=INK_SOFT, linewidth=1.4, linestyle="--", zorder=1)
        ax2.text(ax2.get_xlim()[1], chance, "chance  ", color=INK_SOFT,
                 fontsize=9, va="bottom", ha="right")
    ax2.set_xlabel("injection strength (residual-norm units)")
    ax2.set_ylabel("forced-choice accuracy")
    ax2.set_title("Identification", fontsize=11, fontweight="semibold", loc="left", pad=8)

    fig.suptitle(
        "Effect of injection strength",
        fontsize=13, fontweight="semibold", x=0.01, ha="left", color=INK,
    )
    caption = (
        "Left: detection signal versus injection strength. Right: identification accuracy "
        "over the same range. Two shapes matter. A **monotone rise in both panels for the "
        "injected condition only** is the pattern the introspection hypothesis predicts. An "
        "**inverted U** — rising then collapsing — is the 'brain damage' regime the original "
        "work reports at high strength, where the model is overwhelmed rather than informed; "
        "results from that regime should not be interpreted. If the random control rises "
        "just as steeply, the measure is tracking perturbation size."
    )
    return _save(fig, Path(out_dir) / "04_strength_sweep.png", caption)


# ---------------------------------------------------------------------------
# 5. Logit lens
# ---------------------------------------------------------------------------
def plot_logit_lens(
    lens: pd.DataFrame,
    out_dir: str | Path,
    inject_layer: int | None = None,
) -> Path:
    """Two stacked panels: when the concept becomes readable, when Yes is decided."""
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)

    grouped = lens.groupby("read_layer")
    layers = np.asarray(sorted(lens["read_layer"].unique()))
    concept_adv = grouped["concept_logit_adv"].mean().reindex(layers).to_numpy()
    concept_sd = grouped["concept_logit_adv"].std(ddof=1).reindex(layers).to_numpy()
    yes_no = grouped["yes_minus_no"].mean().reindex(layers).to_numpy()
    yes_sd = grouped["yes_minus_no"].std(ddof=1).reindex(layers).to_numpy()

    for ax, mean, sd, color, ylabel, title in (
        (axes[0], concept_adv, concept_sd, "#2a78d6",
         "logit(concept) − mean logit(distractors)",
         "Is the injected concept readable here?"),
        (axes[1], yes_no, yes_sd, "#eb6834",
         "logit(Yes) − logit(No)",
         "Has the model committed to answering Yes?"),
    ):
        _style(ax)
        sd = np.nan_to_num(sd)
        ax.fill_between(layers, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(layers, mean, color=color, linewidth=2.0, zorder=3)
        ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
        if inject_layer is not None:
            ax.axvline(inject_layer, color=INK_SOFT, linewidth=1.4, linestyle=":", zorder=1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="semibold", loc="left", pad=8)

    if inject_layer is not None:
        axes[0].text(inject_layer, axes[0].get_ylim()[1], " injection site",
                     color=INK_SOFT, fontsize=9, va="top", ha="left")
    axes[1].set_xlabel("layer the residual stream was decoded at (logit lens)")
    fig.suptitle(
        "Logit lens through the stack, at the answer position",
        fontsize=13, fontweight="semibold", x=0.01, ha="left", color=INK,
    )
    caption = (
        "The residual stream at the final prompt position is decoded through the "
        "unembedding at every layer (shaded band = ±1 SD across concepts). Top: how far "
        "the injected concept's token outranks its distractors. Bottom: how far the model "
        "leans toward 'Yes'. The dotted line marks where the injection was applied. "
        "The introspection story requires the top curve to rise *before* the bottom one "
        "does — the concept has to be readable before a decision about it can be formed. "
        "If the concept only surfaces in the last layer or two, the injected vector is "
        "reaching the logits through the residual skip connection rather than being read "
        "and acted on, which is a mechanistic confound, not introspection."
    )
    return _save(fig, Path(out_dir) / "05_logit_lens.png", caption)


# ---------------------------------------------------------------------------
# 6. Residual similarity
# ---------------------------------------------------------------------------
def plot_residual_similarity(
    similarity: pd.DataFrame,
    out_dir: str | Path,
    inject_layer: int | None = None,
) -> Path:
    """Cosine similarity between injected and clean residual streams, by layer."""
    grouped = similarity.groupby("read_layer")["cosine_similarity"]
    layers = np.asarray(sorted(similarity["read_layer"].unique()))
    mean = grouped.mean().reindex(layers).to_numpy()
    sd = np.nan_to_num(grouped.std(ddof=1).reindex(layers).to_numpy())

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    _style(ax)
    ax.fill_between(layers, mean - sd, mean + sd, color="#1baf7a", alpha=0.16, linewidth=0, zorder=2)
    ax.plot(layers, mean, color="#1baf7a", linewidth=2.0, zorder=3)
    ax.axhline(1.0, color=INK_SOFT, linewidth=1.0, linestyle="--", zorder=1)
    if inject_layer is not None:
        ax.axvline(inject_layer, color=INK_SOFT, linewidth=1.4, linestyle=":", zorder=1)
        ax.text(inject_layer, ax.get_ylim()[0], " injection site",
                color=INK_SOFT, fontsize=9, va="bottom", ha="left")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity, injected vs clean")
    ax.set_title(
        "How far does the injection push the residual stream?",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    caption = (
        "Cosine similarity between the injected and clean residual stream at the answer "
        "position (band = ±1 SD across concepts). The dashed line at 1.0 is 'no change'. "
        "A healthy intervention drops sharply at the injection site and then partially "
        "recovers as later layers renormalise — the model has been nudged. A curve that "
        "stays near zero for the rest of the stack means the forward pass has been "
        "destroyed rather than perturbed, and no behavioural change measured under those "
        "conditions can be interpreted as introspection."
    )
    return _save(fig, Path(out_dir) / "06_residual_similarity.png", caption)


# ---------------------------------------------------------------------------
# 7. Intervention heatmap
# ---------------------------------------------------------------------------
def plot_intervention(
    ablation: pd.DataFrame,
    out_dir: str | Path,
    inject_layer: int | None = None,
) -> Path:
    """Heatmap of the ablation interaction by downstream layer and component."""
    pivot = (
        ablation.groupby(["component", "ablate_layer"])["interaction"]
        .mean()
        .unstack("ablate_layer")
        .sort_index()
    )
    values = pivot.to_numpy()
    limit = float(np.nanmax(np.abs(values))) if values.size else 1.0
    limit = limit if limit > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    _style(ax)
    ax.grid(False)
    mesh = ax.imshow(
        values,
        aspect="auto",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        ["attention output" if i == "attn_out" else "MLP output" for i in pivot.index],
        fontsize=9,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
    ax.set_xlabel("layer whose component was ablated")
    ax.set_title(
        "Which downstream components carry the injected signal?",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    bar = fig.colorbar(mesh, ax=ax, pad=0.015)
    bar.set_label("interaction (logits)", fontsize=9, color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)

    caption = (
        "Each cell is a double subtraction: how much the injection's effect on the "
        "detection readout changes when that component is knocked out, minus the same "
        "quantity measured without any injection. Subtracting the second term matters — "
        "ablating a component damages the model whether or not anything was injected, so "
        "the raw drop would be uninformative. **Blue (strongly negative)** marks a "
        "component that specifically mediates the injected signal: remove it and the "
        "injection stops working. A heatmap that is uniformly near zero says the injected "
        "vector is not being read by any particular component, which points to it "
        "influencing the logits directly through the residual stream — the mechanistic "
        "confound this analysis exists to detect."
        + (f" The injection was applied at layer {inject_layer}; only later layers are swept."
           if inject_layer is not None else "")
    )
    return _save(fig, Path(out_dir) / "07_intervention_heatmap.png", caption)


# ---------------------------------------------------------------------------
# 8. Restore sweep (optional)
# ---------------------------------------------------------------------------
def plot_restore(restore: pd.DataFrame, out_dir: str | Path, inject_layer: int | None = None) -> Path:
    """Fraction of the injection effect abolished by restoring the clean residual."""
    grouped = restore.groupby("restore_layer")["fraction_abolished"]
    layers = np.asarray(sorted(restore["restore_layer"].unique()))
    mean = grouped.mean().reindex(layers).to_numpy()
    sd = np.nan_to_num(grouped.std(ddof=1).reindex(layers).to_numpy())

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    _style(ax)
    ax.fill_between(layers, mean - sd, mean + sd, color="#2a78d6", alpha=0.16, linewidth=0, zorder=2)
    ax.plot(layers, mean, color="#2a78d6", linewidth=2.0, zorder=3)
    ax.axhline(1.0, color=INK_SOFT, linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(0.0, color=INK_SOFT, linewidth=1.0, zorder=1)
    ax.set_xlabel("layer at which the clean residual stream was restored")
    ax.set_ylabel("fraction of injection effect abolished")
    ax.set_title(
        "How far downstream does the injected signal still need to travel?",
        fontsize=13, fontweight="semibold", loc="left", pad=12,
    )
    caption = (
        "Activation patching in the restore direction: the concept is injected, then at "
        "layer *l* the residual stream is overwritten with its clean value. 1.0 means the "
        "injection's effect was completely undone; 0.0 means it survived. The layer at "
        "which the curve falls away is the point past which the injected information no "
        "longer needs the residual stream — it has already been written into whatever "
        "carries it to the answer."
        + (f" Injection site: layer {inject_layer}." if inject_layer is not None else "")
    )
    return _save(fig, Path(out_dir) / "08_restore_sweep.png", caption)


# ---------------------------------------------------------------------------
def make_all_figures(
    results: dict[str, pd.DataFrame],
    out_dir: str | Path,
    primary_layer: int | None = None,
    seed: int = 0,
) -> list[Path]:
    """Render whichever figures the available result frames support."""
    out_dir = Path(out_dir)
    paths: list[Path] = []
    primary = results.get("primary")
    if primary is not None and len(primary):
        paths.append(plot_detection_by_condition(primary, out_dir, seed=seed))
        if (primary["question_kind"] == "identification").any():
            paths.append(plot_identification_accuracy(primary, out_dir))
    for key, fn, kwargs in (
        ("layer_sweep", plot_layer_sweep, {"primary_layer": primary_layer, "seed": seed}),
        ("strength_sweep", plot_strength_sweep, {"seed": seed}),
        ("logit_lens", plot_logit_lens, {"inject_layer": primary_layer}),
        ("residual_similarity", plot_residual_similarity, {"inject_layer": primary_layer}),
        ("ablation", plot_intervention, {"inject_layer": primary_layer}),
        ("restore", plot_restore, {"inject_layer": primary_layer}),
    ):
        frame = results.get(key)
        if frame is not None and len(frame):
            paths.append(fn(frame, out_dir, **kwargs))
    write_captions(out_dir)
    return paths
