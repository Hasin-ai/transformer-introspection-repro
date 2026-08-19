"""Run the whole pipeline and write a verdict.

    python run_all.py                 # full run
    python run_all.py --quick         # small, fast configuration for a smoke test
    python run_all.py --offline       # random-init model, no downloads at all

The verdict at the end is deliberately mechanical. It compares the measured
numbers against the four outcomes defined in the README *before* the experiment
was run, and names which one the data matches. It does not decide whether the
model is introspective; it decides which of four pre-specified patterns the
numbers fit, and says what that pattern licenses you to conclude.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.interventions import ablation_sweep                       # noqa: E402
from src.introspection import (                                     # noqa: E402
    layer_sweep, logit_lens_by_layer, residual_similarity, run_trials,
    sample_transcripts, strength_sweep,
)
from src.metrics import (                                           # noqa: E402
    detection_table, identification_by_layer, identification_table, layer_sweep_table,
    yesbias_table,
)
from src.runner import (                                            # noqa: E402
    build_context, load_config, record_environment, resolve, save_rows, save_table,
)
from src.visualization import make_all_figures                      # noqa: E402


# ---------------------------------------------------------------------------
def apply_quick(config: dict) -> dict:
    """A cut-down configuration for a fast end-to-end check."""
    config["design"]["n_concepts"] = 6
    config["injection"]["strengths"] = [0.0, 2.0]
    config["injection"]["sweep_layers"] = "auto"
    config["intervention"]["n_concepts"] = 2
    config["generation"]["n_examples"] = 2
    config["generation"]["max_new_tokens"] = 16
    config["stats"]["n_bootstrap"] = 1000
    config["stats"]["n_permutations"] = 2000
    return config


def apply_offline(config: dict) -> dict:
    config["model"]["name"] = "random-tiny"
    config["device"]["force_cpu"] = True
    return config


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:                                              # noqa: BLE001
        return "unavailable"


# ---------------------------------------------------------------------------
def verdict(
    det_table: pd.DataFrame,
    ident_table: pd.DataFrame,
    yb_table: pd.DataFrame | None,
    alpha: float,
    ablation: pd.DataFrame | None,
    lens: pd.DataFrame | None,
) -> str:
    """Classify the result against the four pre-specified outcomes."""
    lines: list[str] = []

    def row(condition: str) -> pd.Series | None:
        match = det_table[det_table["condition"] == condition]
        return match.iloc[0] if len(match) else None

    injected = row("injected")

    if injected is None:
        return "No injected condition in the results; nothing to classify."

    delta = float(injected["delta_vs_ref"])
    p = float(injected["p_value"])
    dz = float(injected["dz"])
    significant = p < alpha and delta > 0

    ident_injected = ident_table[ident_table["condition"] == "injected"]
    ident_acc = float(ident_injected["accuracy"].iloc[0]) if len(ident_injected) else float("nan")
    ident_p = float(ident_injected["p_vs_chance"].iloc[0]) if len(ident_injected) else float("nan")
    ident_chance = float(ident_injected["chance"].iloc[0]) if len(ident_injected) else float("nan")
    ident_significant = ident_p < alpha

    lines.append(f"Detection: injected minus random control = {delta:+.3f} logits "
                 f"(dz = {dz:+.2f}, permutation p = {p:.4f}, alpha = {alpha}).")
    lines.append(f"Identification: forced-choice accuracy {ident_acc:.3f} vs chance "
                 f"{ident_chance:.3f} (binomial p = {ident_p:.4f}).")

    # Confound check: did the injection just make the model agreeable?
    confounded = False
    if yb_table is not None and len(yb_table):
        yb_shift = float(yb_table["shift"].iloc[0])
        yb_p = float(yb_table["p_value"].iloc[0])
        lines.append(f"Yes-bias control: injecting the concept shifted an unrelated "
                     f"no-answer question by {yb_shift:+.3f} logits "
                     f"(paired, p = {yb_p:.4f}).")
        if significant and yb_shift > 0.5 * delta:
            confounded = True

    # Mechanism check: does the concept become readable before the decision?
    ordering_ok = None
    if lens is not None and len(lens):
        peak_concept = int(lens.groupby("read_layer")["concept_logit_adv"].mean().idxmax())
        peak_yes = int(lens.groupby("read_layer")["yes_minus_no"].mean().idxmax())
        ordering_ok = peak_concept < peak_yes
        lines.append(f"Logit lens: concept peaks at layer {peak_concept}, "
                     f"'Yes' preference peaks at layer {peak_yes}.")

    mediated = None
    if ablation is not None and len(ablation):
        strongest = float(ablation["interaction"].min())
        intact = float(ablation["injection_effect_intact"].mean())
        mediated = abs(intact) > 1e-3 and strongest < -0.25 * abs(intact)
        lines.append(f"Ablation: strongest mediating interaction {strongest:+.3f} "
                     f"against a mean intact injection effect of {intact:+.3f}.")

    lines.append("")
    if confounded:
        lines.append("VERDICT: Outcome D — apparent effect, but it does not survive the "
                     "controls. The same injection moves an unrelated no-answer question "
                     "by a comparable amount, so the detection measure is picking up a "
                     "general shift in the model's willingness to answer 'Yes' rather "
                     "than anything about its internal state. This is a confound, not "
                     "introspection.")
    elif significant and ident_significant:
        lines.append("VERDICT: Outcome A — an introspection-like effect. Injection raises "
                     "the detection signal above a norm-matched random control, AND the "
                     "model identifies the injected concept above chance, AND the "
                     "yes-bias control does not move. That is the pattern the hypothesis "
                     "predicted.")
        if ordering_ok is False:
            lines.append("  CAVEAT: the logit lens shows the concept becoming readable "
                         "only at or after the layer where the Yes/No decision forms. "
                         "That ordering is hard to reconcile with the model reading its "
                         "own state and then reporting on it; the injected vector may be "
                         "reaching the logits directly through the residual stream.")
        if mediated is False:
            lines.append("  CAVEAT: no downstream component's ablation meaningfully "
                         "reduces the injection's effect, which again points to a direct "
                         "residual-stream path rather than a read-and-report computation.")
    elif significant and not ident_significant:
        lines.append("VERDICT: Outcome B — a weak or partial effect. The model's detection "
                     "signal responds to the *content* of the injection, but it cannot "
                     "name what was injected above chance. That is consistent with the "
                     "injected vector perturbing the model in a content-dependent way "
                     "without the model having usable access to what the content is. At "
                     "this scale that is unsurprising and should not be read as evidence "
                     "of introspection.")
    else:
        lines.append("VERDICT: Outcome C — no detectable effect. Injecting a concept "
                     "direction does not move the detection signal beyond a norm-matched "
                     "random vector, and identification is at chance.")
        lines.append("  This is a useful and expected result. A 0.5B-parameter model is "
                     "roughly three orders of magnitude smaller than the models in which "
                     "the effect was reported, and the original work found the effect "
                     "present only ~20% of the time in a frontier model. A null here is "
                     "evidence about THIS model at THIS scale with THIS prompt. It is not "
                     "evidence against the published result.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--quick", action="store_true", help="small, fast configuration")
    parser.add_argument("--offline", action="store_true",
                        help="random-init model; no downloads; plumbing test only")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-intervention", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    if args.quick:
        config = apply_quick(config)
    if args.offline:
        config = apply_offline(config)

    print(RULE := "=" * 78)
    print("Introspection reproduction — full pipeline")
    print(RULE)
    print(f"model     {config['model']['name']}")
    print(f"seed      {config['seed']}")
    print(f"git       {git_commit()}")
    if args.offline:
        print("\n*** OFFLINE MODE: the model is randomly initialised. Every number "
              "below is meaningless. This mode exists to prove the code runs. ***")

    ctx, trials = build_context(config)
    record = record_environment(config, torch.device(ctx.model.cfg.device), ctx.summary)
    layer = ctx.primary_layer
    strength = config["injection"]["primary_strength"]
    results: dict[str, pd.DataFrame] = {}

    # ---- 1. baseline -------------------------------------------------------
    print(f"\n[1/6] Behavioural baseline (no injection)")
    baseline_rows = run_trials(ctx, trials, layer=layer, strength=0.0, desc="baseline")
    save_rows(baseline_rows, resolve(config, "raw") / "02_behavioral_baseline",
              config=config, extra=record)
    baseline = pd.DataFrame(baseline_rows)
    results["baseline"] = baseline

    # ---- 2. confirmatory ---------------------------------------------------
    print(f"[2/6] Confirmatory test (layer {layer}, strength {strength})")
    primary_rows = run_trials(ctx, trials, layer=layer, strength=strength, desc="confirmatory")
    save_rows(primary_rows, resolve(config, "raw") / "03_primary", config=config,
              extra=record | {"pre_registered_layer": layer})
    primary = pd.DataFrame(primary_rows)
    results["primary"] = primary

    det_table = detection_table(
        primary, reference="random_control",
        n_boot=config["stats"]["n_bootstrap"], n_perm=config["stats"]["n_permutations"],
        alpha=config["stats"]["alpha"], seed=config["seed"],
    )
    ident_table = identification_table(primary, seed=config["seed"],
                                       n_boot=config["stats"]["n_bootstrap"])
    yb_table = yesbias_table(
        primary, baseline, n_perm=config["stats"]["n_permutations"],
        n_boot=config["stats"]["n_bootstrap"], seed=config["seed"],
    )
    save_table(det_table, resolve(config, "processed") / "03_detection_table")
    save_table(ident_table, resolve(config, "processed") / "03_identification_table")
    save_table(yb_table, resolve(config, "processed") / "03_yesbias_control_table")

    # ---- 3. sweeps ---------------------------------------------------------
    if not args.skip_sweeps:
        print("[3/6] Exploratory layer sweep")
        sweep_rows = layer_sweep(ctx, trials, strength=strength)
        save_rows(sweep_rows, resolve(config, "raw") / "03_layer_sweep", config=config, extra=record)
        sweep = pd.DataFrame(sweep_rows)
        results["layer_sweep"] = sweep
        save_table(
            layer_sweep_table(sweep, n_perm=max(2000, config["stats"]["n_permutations"] // 5),
                              alpha=config["stats"]["alpha"], seed=config["seed"]),
            resolve(config, "processed") / "03_layer_sweep_table",
        )
        save_table(identification_by_layer(sweep),
                   resolve(config, "processed") / "03_identification_by_layer")

        print("[4/6] Exploratory strength sweep")
        strength_rows = strength_sweep(ctx, trials, layer=layer,
                                       strengths=config["injection"]["strengths"])
        save_rows(strength_rows, resolve(config, "raw") / "03_strength_sweep",
                  config=config, extra=record)
        results["strength_sweep"] = pd.DataFrame(strength_rows)
    else:
        print("[3/6] layer sweep   — skipped")
        print("[4/6] strength sweep — skipped")

    # ---- 4. internal analyses ---------------------------------------------
    print("[5/6] Internal analyses (logit lens, residual similarity)")
    concepts = list(dict.fromkeys(
        t.concept for t in trials
        if t.condition == "injected" and t.question_kind == "identification"
    ))[: config["intervention"]["n_concepts"]]
    distractor_map = {
        t.concept: t.distractors for t in trials
        if t.question_kind == "identification" and t.condition == "injected"
    }
    lens_rows, sim_rows = [], []
    for concept in concepts:
        lens_rows += logit_lens_by_layer(ctx, concept, inject_layer=layer, strength=strength,
                                         distractors=distractor_map.get(concept, []))
        sim_rows += residual_similarity(ctx, concept, inject_layer=layer, strength=strength)
    save_rows(lens_rows, resolve(config, "raw") / "03_logit_lens", config=config, extra=record)
    save_rows(sim_rows, resolve(config, "raw") / "03_residual_similarity",
              config=config, extra=record)
    results["logit_lens"] = pd.DataFrame(lens_rows)
    results["residual_similarity"] = pd.DataFrame(sim_rows)

    # ---- 5. intervention ---------------------------------------------------
    ablation = None
    if config["intervention"]["enabled"] and not args.skip_intervention:
        print("[6/6] Causal ablation sweep")
        ablation_rows = ablation_sweep(
            ctx, trials, layer=layer, strength=strength,
            components=config["intervention"]["components"],
            mode=config["intervention"]["mode"],
            n_concepts=config["intervention"]["n_concepts"],
        )
        save_rows(ablation_rows, resolve(config, "raw") / "04_ablation",
                  config=config, extra=record)
        ablation = pd.DataFrame(ablation_rows)
        results["ablation"] = ablation
        save_table(
            ablation.groupby(["component", "ablate_layer"])
            .agg(mean_interaction=("interaction", "mean"),
                 mean_effect_intact=("injection_effect_intact", "mean"),
                 N=("interaction", "size")).reset_index().sort_values("mean_interaction"),
            resolve(config, "processed") / "04_ablation_table",
        )
    else:
        print("[6/6] intervention — skipped")

    # ---- 6. transcripts ----------------------------------------------------
    if config["generation"]["enabled"]:
        transcripts = []
        for condition in ("injected", "random_control", "no_injection"):
            subset = [t for t in trials if t.condition == condition]
            transcripts += sample_transcripts(
                ctx, subset, layer=layer,
                strength=0.0 if condition == "no_injection" else strength,
                n_examples=config["generation"]["n_examples"],
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
            )
        save_rows(transcripts, resolve(config, "raw") / "03_transcripts", config=config)

    # ---- figures -----------------------------------------------------------
    print("\nRendering figures")
    figures = make_all_figures(results, resolve(config, "figures"),
                               primary_layer=layer, seed=config["seed"])
    for path in figures:
        print(f"  {path}")

    # ---- report ------------------------------------------------------------
    from src.metrics import format_markdown

    text = verdict(det_table, ident_table, yb_table, config["stats"]["alpha"],
                   ablation, results.get("logit_lens"))

    elapsed = time.time() - started
    report = [
        "# Results",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- model: `{config['model']['name']}` "
        f"({ctx.summary['n_params']/1e6:.0f}M params, {ctx.summary['n_layers']} layers)",
        f"- device: `{record['environment']['device']}`, seed `{config['seed']}`, "
        f"git `{git_commit()}`",
        f"- injection: layer {layer} (pre-registered, {config['injection']['primary_layer_frac']:.0%} "
        f"depth), strength {strength} residual norms, window `{config['injection']['window']}`",
        f"- runtime: {elapsed/60:.1f} min",
        "",
        "## Detection — logit(Yes) − logit(No), reference = random_control",
        "",
        format_markdown(det_table),
        "",
        "## Identification — forced choice",
        "",
        format_markdown(ident_table),
        "",
        "## Yes-bias control — same injection, unrelated no-answer question",
        "",
        format_markdown(yb_table),
        "",
        "## Verdict",
        "",
        "```",
        text,
        "```",
        "",
        "## Figures",
        "",
        *[f"- `{p.relative_to(ROOT)}`" for p in figures],
        "",
        "See `results/figures/CAPTIONS.md` for how to read each figure.",
        "",
    ]
    if args.offline:
        report.insert(1, "\n> **OFFLINE MODE.** The model was randomly initialised. "
                         "These numbers are a plumbing check, not a result.\n")
    report_path = ROOT / "results" / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    (ROOT / "results" / "run_record.json").write_text(
        json.dumps({"config": config, **record, "elapsed_seconds": elapsed,
                    "git": git_commit()}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + RULE)
    print(det_table.to_string(index=False))
    print()
    print(ident_table.to_string(index=False))
    print("\n" + RULE)
    print(text)
    print(RULE)
    print(f"\nReport written to {report_path}")
    print(f"Total runtime {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
