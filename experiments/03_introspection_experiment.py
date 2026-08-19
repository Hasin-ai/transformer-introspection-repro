"""Experiment 03 — the introspection experiment.

Three parts, in this order:

1. **Confirmatory test.** Injected vs norm-matched random control at the
   pre-registered layer and strength from ``config.yaml``, paired by concept,
   two-sided permutation test. Plus the two other controls (no injection,
   yes-bias question) as descriptive context.

2. **Exploratory sweeps.** Layer sweep and strength sweep. Labelled as
   exploratory everywhere, Holm-corrected across layers.

3. **Internal analyses.** Logit lens through the stack and cosine similarity
   between injected and clean residual streams — the checks that distinguish
   "the model read the injected concept" from "the injected vector landed in the
   logits by the shortest available path".

Run:  python experiments/03_introspection_experiment.py
      python experiments/03_introspection_experiment.py --skip-sweeps   (fast)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.introspection import (                                   # noqa: E402
    logit_lens_by_layer, layer_sweep, residual_similarity, run_trials,
    sample_transcripts, strength_sweep,
)
from src.metrics import (                                          # noqa: E402
    detection_table, identification_by_layer, identification_table, layer_sweep_table,
)
from src.runner import (                                           # noqa: E402
    build_context, load_config, project_root, record_environment, resolve, save_rows, save_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "config.yaml"))
    parser.add_argument("--skip-sweeps", action="store_true",
                        help="run only the confirmatory test and internal analyses")
    parser.add_argument("--layer", type=int, default=None,
                        help="override the pre-registered injection layer (marks the run exploratory)")
    args = parser.parse_args()

    config = load_config(args.config)
    ctx, trials = build_context(config)
    record = record_environment(config, torch.device(ctx.model.cfg.device), ctx.summary)

    layer = args.layer if args.layer is not None else ctx.primary_layer
    strength = config["injection"]["primary_strength"]
    if args.layer is not None:
        print(f"[03] NOTE: layer overridden to {layer}; this is no longer the "
              "pre-registered confirmatory test.")

    # ---------------------------------------------------------------- 1. confirmatory
    print(f"\n=== Confirmatory test: layer {layer}, strength {strength}, "
          f"window {config['injection']['window']} ===")
    primary_rows = run_trials(ctx, trials, layer=layer, strength=strength, desc="confirmatory")
    save_rows(
        primary_rows,
        resolve(config, "raw") / "03_primary",
        config=config,
        extra=record | {"pre_registered_layer": ctx.primary_layer, "layer_used": layer},
    )
    primary = pd.DataFrame(primary_rows)

    det_table = detection_table(
        primary,
        reference="random_control",
        n_boot=config["stats"]["n_bootstrap"],
        n_perm=config["stats"]["n_permutations"],
        alpha=config["stats"]["alpha"],
        seed=config["seed"],
    )
    ident_table = identification_table(primary, seed=config["seed"],
                                       n_boot=config["stats"]["n_bootstrap"])
    save_table(det_table, resolve(config, "processed") / "03_detection_table")
    save_table(ident_table, resolve(config, "processed") / "03_identification_table")

    print("\n--- Detection (logit(Yes) - logit(No)), reference = random_control ---")
    print(det_table.to_string(index=False))
    print("\n--- Identification (forced choice) ---")
    print(ident_table.to_string(index=False))

    # ---------------------------------------------------------------- 2. sweeps
    if not args.skip_sweeps:
        print("\n=== Exploratory layer sweep ===")
        sweep_rows = layer_sweep(ctx, trials, strength=strength)
        save_rows(sweep_rows, resolve(config, "raw") / "03_layer_sweep",
                  config=config, extra=record)
        sweep = pd.DataFrame(sweep_rows)
        ls_table = layer_sweep_table(
            sweep, reference="random_control",
            n_perm=max(2000, config["stats"]["n_permutations"] // 5),
            alpha=config["stats"]["alpha"], seed=config["seed"],
        )
        save_table(ls_table, resolve(config, "processed") / "03_layer_sweep_table")
        save_table(identification_by_layer(sweep),
                   resolve(config, "processed") / "03_identification_by_layer")
        print(ls_table.to_string(index=False))

        print("\n=== Exploratory strength sweep ===")
        strength_rows = strength_sweep(
            ctx, trials, layer=layer, strengths=config["injection"]["strengths"]
        )
        save_rows(strength_rows, resolve(config, "raw") / "03_strength_sweep",
                  config=config, extra=record)

    # ---------------------------------------------------------------- 3. internals
    print("\n=== Internal analyses ===")
    concepts = [t.concept for t in trials
                if t.condition == "injected" and t.question_kind == "identification"]
    concepts = list(dict.fromkeys(concepts))[: config["intervention"]["n_concepts"]]
    distractor_map = {
        t.concept: t.distractors for t in trials
        if t.question_kind == "identification" and t.condition == "injected"
    }

    lens_rows: list[dict] = []
    sim_rows: list[dict] = []
    for concept in concepts:
        lens_rows += logit_lens_by_layer(
            ctx, concept, inject_layer=layer, strength=strength,
            distractors=distractor_map.get(concept, []),
        )
        sim_rows += residual_similarity(ctx, concept, inject_layer=layer, strength=strength)
    save_rows(lens_rows, resolve(config, "raw") / "03_logit_lens", config=config, extra=record)
    save_rows(sim_rows, resolve(config, "raw") / "03_residual_similarity",
              config=config, extra=record)

    lens = pd.DataFrame(lens_rows)
    if len(lens):
        peak_concept = int(lens.groupby("read_layer")["concept_logit_adv"].mean().idxmax())
        peak_yes = int(lens.groupby("read_layer")["yes_minus_no"].mean().idxmax())
        print(f"  concept becomes most readable at layer {peak_concept}")
        print(f"  'Yes' preference peaks at layer         {peak_yes}")
        if peak_concept >= peak_yes:
            print("  NOTE: the concept does not become readable before the Yes/No "
                  "decision forms. That ordering is inconsistent with the model "
                  "reading the injected concept and then reporting on it.")

    # ---------------------------------------------------------------- transcripts
    if config["generation"]["enabled"]:
        transcripts: list[dict] = []
        for condition in ("injected", "random_control"):
            subset = [t for t in trials if t.condition == condition]
            transcripts += sample_transcripts(
                ctx, subset, layer=layer, strength=strength,
                n_examples=config["generation"]["n_examples"],
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
            )
        save_rows(transcripts, resolve(config, "raw") / "03_transcripts", config=config)
        print("\n=== Sample responses under injection (qualitative only) ===")
        for row in transcripts:
            if row["condition"] == "injected":
                flag = "  <- names the concept" if row["mentions_concept"] else ""
                print(f"  [{row['concept']}] {row['response'].strip()[:150]!r}{flag}")

    print("\nDone. Tables in results/processed/, raw rows in results/raw/.")


if __name__ == "__main__":
    main()
