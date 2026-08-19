"""Experiment 04 — causal interventions downstream of the injection.

The injection itself already establishes causality in one direction: change the
residual stream, change the answer. What it does not establish is *mechanism*.
An injected vector can reach the logits two ways:

  (a) later layers read it, compute something about it, and write a decision;
  (b) it simply rides the residual stream's skip connection into the
      unembedding, tilting the logits directly.

Only (a) is interesting. (b) would produce a perfectly real, perfectly
significant behavioural effect that has nothing to do with introspection.

Two sweeps distinguish them:

* ``ablation_sweep`` — knock out each downstream attention/MLP output and see
  whether the injection's effect survives. Reported as a double subtraction so
  that generic ablation damage cancels.
* ``restore_sweep`` — patch the clean residual stream back in at each downstream
  layer. If restoring immediately after the injection site already abolishes the
  effect, the signal was still travelling in the residual stream and had not yet
  been read by anything.

Run:  python experiments/04_activation_intervention.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interventions import ablation_sweep, restore_sweep       # noqa: E402
from src.runner import (                                           # noqa: E402
    build_context, load_config, project_root, record_environment, resolve, save_rows, save_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "config.yaml"))
    parser.add_argument("--skip-restore", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config["intervention"]["enabled"]:
        print("[04] intervention.enabled is false in config.yaml; nothing to do.")
        return

    ctx, trials = build_context(config)
    record = record_environment(config, torch.device(ctx.model.cfg.device), ctx.summary)
    layer = ctx.primary_layer
    strength = config["injection"]["primary_strength"]

    print(f"\n=== Ablation sweep: injection at layer {layer}, strength {strength} ===")
    rows = ablation_sweep(
        ctx, trials, layer=layer, strength=strength,
        components=config["intervention"]["components"],
        mode=config["intervention"]["mode"],
        n_concepts=config["intervention"]["n_concepts"],
    )
    save_rows(rows, resolve(config, "raw") / "04_ablation", config=config, extra=record)
    ablation = pd.DataFrame(rows)

    table = (
        ablation.groupby(["component", "ablate_layer"])
        .agg(
            mean_interaction=("interaction", "mean"),
            sd_interaction=("interaction", "std"),
            mean_effect_intact=("injection_effect_intact", "mean"),
            mean_effect_ablated=("injection_effect_ablated", "mean"),
            N=("interaction", "size"),
        )
        .reset_index()
        .sort_values("mean_interaction")
    )
    save_table(table, resolve(config, "processed") / "04_ablation_table")

    print("\n--- Components whose removal most reduces the injection effect ---")
    print(table.head(10).to_string(index=False))

    intact = float(ablation["injection_effect_intact"].mean())
    print(f"\nMean injection effect with the model intact: {intact:+.3f}")
    if abs(intact) < 1e-3:
        print("  The injection has essentially no behavioural effect to begin with, so "
              "the ablation interaction is not interpretable. Read experiment 03 first.")

    if not args.skip_restore:
        print(f"\n=== Restore sweep: injection at layer {layer} ===")
        restore_rows = restore_sweep(
            ctx, trials, layer=layer, strength=strength,
            n_concepts=config["intervention"]["n_concepts"],
        )
        save_rows(restore_rows, resolve(config, "raw") / "04_restore", config=config, extra=record)
        restore = pd.DataFrame(restore_rows)
        restore_table = (
            restore.groupby("restore_layer")
            .agg(mean_fraction_abolished=("fraction_abolished", "mean"),
                 sd=("fraction_abolished", "std"), N=("fraction_abolished", "size"))
            .reset_index()
        )
        save_table(restore_table, resolve(config, "processed") / "04_restore_table")
        print(restore_table.to_string(index=False))

    print("\nDone. Tables in results/processed/, raw rows in results/raw/.")


if __name__ == "__main__":
    main()
