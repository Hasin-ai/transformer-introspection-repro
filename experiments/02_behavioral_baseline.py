"""Experiment 02 — behavioural baseline, with no injection at all.

This has to run first, and it has to be read before anything else, because it
establishes whether the detection measure is even usable on this model:

* Does the model answer the yes/no question with "Yes" or "No" at all, or does
  it put its probability mass on a newline or a chat token? (``argmax_is_yes_no``)
* What is its *default* answer to "do you detect an injected thought"? If a
  0.5B model says "Yes" unprompted, the measure has a floor problem and every
  later comparison must be read relative to this baseline, not to zero.
* Is forced-choice identification already above chance with nothing injected?
  It should be at chance. If it is not, the prompt is leaking the answer, and
  the whole experiment is invalid before it starts.

Run:  python experiments/02_behavioral_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavioral import check_answer_tokens                   # noqa: E402
from src.introspection import run_trials, sample_transcripts     # noqa: E402
from src.metrics import bootstrap_mean, binomial_test_vs_chance  # noqa: E402
from src.runner import (                                          # noqa: E402
    build_context, load_config, project_root, record_environment, resolve, save_rows, save_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    ctx, trials = build_context(config)

    device = ctx.model.cfg.device
    record = record_environment(config, __import__("torch").device(device), ctx.summary)

    clean = check_answer_tokens(ctx.model, ("Yes", "No"))
    print(f"[baseline] Yes/No are single tokens: {clean}")

    # Strength 0 on every trial: this is the model's untouched behaviour.
    rows = run_trials(ctx, trials, layer=ctx.primary_layer, strength=0.0, desc="baseline")
    raw_path = save_rows(
        rows,
        resolve(config, "raw") / "02_behavioral_baseline",
        config=config,
        extra=record | {"answer_tokens_single": clean},
    )
    print(f"[baseline] wrote {raw_path}")

    frame = pd.DataFrame(rows)
    detection = frame[frame["question_kind"] == "detection"]
    ident = frame[frame["question_kind"] == "identification"]
    yesbias = frame[frame["question_kind"] == "yesbias"]

    yes_id = ctx.answer_ids["Yes"]
    no_id = ctx.answer_ids["No"]
    argmax_ok = detection["argmax_token_id"].isin([yes_id, no_id]).mean() if len(detection) else 0.0

    summary_rows = []
    for name, group in (("detection", detection), ("yesbias", yesbias)):
        if not len(group):
            continue
        stats = bootstrap_mean(group["yes_minus_no"].to_numpy(), seed=config["seed"])
        summary_rows.append(
            {
                "measure": f"{name}: yes_minus_no",
                "N": stats["n"],
                "mean": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "frac_yes": float((group["yes_minus_no"] > 0).mean()),
            }
        )
    if len(ident):
        test = binomial_test_vs_chance(
            int(ident["id_correct"].sum()), len(ident), float(ident["chance"].iloc[0])
        )
        summary_rows.append(
            {
                "measure": "identification: accuracy",
                "N": test["n"],
                "mean": test["accuracy"],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "frac_yes": test["chance"],
            }
        )

    table = pd.DataFrame(summary_rows)
    save_table(table, resolve(config, "processed") / "02_baseline_summary")
    print("\n=== Baseline (no injection) ===")
    print(table.to_string(index=False))
    print(f"\nFraction of detection trials whose argmax token is Yes or No: {argmax_ok:.2f}")
    if argmax_ok < 0.5:
        print("  WARNING: the model mostly does not answer with Yes/No. The pairwise "
              "logit readout is still valid, but free generations will look unlike the "
              "transcripts in the original paper.")
    if len(ident):
        chance = float(ident["chance"].iloc[0])
        acc = float(ident["id_correct"].mean())
        print(f"Baseline identification accuracy {acc:.3f} vs chance {chance:.3f}")
        if acc > chance * 2:
            print("  WARNING: identification is well above chance with NOTHING injected. "
                  "The prompt or the candidate set is leaking. Fix this before "
                  "interpreting any injected condition.")

    if config["generation"]["enabled"]:
        transcripts = sample_transcripts(
            ctx, trials, layer=ctx.primary_layer, strength=0.0,
            n_examples=config["generation"]["n_examples"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"]["temperature"],
        )
        save_rows(transcripts, resolve(config, "raw") / "02_baseline_transcripts", config=config)
        print("\n=== Sample baseline responses (no injection) ===")
        for row in transcripts[:3]:
            print(f"  - {row['response'].strip()[:180]!r}")


if __name__ == "__main__":
    main()
