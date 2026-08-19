"""Experiment 05 — one example, end to end, with everything printed.

Experiments 02-04 aggregate over 30 concepts and report statistics. This script
does the opposite: it takes a single concept and shows every step, so that
someone learning mechanistic interpretability can follow exactly what the
pipeline does to the model.

  1. the input prompt (and its chat formatting)
  2. tokenization, with the injection window marked
  3. a clean forward pass
  4. the model's response with nothing injected
  5. concept-vector extraction
  6. layer-by-layer logit-lens analysis
  7. the introspective report under injection
  8. the norm-matched random control
  9. a causal ablation
 10. how the output changed

Run:  python experiments/05_single_example_walkthrough.py --concept ocean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import behavioral as beh                                 # noqa: E402
from src import hooks as hk                                       # noqa: E402
from src.introspection import logit_lens_by_layer, prepare        # noqa: E402
from src.prompts import (                                          # noqa: E402
    CONCEPT_WORDS, baseline_words, detection_messages, free_messages, identification_messages,
)
from src.runner import load_config, project_root                   # noqa: E402


RULE = "=" * 78


def section(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "config.yaml"))
    parser.add_argument("--concept", default="ocean")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    concept = args.concept
    distractors = [w for w in CONCEPT_WORDS if w != concept][:7]

    ctx = prepare(config, concepts=[concept, *distractors])
    model = ctx.model
    layer = args.layer if args.layer is not None else ctx.primary_layer
    strength = args.strength if args.strength is not None else config["injection"]["primary_strength"]
    hook_name = hk.resid_name(layer, config["concept_vector"]["hook"])

    # ---------------------------------------------------------------- 1 & 2
    section(1, "THE INPUT PROMPT")
    messages = detection_messages()
    for msg in messages:
        print(f"  [{msg['role']:>9}] {msg['content']}")
    print("\n  Note: the word " + repr(concept) + " appears nowhere in this prompt.")

    text, tokens, start = hk.injection_start(model, messages, window=config["injection"]["window"])
    section(2, "TOKENIZATION AND THE INJECTION WINDOW")
    strs = model.tokenizer.convert_ids_to_tokens(tokens[0].tolist())
    print(f"  {len(strs)} tokens; injection applies from index {start} onward "
          f"({len(strs) - start} prompt tokens, plus everything generated).")
    print("\n  ...context before the window:")
    print("   ", " ".join(repr(s) for s in strs[max(0, start - 10):start]))
    print("\n  window begins here:")
    print("   ", " ".join(repr(s) for s in strs[start:start + 24]))
    print("    ...")
    print("   ", " ".join(repr(s) for s in strs[-8:]))

    # ---------------------------------------------------------------- 3 & 4
    section(3, "CLEAN FORWARD PASS")
    clean = beh.yes_no_readout(model, tokens, ctx.answer_ids)
    for key, value in clean.items():
        print(f"  {key:20s} {value}")

    section(4, "MODEL RESPONSE WITH NOTHING INJECTED")
    free_text, free_tokens, free_start = hk.injection_start(
        model, free_messages(), window=config["injection"]["window"]
    )
    baseline_response = beh.generate(
        model, free_tokens, max_new_tokens=config["generation"]["max_new_tokens"], temperature=0.0
    )
    print(f"  {baseline_response.strip()!r}")

    # ---------------------------------------------------------------- 5
    section(5, "CONCEPT-VECTOR EXTRACTION")
    vector = ctx.concept_vectors[concept][layer]
    fillers = baseline_words(config["concept_vector"]["n_baseline_words"], seed=config["seed"])
    print(f"  template          {config['concept_vector']['template']!r}")
    print(f"  read at           blocks.{layer}.hook_{config['concept_vector']['hook']}, "
          f"position {config['concept_vector']['read_position']}")
    print(f"  baseline          mean over {len(fillers)} filler words "
          f"({', '.join(fillers[:5])}, ...)")
    print(f"  vector shape      {tuple(vector.shape)}")
    print(f"  ||v||             {vector.norm().item():.3f}")
    print(f"  layer norm unit   {ctx.norm_units[layer]:.3f}  "
          "(mean ||resid_post|| on this prompt)")
    coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[layer])
    print(f"  strength {strength}  ->  raw coefficient {coefficient:.4f}")
    print(f"  so we add {coefficient:.4f} * v, whose norm is "
          f"{(coefficient * vector.norm()).item():.3f} = {strength} residual norms")

    inject_fn = hk.make_injection_hook(vector, coefficient, start)

    # ---------------------------------------------------------------- 6
    section(6, "LAYER-BY-LAYER LOGIT LENS UNDER INJECTION")
    lens = logit_lens_by_layer(
        ctx, concept, inject_layer=layer, strength=strength, distractors=distractors
    )
    print(f"  {'layer':>5}  {'logit(concept)-mean(distractors)':>34}  {'logit(Yes)-logit(No)':>22}")
    for row in lens:
        marker = "  <- injection site" if row["read_layer"] == layer else ""
        print(f"  {row['read_layer']:>5}  {row['concept_logit_adv']:>34.3f}  "
              f"{row['yes_minus_no']:>22.3f}{marker}")

    # ---------------------------------------------------------------- 7
    section(7, "INTROSPECTIVE REPORT UNDER INJECTION")
    injected = beh.yes_no_readout(model, tokens, ctx.answer_ids, hook_name, inject_fn)
    print(f"  clean     yes_minus_no = {clean['yes_minus_no']:+.4f}   "
          f"p(Yes|Yes,No) = {clean['p_yes_pairwise']:.3f}")
    print(f"  injected  yes_minus_no = {injected['yes_minus_no']:+.4f}   "
          f"p(Yes|Yes,No) = {injected['p_yes_pairwise']:.3f}")
    print(f"  change    {injected['yes_minus_no'] - clean['yes_minus_no']:+.4f}")

    free_inject_fn = hk.make_injection_hook(vector, coefficient, free_start)
    injected_response = beh.generate(
        model, free_tokens, max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=0.0, hook_name=hook_name, hook_fn=free_inject_fn,
    )
    print(f"\n  free response under injection:\n    {injected_response.strip()!r}")
    print(f"  mentions {concept!r}: {concept.lower() in injected_response.lower()}")

    ident_text, ident_tokens, ident_start = hk.injection_start(
        model, identification_messages(), window=config["injection"]["window"]
    )
    ident_fn = hk.make_injection_hook(vector, coefficient, ident_start)
    choice = beh.forced_choice(
        model, ident_tokens, correct=concept, distractors=distractors,
        hook_name=hook_name, hook_fn=ident_fn,
    )
    print(f"\n  forced choice: picked {choice['winner']!r} "
          f"(correct: {choice['correct']}, rank of true concept: {choice['rank']}/"
          f"{choice['n_candidates']}, margin {choice['margin']:+.4f})")
    for word, score in sorted(choice["scores"].items(), key=lambda kv: -kv[1]):
        flag = "  <- true concept" if word == concept else ""
        print(f"    {word:<16} {score:8.4f}{flag}")

    # ---------------------------------------------------------------- 8
    section(8, "NORM-MATCHED RANDOM CONTROL")
    random_vec = hk.norm_matched_random(vector, seed=config["seed"])
    random_coeff = hk.injection_coefficient(random_vec, strength, ctx.norm_units[layer])
    random_fn = hk.make_injection_hook(random_vec, random_coeff, start)
    control = beh.yes_no_readout(model, tokens, ctx.answer_ids, hook_name, random_fn)
    print(f"  ||v_random|| = {random_vec.norm().item():.3f} "
          f"(matched to ||v_concept|| = {vector.norm().item():.3f})")
    print(f"  random    yes_minus_no = {control['yes_minus_no']:+.4f}")
    print(f"  concept   yes_minus_no = {injected['yes_minus_no']:+.4f}")
    print(f"  concept - random = {injected['yes_minus_no'] - control['yes_minus_no']:+.4f}")
    print("\n  This difference, not the difference from the clean run, is the quantity")
    print("  the introspection hypothesis is about. A perturbation of any kind changes")
    print("  the answer; only content-specific change is evidence about introspection.")

    # ---------------------------------------------------------------- 9 & 10
    section(9, "CAUSAL ABLATION DOWNSTREAM OF THE INJECTION")
    ablate_fn = hk.make_ablation_hook(mode="zero")
    print(f"  {'layer':>5}  {'component':>10}  {'injected':>10}  {'ablated':>10}  {'interaction':>12}")
    downstream = [l for l in range(model.cfg.n_layers) if l > layer]
    base_clean = clean["yes_minus_no"]
    base_injected = injected["yes_minus_no"]
    worst = None
    for down in downstream:
        for component in ("attn_out", "mlp_out"):
            comp = hk.component_name(down, component)
            model.reset_hooks()
            model.add_hook(comp, ablate_fn)
            abl_clean = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
            model.reset_hooks()
            model.add_hook(hook_name, inject_fn)
            model.add_hook(comp, ablate_fn)
            abl_injected = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
            model.reset_hooks()
            interaction = (abl_injected - abl_clean) - (base_injected - base_clean)
            print(f"  {down:>5}  {component:>10}  {base_injected:>10.3f}  "
                  f"{abl_injected:>10.3f}  {interaction:>12.3f}")
            if worst is None or interaction < worst[2]:
                worst = (down, component, interaction)

    section(10, "WHAT CHANGED")
    print(f"  concept                      {concept}")
    print(f"  injection layer / strength   {layer} / {strength}")
    print(f"  detection, clean             {base_clean:+.4f}")
    print(f"  detection, random control    {control['yes_minus_no']:+.4f}")
    print(f"  detection, concept injected  {base_injected:+.4f}")
    print(f"  content-specific effect      "
          f"{base_injected - control['yes_minus_no']:+.4f}")
    print(f"  identification correct       {choice['correct']} "
          f"(chance = {1 / choice['n_candidates']:.3f})")
    if worst is not None:
        print(f"  most mediating component     layer {worst[0]} {worst[1]} "
              f"(interaction {worst[2]:+.3f})")
    print("\n  One example proves nothing. Run experiment 03 for the statistics.")


if __name__ == "__main__":
    main()
