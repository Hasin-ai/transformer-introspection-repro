"""Causal interventions downstream of the injection site.

Why a second intervention at all
--------------------------------
The injection is already a causal intervention: it establishes that *adding a
concept direction* changes the model's answer. What it does not establish is
that the model *reads* that direction with any machinery, as opposed to the
added vector simply leaking into the output through the residual stream's
skip connection. That distinction is the whole difference between "the model has
access to information about its internal state" and "we shoved a vector into the
final logits by a slightly indirect route".

The ablation sweep addresses it. For every layer *downstream* of the injection
site, each component's output (attention or MLP) is knocked out and the
detection readout re-measured.

Double subtraction
------------------
Ablating a component damages the model whether or not anything was injected, so
the raw drop is uninformative. What is reported is the *interaction*:

    effect(l, c) = [ injected_ablated - injected_clean_run ]
                 - [ baseline_ablated - baseline_clean_run ]

i.e. how much more (or less) that component matters when the concept is present.
A component that specifically mediates the injected signal shows a large
negative interaction: removing it destroys the injection's effect while leaving
the un-injected model roughly where it was.

Only layers *after* the injection layer are swept. Ablating an earlier layer
cannot mediate an effect that has not been introduced yet, and including them
would just inflate the multiple-comparison burden.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from . import behavioral as beh
from . import hooks as hk
from .introspection import Context
from .prompts import Trial, detection_messages


@torch.no_grad()
def ablation_sweep(
    ctx: Context,
    trials: Sequence[Trial],
    layer: int,
    strength: float,
    components: Sequence[str] = ("attn_out", "mlp_out"),
    mode: str = "zero",
    n_concepts: int = 8,
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Knock out each downstream component and measure the change in detection.

    Returns one row per (concept, downstream layer, component), carrying all four
    measurements needed for the double subtraction.
    """
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]
    inject_hook_name = hk.resid_name(layer, hook_point)

    messages = detection_messages()
    _, tokens, start = hk.injection_start(model, messages, window=window)

    # Only injected-condition detection trials; the control conditions have
    # their own arm of the experiment and are not what this analysis is about.
    selected = [
        t for t in trials if t.condition == "injected" and t.question_kind == "detection"
    ][:n_concepts]

    downstream = [l for l in range(model.cfg.n_layers) if l > layer]
    ablate_fn = hk.make_ablation_hook(mode=mode)

    rows: list[dict[str, Any]] = []
    for trial in tqdm(selected, desc="ablation sweep"):
        vector = ctx.concept_vectors[trial.concept][layer]
        coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[layer])
        inject_fn = hk.make_injection_hook(vector, coefficient, start)

        # Two un-ablated reference runs.
        base_clean = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
        base_injected = beh.yes_no_readout(
            model, tokens, ctx.answer_ids, inject_hook_name, inject_fn
        )["yes_minus_no"]

        for down in downstream:
            for component in components:
                comp_name = hk.component_name(down, component)

                # Ablation without injection.
                model.reset_hooks()
                model.add_hook(comp_name, ablate_fn)
                try:
                    abl_clean = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
                finally:
                    model.reset_hooks()

                # Ablation with injection.
                model.reset_hooks()
                model.add_hook(inject_hook_name, inject_fn)
                model.add_hook(comp_name, ablate_fn)
                try:
                    abl_injected = beh.yes_no_readout(
                        model, tokens, ctx.answer_ids
                    )["yes_minus_no"]
                finally:
                    model.reset_hooks()

                injection_effect_intact = base_injected - base_clean
                injection_effect_ablated = abl_injected - abl_clean
                rows.append(
                    {
                        "concept": trial.concept,
                        "inject_layer": layer,
                        "strength": strength,
                        "ablate_layer": down,
                        "component": component,
                        "mode": mode,
                        "base_clean": base_clean,
                        "base_injected": base_injected,
                        "ablated_clean": abl_clean,
                        "ablated_injected": abl_injected,
                        "injection_effect_intact": injection_effect_intact,
                        "injection_effect_ablated": injection_effect_ablated,
                        # Negative = ablating this component removes the
                        # injection's effect, i.e. it mediates the signal.
                        "interaction": injection_effect_ablated - injection_effect_intact,
                    }
                )
    model.reset_hooks()
    return rows


@torch.no_grad()
def restore_sweep(
    ctx: Context,
    trials: Sequence[Trial],
    layer: int,
    strength: float,
    n_concepts: int = 8,
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Overwrite the injected residual stream with the clean one at layer ``l``.

    This is activation patching in the "restore" direction: the injection is
    applied at ``layer``, then at each downstream layer ``l`` the residual stream
    is replaced wholesale by its clean value. The layer at which restoring stops
    abolishing the effect tells you how far downstream the injected information
    still needs to travel. If the effect survives restoration immediately after
    the injection layer, the vector is reaching the logits through the residual
    skip connection rather than through any computation.
    """
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]
    inject_hook_name = hk.resid_name(layer, hook_point)

    messages = detection_messages()
    _, tokens, start = hk.injection_start(model, messages, window=window)

    names = {hk.resid_name(l, "resid_post") for l in range(model.cfg.n_layers)}
    model.reset_hooks()
    _, clean_cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)

    selected = [
        t for t in trials if t.condition == "injected" and t.question_kind == "detection"
    ][:n_concepts]
    downstream = [l for l in range(model.cfg.n_layers) if l > layer]

    rows: list[dict[str, Any]] = []
    for trial in tqdm(selected, desc="restore sweep"):
        vector = ctx.concept_vectors[trial.concept][layer]
        coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[layer])
        inject_fn = hk.make_injection_hook(vector, coefficient, start)

        base_clean = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
        base_injected = beh.yes_no_readout(
            model, tokens, ctx.answer_ids, inject_hook_name, inject_fn
        )["yes_minus_no"]

        for down in downstream:
            clean_value = clean_cache[hk.resid_name(down, "resid_post")]

            def restore_fn(activation, hook, _clean=clean_value):  # noqa: ANN001, ARG001
                out = activation.clone()
                n = min(out.shape[1], _clean.shape[1])
                out[:, :n, :] = _clean[:, :n, :].to(out.device, out.dtype)
                return out

            model.reset_hooks()
            model.add_hook(inject_hook_name, inject_fn)
            model.add_hook(hk.resid_name(down, "resid_post"), restore_fn)
            try:
                restored = beh.yes_no_readout(model, tokens, ctx.answer_ids)["yes_minus_no"]
            finally:
                model.reset_hooks()

            rows.append(
                {
                    "concept": trial.concept,
                    "inject_layer": layer,
                    "strength": strength,
                    "restore_layer": down,
                    "base_clean": base_clean,
                    "base_injected": base_injected,
                    "restored": restored,
                    # 1.0 = restoring completely abolished the injection effect,
                    # 0.0 = the effect survived restoration.
                    "fraction_abolished": (
                        (base_injected - restored) / (base_injected - base_clean)
                        if abs(base_injected - base_clean) > 1e-6
                        else float("nan")
                    ),
                }
            )
    model.reset_hooks()
    return rows
