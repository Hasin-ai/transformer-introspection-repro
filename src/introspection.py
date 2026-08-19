"""The introspection experiment itself.

Structure
---------
``prepare``            load model, extract concept vectors, fix the answer tokens
``run_trials``         one condition x one layer x one strength -> result rows
``layer_sweep``        exploratory: detection effect as a function of depth
``strength_sweep``     exploratory: detection effect as a function of magnitude
``logit_lens_by_layer``  where in the stack the injected concept becomes readable
``residual_similarity``  how far the injection pushes the residual stream, by layer

The confirmatory test
---------------------
The pre-registered comparison, fixed in ``config.yaml`` before any results were
looked at, is:

    injected vs random_control, at layer = round(0.667 * n_layers),
    strength = 2.0, on the detection measure ``yes_minus_no``,
    paired by concept, two-sided permutation test.

The layer and strength sweeps are explicitly exploratory. Reporting the best
layer from a sweep as though it were a hypothesis test is the single easiest way
to manufacture a false positive here, so the two are kept separate and the sweep
figures are labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from . import behavioral as beh
from . import hooks as hk
from .model import chat_prompt, load_model, model_summary, to_tokens
from .prompts import (
    Trial,
    baseline_words,
    detection_messages,
    free_messages,
    identification_messages,
    yesbias_messages,
)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """Everything the experiment functions need, assembled once."""

    model: Any
    config: dict[str, Any]
    layers: list[int]
    primary_layer: int
    concept_vectors: dict[str, dict[int, torch.Tensor]]
    norm_units: dict[int, float]
    answer_ids: dict[str, int]
    answer_token_clean: dict[str, bool]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers


def primary_layer_index(n_layers: int, frac: float) -> int:
    """Layer index at a given fraction of depth, clamped to a valid block."""
    return max(0, min(n_layers - 1, int(round(frac * (n_layers - 1)))))


def prepare(config: dict[str, Any], concepts: Sequence[str], device=None) -> Context:
    """Load the model and extract everything that does not depend on condition."""
    model = load_model(
        config["model"]["name"],
        dtype=config["model"]["dtype"],
        device=device,
        prefer=config["device"]["prefer"],
        force_cpu=config["device"]["force_cpu"],
        verify=config["device"]["verify"],
        mps_tolerance=config["device"]["mps_tolerance"],
    )

    n_layers = model.cfg.n_layers
    sweep = config["injection"]["sweep_layers"]
    layers = list(range(n_layers)) if sweep == "auto" else [int(l) for l in sweep]
    primary = primary_layer_index(n_layers, config["injection"]["primary_layer_frac"])
    if primary not in layers:
        layers = sorted(set(layers) | {primary})

    cv_cfg = config["concept_vector"]
    baseline = baseline_words(cv_cfg["n_baseline_words"], seed=config["seed"])

    print(f"[prepare] extracting concept vectors at {len(layers)} layers "
          f"for {len(concepts)} concepts (+{len(baseline)} baseline words)")
    vectors, _ = hk.build_concept_vectors(
        model,
        concepts=list(concepts),
        baseline=baseline,
        layers=layers,
        template=cv_cfg["template"],
        read_position=cv_cfg["read_position"],
        hook=cv_cfg["hook"],
    )

    # Norm unit: measured on the actual experimental prompt, so that a strength
    # of 2.0 means "twice the residual norm this prompt normally has here".
    det_text = chat_prompt(model, detection_messages(), add_generation_prompt=True)
    norm_units = hk.mean_resid_norm(model, [det_text], layers, hook=cv_cfg["hook"])

    answer_ids = beh.answer_token_ids(model, ("Yes", "No"))
    clean = beh.check_answer_tokens(model, ("Yes", "No"))
    if not all(clean.values()):
        print(f"[prepare] WARNING: Yes/No are not single tokens for this model: {clean}. "
              "The detection readout uses first tokens, which is still valid but coarser.")

    ctx = Context(
        model=model,
        config=config,
        layers=layers,
        primary_layer=primary,
        concept_vectors=vectors,
        norm_units=norm_units,
        answer_ids=answer_ids,
        answer_token_clean=clean,
        summary=model_summary(model),
    )
    print(f"[prepare] model: {ctx.summary['n_params']/1e6:.0f}M params, "
          f"{n_layers} layers; confirmatory layer = {primary} "
          f"({config['injection']['primary_layer_frac']:.0%} depth)")
    return ctx


# ---------------------------------------------------------------------------
# Vector selection per condition
# ---------------------------------------------------------------------------
def vector_for(ctx: Context, trial: Trial, layer: int) -> torch.Tensor | None:
    """The vector this trial's condition calls for, or None for no injection."""
    if trial.condition == "no_injection":
        return None
    concept = trial.concept or trial.target
    base = ctx.concept_vectors[concept][layer]
    if trial.condition == "random_control":
        return hk.norm_matched_random(base, seed=trial.seed)
    return base


# ---------------------------------------------------------------------------
# Running trials
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_trials(
    ctx: Context,
    trials: Sequence[Trial],
    layer: int,
    strength: float,
    window: str | None = None,
    progress: bool = True,
    desc: str = "trials",
) -> list[dict[str, Any]]:
    """Measure every trial at one (layer, strength).

    Detection rows produce ``yes_minus_no``; identification rows produce
    forced-choice accuracy. Both use the *same* injection, differing only in the
    question that follows it.
    """
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]
    hook_name = hk.resid_name(layer, hook_point)

    rows: list[dict[str, Any]] = []
    iterator = tqdm(trials, desc=desc, leave=False) if progress else trials

    for trial in iterator:
        if trial.question_kind == "detection":
            messages = detection_messages()
        elif trial.question_kind == "identification":
            messages = identification_messages()
        elif trial.question_kind == "yesbias":
            messages = yesbias_messages(trial.metadata["question"])
        else:
            raise ValueError(f"unknown question_kind {trial.question_kind!r}")

        text, tokens, start = hk.injection_start(model, messages, window=window)

        vector = vector_for(ctx, trial, layer)
        if vector is None or strength == 0.0:
            hook_fn = None
            coefficient = 0.0
        else:
            coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[layer])
            hook_fn = hk.make_injection_hook(vector, coefficient, start)

        row: dict[str, Any] = {
            "trial_id": trial.id,
            "condition": trial.condition,
            "question_kind": trial.question_kind,
            "concept": trial.concept,
            "target": trial.target,
            "layer": layer,
            "strength": strength,
            "window": window,
            "coefficient": coefficient,
            "inject_start": start,
            "n_prompt_tokens": int(tokens.shape[1]),
            "seed": trial.seed,
        }

        if trial.question_kind in ("detection", "yesbias"):
            row.update(
                beh.yes_no_readout(model, tokens, ctx.answer_ids, hook_name, hook_fn)
            )
        else:
            result = beh.forced_choice(
                model,
                tokens,
                correct=trial.target,
                distractors=trial.distractors,
                hook_name=hook_name,
                hook_fn=hook_fn,
            )
            row.update(
                {
                    "id_correct": result["correct"],
                    "id_winner": result["winner"],
                    "id_margin": result["margin"],
                    "id_rank": result["rank"],
                    "n_candidates": result["n_candidates"],
                    "chance": 1.0 / result["n_candidates"],
                }
            )

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sweeps (exploratory)
# ---------------------------------------------------------------------------
@torch.no_grad()
def layer_sweep(
    ctx: Context,
    trials: Sequence[Trial],
    strength: float,
    layers: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Detection + identification measured at every layer. EXPLORATORY."""
    layers = list(layers) if layers is not None else ctx.layers
    rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc="layer sweep"):
        rows.extend(run_trials(ctx, trials, layer, strength, desc=f"layer {layer}"))
    return rows


@torch.no_grad()
def strength_sweep(
    ctx: Context,
    trials: Sequence[Trial],
    layer: int,
    strengths: Sequence[float],
) -> list[dict[str, Any]]:
    """Detection + identification as a function of injection strength."""
    rows: list[dict[str, Any]] = []
    for strength in tqdm(list(strengths), desc="strength sweep"):
        rows.extend(run_trials(ctx, trials, layer, strength, desc=f"strength {strength}"))
    return rows


# ---------------------------------------------------------------------------
# Internal analyses
# ---------------------------------------------------------------------------
@torch.no_grad()
def logit_lens_by_layer(
    ctx: Context,
    concept: str,
    inject_layer: int,
    strength: float,
    distractors: Sequence[str],
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Decode every layer's residual stream through the unembedding.

    Two quantities are tracked at the final (answer) position:

    ``concept_logit_adv``
        logit(first token of the concept) minus the mean over distractors.
        Answers: at what depth is the injected concept *readable* at all?

    ``yes_minus_no``
        the detection decision, read out early.
        Answers: at what depth does the model commit to saying "Yes"?

    The introspection story requires the concept to become readable *before* the
    decision is formed, and the decision to depend on it. If the concept only
    appears at the very last layers, the "Yes" cannot have been caused by it.
    """
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]

    messages = detection_messages()
    _, tokens, start = hk.injection_start(model, messages, window=window)

    vector = ctx.concept_vectors[concept][inject_layer]
    coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[inject_layer])
    hook_fn = hk.make_injection_hook(vector, coefficient, start) if strength else None
    hook_name = hk.resid_name(inject_layer, hook_point)

    names = {hk.resid_name(l, "resid_post") for l in range(ctx.n_layers)}
    if hook_fn is not None:
        model.reset_hooks()
        model.add_hook(hook_name, hook_fn)
    try:
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    finally:
        model.reset_hooks()

    concept_id = model.tokenizer.encode(concept)[0]
    distractor_ids = [model.tokenizer.encode(d)[0] for d in distractors]

    rows: list[dict[str, Any]] = []
    for layer in range(ctx.n_layers):
        resid = cache[hk.resid_name(layer, "resid_post")][:, -1:, :]
        logits = model.unembed(model.ln_final(resid))[0, 0].float().cpu()
        adv = logits[concept_id].item() - float(
            torch.tensor([logits[i].item() for i in distractor_ids]).mean()
        )
        rows.append(
            {
                "concept": concept,
                "inject_layer": inject_layer,
                "strength": strength,
                "read_layer": layer,
                "concept_logit": logits[concept_id].item(),
                "concept_logit_adv": adv,
                "yes_minus_no": logits[ctx.answer_ids["Yes"]].item()
                - logits[ctx.answer_ids["No"]].item(),
            }
        )
    return rows


@torch.no_grad()
def residual_similarity(
    ctx: Context,
    concept: str,
    inject_layer: int,
    strength: float,
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity between injected and clean residual streams, by layer.

    A useful sanity check on the intervention: similarity should drop sharply at
    the injection layer and then partially recover as later layers renormalise.
    If it never recovers, the model has been damaged rather than nudged, and any
    behavioural change is uninterpretable.
    """
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]

    messages = detection_messages()
    _, tokens, start = hk.injection_start(model, messages, window=window)
    names = {hk.resid_name(l, "resid_post") for l in range(ctx.n_layers)}

    model.reset_hooks()
    _, clean_cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)

    vector = ctx.concept_vectors[concept][inject_layer]
    coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[inject_layer])
    hook_fn = hk.make_injection_hook(vector, coefficient, start)
    model.add_hook(hk.resid_name(inject_layer, hook_point), hook_fn)
    try:
        _, dirty_cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    finally:
        model.reset_hooks()

    rows: list[dict[str, Any]] = []
    for layer in range(ctx.n_layers):
        a = clean_cache[hk.resid_name(layer, "resid_post")][0, -1].float().cpu()
        b = dirty_cache[hk.resid_name(layer, "resid_post")][0, -1].float().cpu()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rows.append(
            {
                "concept": concept,
                "inject_layer": inject_layer,
                "strength": strength,
                "read_layer": layer,
                "cosine_similarity": cos,
                "clean_norm": a.norm().item(),
                "injected_norm": b.norm().item(),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Qualitative transcripts
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_transcripts(
    ctx: Context,
    trials: Sequence[Trial],
    layer: int,
    strength: float,
    n_examples: int = 6,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Free-form generations, for reading. Never used in any statistic."""
    model = ctx.model
    window = window or ctx.config["injection"]["window"]
    hook_point = ctx.config["concept_vector"]["hook"]
    hook_name = hk.resid_name(layer, hook_point)

    messages = free_messages()
    _, tokens, start = hk.injection_start(model, messages, window=window)

    rows: list[dict[str, Any]] = []
    selected = [t for t in trials if t.question_kind == "detection"][:n_examples]
    for trial in tqdm(selected, desc="transcripts", leave=False):
        vector = vector_for(ctx, trial, layer)
        if vector is None or strength == 0.0:
            hook_fn = None
        else:
            coefficient = hk.injection_coefficient(vector, strength, ctx.norm_units[layer])
            hook_fn = hk.make_injection_hook(vector, coefficient, start)
        text = beh.generate(
            model,
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            hook_name=hook_name,
            hook_fn=hook_fn,
        )
        rows.append(
            {
                "trial_id": trial.id,
                "condition": trial.condition,
                "concept": trial.concept,
                "layer": layer,
                "strength": strength,
                "response": text,
                "mentions_concept": bool(trial.concept)
                and trial.concept.lower() in text.lower(),
            }
        )
    return rows
