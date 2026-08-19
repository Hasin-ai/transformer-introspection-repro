"""Reading and writing the residual stream.

This is the mechanical core of the project. Three things happen here:

* **Reading** a concept representation out of the residual stream.
* **Building** an injection vector (concept, or a norm-matched random control).
* **Writing** that vector back into the residual stream during a forward pass,
  at a controlled set of token positions.

Hook naming
-----------
TransformerLens names every internal tensor. The ones used here:

    blocks.{L}.hook_resid_pre    [batch, pos, d_model]  input to block L
    blocks.{L}.hook_resid_mid    [batch, pos, d_model]  after attention
    blocks.{L}.hook_resid_post   [batch, pos, d_model]  after MLP = output of L
    blocks.{L}.hook_attn_out     [batch, pos, d_model]  attention's contribution
    blocks.{L}.hook_mlp_out      [batch, pos, d_model]  MLP's contribution
    blocks.{L}.attn.hook_pattern [batch, head, q, k]    attention probabilities

We read from and write to ``hook_resid_post``, i.e. the residual stream as it
leaves block L. This matches "inject at layer L" in the original work.

Injection strength
------------------
Strength is expressed in units of the *mean residual-stream norm at the
injection layer*:

    h[pos] <- h[pos] + strength * (mean_norm_L / ||v||) * v

The original paper multiplies its vector by a raw scalar. Norm-relative scaling
is used here because the residual-stream norm in a small model grows by an order
of magnitude across depth, so a fixed raw scalar would mean something completely
different at layer 4 than at layer 20 — and this experiment sweeps layers. The
choice is documented as a deviation in the README.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import torch

from .model import chat_prompt, to_tokens


# ---------------------------------------------------------------------------
# Hook names
# ---------------------------------------------------------------------------
def resid_name(layer: int, hook: str = "resid_post") -> str:
    return f"blocks.{layer}.hook_{hook}"


def component_name(layer: int, component: str) -> str:
    """``attn_out`` / ``mlp_out`` -> full TransformerLens hook name."""
    return f"blocks.{layer}.hook_{component}"


def pattern_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"


# ---------------------------------------------------------------------------
# Reading activations
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_resid(
    model,
    tokens: torch.Tensor,
    layers: Sequence[int] | None = None,
    hook: str = "resid_post",
) -> dict[int, torch.Tensor]:
    """Run a forward pass and return ``{layer: [seq, d_model]}`` (batch 1)."""
    layers = list(range(model.cfg.n_layers)) if layers is None else list(layers)
    wanted = {resid_name(l, hook) for l in layers}
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    return {l: cache[resid_name(l, hook)][0].detach().float().cpu() for l in layers}


@torch.no_grad()
def read_concept_activations(
    model,
    word: str,
    template: str,
    layers: Sequence[int],
    read_position: int = -1,
    hook: str = "resid_post",
) -> dict[int, torch.Tensor]:
    """Residual stream at one token position of "Tell me about {word}."

    Returns ``{layer: [d_model]}``.
    """
    from .prompts import concept_messages

    text = chat_prompt(model, concept_messages(word, template), add_generation_prompt=True)
    tokens = to_tokens(model, text)
    per_layer = cache_resid(model, tokens, layers=layers, hook=hook)
    return {l: acts[read_position].clone() for l, acts in per_layer.items()}


@torch.no_grad()
def mean_resid_norm(
    model,
    texts: Iterable[str],
    layers: Sequence[int],
    hook: str = "resid_post",
) -> dict[int, float]:
    """Mean L2 norm of the residual stream per layer, averaged over positions.

    Used as the unit for injection strength.
    """
    totals = {l: 0.0 for l in layers}
    count = 0
    for text in texts:
        tokens = to_tokens(model, text)
        per_layer = cache_resid(model, tokens, layers=layers, hook=hook)
        for l, acts in per_layer.items():
            totals[l] += acts.norm(dim=-1).mean().item()
        count += 1
    return {l: totals[l] / max(count, 1) for l in layers}


# ---------------------------------------------------------------------------
# Building injection vectors
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_concept_vectors(
    model,
    concepts: Sequence[str],
    baseline: Sequence[str],
    layers: Sequence[int],
    template: str = "Tell me about {word}.",
    read_position: int = -1,
    hook: str = "resid_post",
    progress: bool = True,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[int, torch.Tensor]]:
    """Extract ``v_word = act(word) - mean_over_baseline(act(filler))`` per layer.

    Subtracting the baseline mean is what turns "the activation while discussing
    a word" into "the activation *specific to this word*". Without it the vector
    is dominated by the shared prompt template and injecting it would mostly
    push the model toward "someone asked me to describe something".

    Returns ``(vectors[word][layer], baseline_mean[layer])``.
    """
    iterator = baseline
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(baseline, desc="baseline activations", leave=False)

    sums: dict[int, torch.Tensor] = {}
    n = 0
    for word in iterator:
        acts = read_concept_activations(model, word, template, layers, read_position, hook)
        for l, vec in acts.items():
            sums[l] = vec.clone() if l not in sums else sums[l] + vec
        n += 1
    baseline_mean = {l: sums[l] / n for l in sums}

    iterator = concepts
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(concepts, desc="concept activations", leave=False)

    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for word in iterator:
        acts = read_concept_activations(model, word, template, layers, read_position, hook)
        vectors[word] = {l: acts[l] - baseline_mean[l] for l in layers}
    return vectors, baseline_mean


def norm_matched_random(vector: torch.Tensor, seed: int) -> torch.Tensor:
    """A random Gaussian direction with the same L2 norm as ``vector``.

    This is the key control for "the model says Yes because *something* was
    added to its residual stream". If the random control produces the same
    detection rate, the effect is about perturbation magnitude, not content.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    return noise / (noise.norm() + 1e-8) * vector.norm()


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def make_injection_hook(
    vector: torch.Tensor,
    coefficient: float,
    start_position: int,
) -> Callable:
    """Return a TransformerLens forward hook that adds ``coefficient * vector``.

    The hook adds to every position from ``start_position`` onward, *including*
    positions generated later in the sequence, because the hook fires on every
    forward pass.

    Implementation notes
    --------------------
    * The activation tensor is cloned rather than mutated in place. In-place
      mutation of a hooked tensor is a well-known source of corrupted caches and
      silently wrong gradients.
    * The vector is cast to the activation's dtype/device inside the hook, so a
      float32 vector works with a bfloat16 model.
    """

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001
        seq_len = activation.shape[1]
        if start_position >= seq_len:
            return activation
        vec = vector.to(device=activation.device, dtype=activation.dtype)
        out = activation.clone()
        out[:, start_position:, :] = out[:, start_position:, :] + coefficient * vec.view(1, 1, -1)
        return out

    return hook_fn


def make_ablation_hook(mode: str = "zero", mean_value: torch.Tensor | None = None) -> Callable:
    """Zero- or mean-ablate a component's output at all positions."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001
        if mode == "zero":
            return torch.zeros_like(activation)
        if mean_value is None:
            raise ValueError("mean ablation requires mean_value")
        return mean_value.to(activation.device, activation.dtype).expand_as(activation).clone()

    return hook_fn


def injection_coefficient(
    vector: torch.Tensor,
    strength: float,
    layer_norm_unit: float,
) -> float:
    """Convert a strength in 'residual norms' into a raw multiplier on ``vector``."""
    if strength == 0.0:
        return 0.0
    return float(strength) * float(layer_norm_unit) / float(vector.norm().item() + 1e-8)


@contextmanager
def injected(model, layer: int, hook_fn: Callable | None, hook: str = "resid_post"):
    """Context manager that installs an injection hook for its duration."""
    if hook_fn is None:
        yield
        return
    name = resid_name(layer, hook)
    model.add_hook(name, hook_fn)
    try:
        yield
    finally:
        model.reset_hooks()


# ---------------------------------------------------------------------------
# Injection window
# ---------------------------------------------------------------------------
def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the shared leading token ids of two ``[1, seq]`` tensors.

    Used instead of ``len(tokenize(prefix))`` because tokenizers are not
    guaranteed to be prefix-consistent: tokenising a prefix and tokenising the
    whole string can disagree at the boundary, which would silently shift the
    injection window by a token.
    """
    ids_a, ids_b = a[0].tolist(), b[0].tolist()
    limit = min(len(ids_a), len(ids_b))
    i = 0
    while i < limit and ids_a[i] == ids_b[i]:
        i += 1
    return i


def injection_start(
    model,
    messages: Sequence[dict[str, str]],
    window: str = "trial",
) -> tuple[str, torch.Tensor, int]:
    """Render a conversation and work out where the injection window starts.

    Returns ``(prompt_text, tokens, start_position)``.

    Windows
    -------
    ``"all"``     position 0 — every token is injected into.
    ``"trial"``   from the first token of the final user turn. This matches the
                  original experiment, where the injection covers the trial.
    ``"answer"``  from the assistant's generation-prompt marker onwards, so the
                  *question tokens themselves* are untouched. This is the
                  robustness variant: it rules out the model reacting to a
                  corrupted reading of the question rather than to its own state.
    """
    full_text = chat_prompt(model, messages, add_generation_prompt=True)
    full_tokens = to_tokens(model, full_text)

    if window == "all":
        return full_text, full_tokens, 0

    if window == "answer":
        prefix_text = chat_prompt(model, messages, add_generation_prompt=False)
    elif window == "trial":
        prefix_text = chat_prompt(model, list(messages)[:-1], add_generation_prompt=False)
    else:
        raise ValueError(f"unknown injection window {window!r}")

    prefix_tokens = to_tokens(model, prefix_text)
    start = common_prefix_len(full_tokens, prefix_tokens)
    # Guard against a degenerate template that produces an empty or full window.
    start = max(0, min(start, full_tokens.shape[1] - 1))
    return full_text, full_tokens, start
