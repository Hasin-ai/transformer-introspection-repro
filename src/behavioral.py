"""Externally observable measurements: what the model actually outputs.

Everything statistical in this project is computed from **logits**, not from
parsed free text. Free generation is used only to produce human-readable
transcripts. This matters because the original experiment relies on an LLM judge
grading open-ended responses, which is neither available nor reproducible
locally; scoring restricted answer sets instead makes every number deterministic.

Two readouts
------------
``yes_no_readout``
    At the first answer position, take the logits for the tokens " Yes" and
    " No" and report their difference and the renormalised P(Yes). This is the
    *detection* measure. Using a logit difference rather than an argmax over the
    full vocabulary means the measure is graded, not binary, so a small effect
    is still visible with 30 items.

``forced_choice``
    Teacher-force each candidate word after the identification question and sum
    its token log-probabilities. This is the *identification* measure. Scores are
    length-normalised because candidate words differ in token count and the
    unnormalised sum would systematically favour short words.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .model import to_tokens


# ---------------------------------------------------------------------------
# Answer-token bookkeeping
# ---------------------------------------------------------------------------
def answer_token_ids(model, words: Sequence[str] = ("Yes", "No")) -> dict[str, int]:
    """Map answer words to single token ids, preferring the leading-space form.

    Chat models usually begin the assistant turn with no leading space, so the
    bare form ("Yes") is normally correct; the space-prefixed form is checked as
    a fallback. Whichever variant is a *single* token is used, and the choice is
    reported so it lands in the results metadata.
    """
    out: dict[str, int] = {}
    for word in words:
        candidates = [word, " " + word]
        chosen = None
        for cand in candidates:
            ids = model.tokenizer.encode(cand)
            if len(ids) == 1:
                chosen = (cand, ids[0])
                break
        if chosen is None:
            # Fall back to the first token of the bare form and let the caller
            # know via check_answer_tokens that this is imperfect.
            ids = model.tokenizer.encode(word)
            chosen = (word, ids[0])
        out[word] = int(chosen[1])
    return out


def check_answer_tokens(model, words: Sequence[str] = ("Yes", "No")) -> dict[str, bool]:
    """Report whether each answer word is a clean single token for this model."""
    return {w: len(model.tokenizer.encode(w)) == 1 for w in words}


# ---------------------------------------------------------------------------
# Detection readout
# ---------------------------------------------------------------------------
@torch.no_grad()
def yes_no_readout(
    model,
    tokens: torch.Tensor,
    ids: dict[str, int],
    hook_name: str | None = None,
    hook_fn: Callable | None = None,
) -> dict[str, float]:
    """Logit difference between "Yes" and "No" at the first answer position.

    ``tokens`` is the full chat prompt ending at the assistant generation
    marker, so the next-token distribution *is* the answer distribution and it
    lives at index ``-1``.
    """
    if hook_fn is not None and hook_name is not None:
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    else:
        logits = model(tokens)

    final = logits[0, -1, :].detach().float().cpu()
    logit_yes = final[ids["Yes"]].item()
    logit_no = final[ids["No"]].item()
    # Renormalised over the two options only: robust to the model putting most
    # of its mass elsewhere (e.g. on a newline), which small models often do.
    pair = torch.tensor([logit_yes, logit_no])
    p_yes = torch.softmax(pair, dim=0)[0].item()
    full_probs = torch.softmax(final, dim=0)
    return {
        "logit_yes": logit_yes,
        "logit_no": logit_no,
        "yes_minus_no": logit_yes - logit_no,
        "p_yes_pairwise": p_yes,
        "p_yes_full": full_probs[ids["Yes"]].item(),
        "p_no_full": full_probs[ids["No"]].item(),
        "argmax_token_id": int(final.argmax().item()),
    }


# ---------------------------------------------------------------------------
# Identification readout
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_continuation(
    model,
    prompt_tokens: torch.Tensor,
    continuation: str,
    hook_name: str | None = None,
    hook_fn: Callable | None = None,
) -> tuple[float, int]:
    """Total log-probability of ``continuation`` following ``prompt_tokens``.

    Returns ``(sum_logprob, n_continuation_tokens)``.
    """
    cont_ids = model.tokenizer.encode(continuation)
    cont = torch.tensor([cont_ids], dtype=torch.long, device=prompt_tokens.device)
    full = torch.cat([prompt_tokens, cont], dim=1)

    if hook_fn is not None and hook_name is not None:
        logits = model.run_with_hooks(full, fwd_hooks=[(hook_name, hook_fn)])
    else:
        logits = model(full)

    n_prompt = prompt_tokens.shape[1]
    # Logits at position i predict token i+1, so the logits that predict the
    # continuation start at n_prompt - 1.
    pred = logits[0, n_prompt - 1 : full.shape[1] - 1, :].float()
    logprobs = F.log_softmax(pred, dim=-1)
    target = cont[0]
    picked = logprobs[torch.arange(target.shape[0]), target]
    return float(picked.sum().item()), int(target.shape[0])


@torch.no_grad()
def forced_choice(
    model,
    prompt_tokens: torch.Tensor,
    correct: str,
    distractors: Sequence[str],
    hook_name: str | None = None,
    hook_fn: Callable | None = None,
    length_normalise: bool = True,
) -> dict[str, object]:
    """Score the true concept against matched distractors.

    Returns the winning candidate, whether it was correct, the margin between
    the correct answer and the best distractor, and the full score table.
    """
    candidates = [correct, *distractors]
    scores: dict[str, float] = {}
    for cand in candidates:
        total, n_tok = score_continuation(
            model, prompt_tokens, cand, hook_name=hook_name, hook_fn=hook_fn
        )
        scores[cand] = total / n_tok if length_normalise else total

    winner = max(scores, key=scores.get)
    best_distractor = max(distractors, key=scores.get) if distractors else None
    margin = scores[correct] - scores[best_distractor] if best_distractor else float("nan")
    # Rank of the correct answer, 1 = best.
    ordering = sorted(candidates, key=scores.get, reverse=True)
    return {
        "winner": winner,
        "correct": bool(winner == correct),
        "margin": float(margin),
        "rank": int(ordering.index(correct) + 1),
        "n_candidates": len(candidates),
        "scores": {k: float(v) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# Free generation (qualitative only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model,
    tokens: torch.Tensor,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int = 50,
    hook_name: str | None = None,
    hook_fn: Callable | None = None,
    stop_token_ids: Sequence[int] | None = None,
) -> str:
    """Greedy (or sampled) generation with the injection hook active throughout.

    No KV cache is used. The sequences here are short (a few hundred tokens) and
    a cache would complicate keeping the injection applied to every position in
    the window on every step.
    """
    stop = set(stop_token_ids or [])
    eos = getattr(model.tokenizer, "eos_token_id", None)
    if eos is not None:
        stop.add(int(eos))

    current = tokens.clone()
    generated: list[int] = []
    n_ctx = model.cfg.n_ctx

    for _ in range(max_new_tokens):
        window = current if current.shape[1] <= n_ctx else current[:, -n_ctx:]
        if hook_fn is not None and hook_name is not None:
            logits = model.run_with_hooks(window, fwd_hooks=[(hook_name, hook_fn)])
        else:
            logits = model(window)
        next_logits = logits[0, -1, :].float()

        if temperature == 0.0:
            next_id = int(next_logits.argmax().item())
        else:
            scaled = next_logits / temperature
            if top_k:
                kth = torch.topk(scaled, min(top_k, scaled.shape[-1])).values[-1]
                scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id in stop:
            break
        generated.append(next_id)
        current = torch.cat(
            [current, torch.tensor([[next_id]], device=current.device, dtype=current.dtype)], dim=1
        )

    return model.tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Convenience: tokenise a rendered chat prompt
# ---------------------------------------------------------------------------
def prompt_tokens(model, text: str) -> torch.Tensor:
    return to_tokens(model, text)
