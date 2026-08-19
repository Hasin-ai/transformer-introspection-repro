"""An educational trace of one forward pass, in the spirit of the Clean
Transformer Demo.

On the Clean Transformer Demo
-----------------------------
Neel Nanda's demo notebook builds a GPT-2-shaped transformer from scratch so you
can see every tensor. That notebook targets a 2022-era Easy-Transformer API and
learned absolute position embeddings; running it unchanged against a 2026 PyTorch
on macOS with a rotary-embedding model like Qwen would mean either pinning dead
packages or quietly describing an architecture the experiment does not use.

So the *philosophy* is kept — walk the residual stream, print every shape, never
treat the model as a black box — and the *mechanism* is TransformerLens's cache,
which exposes the same tensors for the model actually under test. The one real
architectural difference from the demo is called out below: Qwen uses rotary
position embeddings applied inside attention, so there is no separate
``hook_pos_embed`` to look at.

Run it:  ``python -m src.transformer_walkthrough``
"""

from __future__ import annotations

from typing import Any

import torch

from .hooks import pattern_name, resid_name
from .model import chat_prompt, load_model, to_tokens, token_strs


SHAPE_GUIDE = """
Tensor shapes in a decoder-only transformer
-------------------------------------------
  tokens              [batch, position]              integer token ids
  embed               [batch, position, d_model]     token -> vector lookup
  resid_pre  (L)      [batch, position, d_model]     residual stream entering block L
  ln1.normalized (L)  [batch, position, d_model]     RMSNorm'd copy attention reads
  q, k, v    (L)      [batch, position, head, d_head]  per-head queries/keys/values
  attn_scores(L)      [batch, head, query, key]      q·k / sqrt(d_head), causally masked
  pattern    (L)      [batch, head, query, key]      softmax over keys; rows sum to 1
  z          (L)      [batch, position, head, d_head]  pattern-weighted values
  attn_out   (L)      [batch, position, d_model]     attention's write to the stream
  resid_mid  (L)      [batch, position, d_model]     resid_pre + attn_out
  mlp_out    (L)      [batch, position, d_model]     MLP's write to the stream
  resid_post (L)      [batch, position, d_model]     resid_mid + mlp_out  <- INJECTION SITE
  ln_final.normalized [batch, position, d_model]     final normalisation
  logits              [batch, position, d_vocab]     unembedding of the final stream

The residual stream is the spine. Every block *reads* a normalised copy of it and
*adds* its output back. That additive structure is exactly why concept injection
works at all: adding a vector to `resid_post` at layer L is indistinguishable, to
every later layer, from block L having written that vector itself.

The three tensors this experiment actually depends on:

  resid_post(L)   read to build a concept vector; written to inject it
  logits          read at the answer position for the Yes/No decision
  resid_post(l)   for every l, decoded through the unembedding for the logit lens
"""


def walkthrough(model, text: str | None = None, layer: int | None = None) -> dict[str, Any]:
    """Run one forward pass and print every intermediate shape.

    Returns a dict of the interesting tensors so a notebook can poke at them.
    """
    if text is None:
        text = chat_prompt(
            model, [{"role": "user", "content": "The capital of France is"}]
        )
    layer = layer if layer is not None else model.cfg.n_layers // 2

    tokens = to_tokens(model, text)
    strs = token_strs(model, tokens)

    logits, cache = model.run_with_cache(tokens)

    print(SHAPE_GUIDE)
    print("=" * 78)
    print("PROMPT")
    print("=" * 78)
    print(repr(text[:400]) + ("..." if len(text) > 400 else ""))
    print(f"\n{len(strs)} tokens; last 12: {strs[-12:]}")

    print("\n" + "=" * 78)
    print(f"FORWARD PASS  (showing block {layer} of {model.cfg.n_layers})")
    print("=" * 78)

    def show(label: str, tensor: torch.Tensor, note: str) -> None:
        print(f"  {label:<26} {str(tuple(tensor.shape)):<28} {note}")

    show("tokens", tokens, "integer ids")
    show("embed", cache["hook_embed"], "token embeddings")
    show(f"blocks.{layer}.resid_pre", cache[resid_name(layer, "resid_pre")],
         "stream entering the block")
    show(f"blocks.{layer}.ln1.normalized", cache[f"blocks.{layer}.ln1.hook_normalized"],
         "what attention reads")
    show(f"blocks.{layer}.attn.q", cache[f"blocks.{layer}.attn.hook_q"], "queries, per head")
    show(f"blocks.{layer}.attn.k", cache[f"blocks.{layer}.attn.hook_k"], "keys, per head")
    show(f"blocks.{layer}.attn.v", cache[f"blocks.{layer}.attn.hook_v"], "values, per head")
    show(f"blocks.{layer}.attn.scores", cache[f"blocks.{layer}.attn.hook_attn_scores"],
         "pre-softmax, causally masked")
    show(f"blocks.{layer}.attn.pattern", cache[pattern_name(layer)], "post-softmax probabilities")
    show(f"blocks.{layer}.attn.z", cache[f"blocks.{layer}.attn.hook_z"], "weighted values")
    show(f"blocks.{layer}.attn_out", cache[f"blocks.{layer}.hook_attn_out"],
         "attention's write-back")
    show(f"blocks.{layer}.resid_mid", cache[resid_name(layer, "resid_mid")],
         "after attention")
    show(f"blocks.{layer}.mlp_out", cache[f"blocks.{layer}.hook_mlp_out"], "MLP's write-back")
    show(f"blocks.{layer}.resid_post", cache[resid_name(layer, "resid_post")],
         "<- concept injection happens here")
    show("ln_final.normalized", cache["ln_final.hook_normalized"], "final normalisation")
    show("logits", logits, "unembedded")

    # Sanity check the additive structure. If this fails, the mental model above
    # is wrong for this architecture and the injection reasoning does not hold.
    pre = cache[resid_name(layer, "resid_pre")]
    mid = cache[resid_name(layer, "resid_mid")]
    post = cache[resid_name(layer, "resid_post")]
    attn = cache[f"blocks.{layer}.hook_attn_out"]
    mlp = cache[f"blocks.{layer}.hook_mlp_out"]
    err_mid = (pre + attn - mid).abs().max().item()
    err_post = (mid + mlp - post).abs().max().item()
    print("\n  residual-stream identity checks (should be ~0):")
    print(f"    max |resid_pre + attn_out - resid_mid|   = {err_mid:.3e}")
    print(f"    max |resid_mid + mlp_out - resid_post|   = {err_post:.3e}")

    print("\n  residual-stream norm by layer (why injection strength is norm-relative):")
    for l in range(model.cfg.n_layers):
        norm = cache[resid_name(l, "resid_post")][0, -1].norm().item()
        bar = "#" * max(1, int(28 * norm / max(
            cache[resid_name(k, "resid_post")][0, -1].norm().item()
            for k in range(model.cfg.n_layers)
        )))
        print(f"    layer {l:>2}  ||resid_post[-1]|| = {norm:9.2f}  {bar}")

    print("\n" + "=" * 78)
    print("NEXT-TOKEN PREDICTION")
    print("=" * 78)
    final = logits[0, -1].float()
    top = torch.topk(final, k=8)
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        piece = model.tokenizer.convert_ids_to_tokens([idx])[0]
        prob = torch.softmax(final, dim=-1)[idx].item()
        print(f"  {piece!r:<20} logit {score:8.3f}   p {prob:.4f}")

    return {
        "tokens": tokens,
        "token_strs": strs,
        "logits": logits,
        "cache": cache,
        "layer": layer,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    m = load_model(args.model, dtype=args.dtype)
    walkthrough(m, text=args.text, layer=args.layer)
