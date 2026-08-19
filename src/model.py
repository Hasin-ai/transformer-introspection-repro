"""Model loading and tokenisation helpers.

Why TransformerLens
-------------------
The experiment needs to read and write the residual stream at a named layer,
mid-forward-pass, without reimplementing the model. TransformerLens gives every
internal tensor a stable name (``blocks.14.hook_resid_post``) and lets you
register hooks that *return a modified tensor*, which is exactly the primitive
concept injection needs. Doing this with raw Hugging Face modules is possible
(the old ``introspection.py`` in this repo's history did it) but the hook points
are architecture-specific and easy to get subtly wrong.

Why Qwen2.5-0.5B-Instruct
-------------------------
The introspection task is a *chat* task: the model has to be told the rules of
an experiment and then answer a question about its own state. A base LM like
GPT-2 cannot do this at all — it will simply continue the text. So an
instruction-tuned model is required, and Qwen2.5-0.5B-Instruct is the smallest
widely used one that (a) TransformerLens supports natively, (b) fits in ~2 GB of
unified memory in float32, and (c) reliably emits "Yes"/"No" when asked to.
It is roughly *three orders of magnitude* smaller than the models in Lindsey
(2025); see README for what that means for interpreting a null result.

Offline stub
------------
``model.name: random-tiny`` builds a randomly initialised model with the same
architecture family and a toy tokenizer. Nothing is downloaded. Every code path
in this repository runs. The numbers are meaningless by construction, which is
the point: it is a plumbing test, not an experiment.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import torch

from .device import get_device


# ---------------------------------------------------------------------------
# Offline stub tokenizer
# ---------------------------------------------------------------------------
class StubTokenizer:
    """A deterministic whitespace tokenizer used only by ``random-tiny``.

    It implements the small slice of the Hugging Face tokenizer API that this
    project actually calls. It is *not* a general-purpose tokenizer.
    """

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.padding_side = "right"
        self.truncation_side = "right"
        self.chat_template = "stub"
        self._id_to_str: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    # -- core ---------------------------------------------------------------
    def _piece_id(self, piece: str) -> int:
        digest = hashlib.md5(piece.encode("utf-8")).hexdigest()
        # Reserve 0/1 for pad/eos.
        token_id = 2 + int(digest, 16) % (self.vocab_size - 2)
        self._id_to_str.setdefault(token_id, piece)
        return token_id

    def encode(self, text: str, **_: Any) -> list[int]:
        pieces = text.replace("\n", " \n ").split(" ")
        return [self._piece_id(p) for p in pieces if p != ""]

    def decode(self, ids: Iterable[int], **_: Any) -> str:
        return " ".join(self._id_to_str.get(int(i), "<unk>") for i in ids)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self._id_to_str.get(int(i), "<unk>") for i in ids]

    def __call__(self, text: str, return_tensors: str | None = None, **_: Any):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    # -- chat ---------------------------------------------------------------
    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(
    model_name: str,
    dtype: str = "float32",
    device: torch.device | str | None = None,
    **device_kwargs: Any,
):
    """Load a ``HookedTransformer`` (or the offline stub) in eval mode.

    Returns the model. ``model.cfg`` carries ``n_layers`` / ``d_model`` etc.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    if device is None:
        device = get_device(**device_kwargs)
    device = torch.device(device)
    torch_dtype = _DTYPES[dtype]

    if model_name == "random-tiny":
        cfg = HookedTransformerConfig(
            n_layers=6,
            d_model=128,
            n_ctx=512,
            d_head=32,
            n_heads=4,
            d_mlp=256,
            act_fn="silu",
            gated_mlp=True,
            normalization_type="RMS",
            positional_embedding_type="rotary",
            rotary_dim=32,
            d_vocab=2048,
            device=str(device),
            dtype=torch_dtype,
        )
        model = HookedTransformer(cfg)
        # Attach the stub directly: HookedTransformer.set_tokenizer expects a
        # real HF tokenizer, and nothing in this project calls model.to_tokens.
        model.tokenizer = StubTokenizer(vocab_size=cfg.d_vocab)
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=str(device),
            dtype=torch_dtype,
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def model_summary(model) -> dict[str, Any]:
    """A small dict describing the loaded model, saved alongside results."""
    cfg = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": getattr(cfg, "model_name", "unknown"),
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "d_mlp": cfg.d_mlp,
        "d_vocab": cfg.d_vocab,
        "n_ctx": cfg.n_ctx,
        "n_params": int(n_params),
        "dtype": str(cfg.dtype),
        "device": str(cfg.device),
    }


# ---------------------------------------------------------------------------
# Tokenisation helpers
#
# Everything in this project tokenises through these three functions so that
# the real Qwen tokenizer and the offline stub are interchangeable.
# ---------------------------------------------------------------------------
def chat_prompt(
    model,
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Render a list of ``{"role", "content"}`` messages to a prompt string."""
    tok = model.tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", True):
        return tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )
    # Fallback for base models with no chat template (e.g. plain gpt2).
    rendered = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    return rendered + ("\nAssistant:" if add_generation_prompt else "")


def to_tokens(model, text: str, device: torch.device | str | None = None) -> torch.Tensor:
    """Tokenise ``text`` to a ``[1, seq]`` LongTensor on the model's device.

    Note: the chat template already inserts whatever BOS/turn markers the model
    expects, so no extra special tokens are added here. Adding them twice is a
    classic source of off-by-one position bugs.
    """
    device = device if device is not None else model.cfg.device
    encoded = model.tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return ids.to(device)


def token_strs(model, tokens: torch.Tensor) -> list[str]:
    """Human-readable string for each token id in a ``[1, seq]`` tensor."""
    ids = tokens[0].tolist() if tokens.ndim == 2 else tokens.tolist()
    return list(model.tokenizer.convert_ids_to_tokens(ids))


def single_token_id(model, text: str) -> int:
    """First token id of ``text``.

    Used for the Yes/No readout. Callers must verify that the words they care
    about really are single tokens for the model in question — see
    ``introspection.check_answer_tokens``.
    """
    ids = model.tokenizer.encode(text)
    if len(ids) == 0:
        raise ValueError(f"{text!r} tokenised to nothing")
    return int(ids[0])
