"""Device selection, seeding, and environment capture.

The point of this module is that nothing else in the codebase should ever
mention "mps" or "cuda" directly. Everything asks for a device once, here.

Apple MPS notes
---------------
MPS is fast but historically has had gaps: some reductions, some indexing
patterns, and some dtypes either fall over or silently return wrong numbers.
Two mitigations are provided:

1. ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set on import. Any operation without an
   MPS kernel then silently runs on the CPU instead of raising. This is a
   correctness/robustness win and costs only a little speed.

2. ``verify_device`` runs a small forward-pass-like computation on both the
   selected device and the CPU and compares. If they disagree beyond tolerance
   the caller is told to fall back to CPU. This experiment is small enough that
   CPU is a perfectly acceptable fallback (minutes, not hours).
"""

from __future__ import annotations

import os

# Must be set BEFORE torch is imported for the fallback to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import platform
import random
import sys
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all backends)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS reads from the global torch seed, so no separate call is needed.
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device(prefer: str = "auto", force_cpu: bool = False) -> torch.device:
    """Choose a torch device.

    Parameters
    ----------
    prefer
        ``"auto"``, or an explicit ``"mps"`` / ``"cuda"`` / ``"cpu"``.
    force_cpu
        Short-circuit to CPU regardless of what is available.
    """
    if force_cpu:
        return torch.device("cpu")

    if prefer != "auto":
        return torch.device(prefer)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def verify_device(device: torch.device, tolerance: float = 0.02) -> tuple[bool, float]:
    """Check that the device agrees with the CPU on a representative workload.

    The workload deliberately mixes the operations this project depends on:
    matmul, softmax, RMS-style normalisation, gather, and a masked reduction.
    Returns ``(ok, max_abs_difference)``.
    """
    if device.type == "cpu":
        return True, 0.0

    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    w = torch.randn(64, 64)

    def workload(t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        h = t @ weight
        h = h / (h.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)   # RMSNorm-ish
        scores = h @ h.transpose(-1, -2)
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = torch.softmax(scores, dim=-1)
        out = pattern @ h
        return out.sum(dim=1)

    ref = workload(x, w)
    try:
        got = workload(x.to(device), w.to(device)).cpu()
    except Exception:                                   # noqa: BLE001 - any backend error
        return False, float("inf")

    diff = (ref - got).abs().max().item()
    return diff <= tolerance, diff


def get_device(
    prefer: str = "auto",
    force_cpu: bool = False,
    verify: bool = True,
    mps_tolerance: float = 0.02,
    quiet: bool = False,
) -> torch.device:
    """Select a device and, optionally, validate it, falling back to CPU."""
    device = select_device(prefer=prefer, force_cpu=force_cpu)
    if verify:
        ok, diff = verify_device(device, tolerance=mps_tolerance)
        if not ok:
            if not quiet:
                print(
                    f"[device] {device} failed the numerical self-test "
                    f"(max abs diff {diff:.4g} > {mps_tolerance}); falling back to CPU."
                )
            device = torch.device("cpu")
        elif not quiet and device.type != "cpu":
            print(f"[device] {device} passed the numerical self-test (max abs diff {diff:.2e}).")
    if not quiet:
        print(f"[device] using {device}")
    return device


# ---------------------------------------------------------------------------
# Environment capture (for reproducibility records)
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentInfo:
    python_version: str
    platform: str
    machine: str
    torch_version: str
    transformer_lens_version: str
    transformers_version: str
    numpy_version: str
    device: str
    mps_available: bool
    cuda_available: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:                                   # noqa: BLE001
        return "unknown"


def describe_environment(device: torch.device, seed: int) -> EnvironmentInfo:
    """Capture everything needed to reproduce a run."""
    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        transformer_lens_version=_pkg_version("transformer-lens"),
        transformers_version=_pkg_version("transformers"),
        numpy_version=np.__version__,
        device=str(device),
        mps_available=bool(torch.backends.mps.is_available()),
        cuda_available=bool(torch.cuda.is_available()),
        seed=seed,
    )


if __name__ == "__main__":
    set_seed(0)
    dev = get_device()
    info = describe_environment(dev, seed=0)
    for key, value in info.to_dict().items():
        print(f"{key:28s} {value}")
