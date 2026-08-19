"""Experiment 01 — transformer mechanics.

Nothing scientific happens here. It exists so that, before any claim is made
about a model's internal states, you have seen those internal states with your
own eyes: every tensor in one forward pass, the residual-stream identity
verified numerically, and the norm growth across depth that motivates
norm-relative injection strength.

Run:  python experiments/01_transformer_basics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.device import get_device, set_seed                     # noqa: E402
from src.model import load_model                                 # noqa: E402
from src.runner import load_config, project_root, record_environment  # noqa: E402
from src.transformer_walkthrough import walkthrough              # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "config.yaml"))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device(
        prefer=config["device"]["prefer"],
        force_cpu=config["device"]["force_cpu"],
        verify=config["device"]["verify"],
        mps_tolerance=config["device"]["mps_tolerance"],
    )

    model = load_model(config["model"]["name"], dtype=config["model"]["dtype"], device=device)
    from src.model import model_summary

    record = record_environment(config, device, model_summary(model))
    print("\nEnvironment")
    for key, value in record["environment"].items():
        print(f"  {key:24s} {value}")
    print("\nModel")
    for key, value in record["model"].items():
        print(f"  {key:24s} {value}")
    print()

    walkthrough(model, text=args.text, layer=args.layer)


if __name__ == "__main__":
    main()
