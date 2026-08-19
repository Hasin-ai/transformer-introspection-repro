"""Configuration loading, result persistence, and the experiment pipeline.

Every artefact this project writes is accompanied by a ``*_meta.json`` recording
the config, the environment, and the model summary that produced it, so no
number in ``results/`` is ever orphaned from the conditions that made it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from .device import describe_environment, set_seed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_root() -> Path:
    """Repository root, inferred from this file's location."""
    return Path(__file__).resolve().parent.parent


def resolve(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = project_root() / config["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_rows(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write result rows to CSV (plus a JSONL copy) with a metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path.with_suffix(".csv"), index=False)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    meta: dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(frame),
        "columns": list(frame.columns),
    }
    if config is not None:
        meta["config"] = config
    if extra:
        meta.update(extra)
    path.with_name(path.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return path.with_suffix(".csv")


def save_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a processed table as CSV and as markdown next to it."""
    from .metrics import format_markdown

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(format_markdown(table), encoding="utf-8")
    return path.with_suffix(".csv")


def record_environment(config: dict[str, Any], device, model_summary: dict[str, Any]) -> dict[str, Any]:
    """Build the reproducibility record saved alongside every result file."""
    env = describe_environment(device, seed=config["seed"])
    return {"environment": env.to_dict(), "model": model_summary}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_context(config: dict[str, Any], trials=None):
    """Seed, build the dataset if needed, and prepare the experiment context."""
    from .introspection import prepare
    from .prompts import build_dataset, save_dataset

    set_seed(config["seed"])
    if trials is None:
        trials = build_dataset(
            n_concepts=config["design"]["n_concepts"],
            n_distractors=config["design"]["n_distractors"],
            conditions=config["design"]["conditions"],
            seed=config["seed"],
        )
        save_dataset(trials, resolve(config, "data") / "introspection_prompts.jsonl")

    concepts = sorted({t.concept or t.target for t in trials if t.condition != "no_injection"})
    # ``no_injection`` rows still carry a target concept for the identification
    # question, so make sure every concept referenced anywhere has a vector.
    concepts = sorted(set(concepts) | {t.target for t in trials if t.question_kind == "identification"})
    concepts = [c for c in concepts if c not in ("Yes", "No")]

    ctx = prepare(config, concepts)
    return ctx, trials
