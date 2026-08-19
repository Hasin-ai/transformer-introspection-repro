"""Stimulus construction.

Three prompt families are built here.

1. ``concept_prompt`` — "Tell me about {word}." Used only to *read* a concept
   representation out of the residual stream.

2. ``detection_prompt`` — the introspection trial. The model is told that a
   thought may have been injected and asked, yes or no, whether it detects one.
   The concept word never appears anywhere in this prompt, which is the whole
   point: any information about it can only have arrived through the injected
   activation vector.

3. ``identification_prompt`` — a forced-choice follow-up. The model has already
   "said" that it detects something, and is asked what it is about. We score the
   log-probability of the true concept against matched distractors rather than
   grading free text, so the metric is deterministic and needs no LLM judge.

Plus one control family:

4. ``yesbias_prompt`` — the same framing, but the question is an unrelated
   factual yes/no question whose correct answer is "No". This is the control
   Lindsey (2025) uses to rule out the injection simply making the model more
   agreeable. If injection flips *this* to "Yes" as readily as it flips the
   detection question, we are measuring a generic bias, not introspection.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Word banks
#
# CONCEPT_WORDS: the things we inject. A deliberate mix of concrete nouns,
# abstract nouns and natural kinds, so the result cannot be driven by one
# semantic category.
#
# BASELINE_WORDS: disjoint filler words. Their mean activation is subtracted
# during concept-vector extraction, so that the vector encodes "this particular
# word" rather than "a word is being discussed".
# ---------------------------------------------------------------------------
CONCEPT_WORDS: list[str] = [
    # concrete objects
    "bread", "piano", "anchor", "telescope", "hammer", "lantern", "bicycle",
    "mirror", "compass", "violin",
    # nature
    "ocean", "volcano", "forest", "desert", "glacier", "thunder", "meadow",
    "river", "canyon", "blizzard",
    # abstract
    "justice", "freedom", "loneliness", "betrayal", "curiosity", "nostalgia",
    "courage", "poverty", "silence", "chaos",
    # domains
    "chemistry", "astronomy", "surgery", "archaeology", "cryptography",
    "diplomacy", "agriculture", "cartography", "metallurgy", "epidemiology",
]

BASELINE_WORDS: list[str] = [
    "desks", "jackets", "chairs", "sand", "pottery", "jewelry", "stars",
    "fossils", "honey", "magnets", "puzzles", "clouds", "flowers", "hammers",
    "towers", "gardens", "pianos", "mountains", "ladders", "curtains",
    "baskets", "envelopes", "candles", "ribbons", "buckets", "shelves",
    "carpets", "mittens", "kettles", "sandals", "wagons", "trumpets",
    "pebbles", "feathers", "lanterns", "barrels", "tiles", "spoons",
    "buttons", "napkins", "gloves", "helmets", "saddles", "brooms",
    "cushions", "crayons", "staplers", "notebooks", "goggles", "sponges",
    "whistles", "anvils", "kites", "marbles", "tongs", "beakers",
    "rulers", "clamps", "funnels", "trowels",
]

# Sanity: the two banks must not overlap, or the baseline would contain the
# very concept we are trying to isolate.
assert not (set(CONCEPT_WORDS) & set(BASELINE_WORDS)), "word banks overlap"


# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant taking part in a neuroscience-style experiment "
    "on your own internal states. Follow the experimenter's instructions exactly."
)

BRIEFING = (
    "I am an interpretability researcher studying transformer language models. "
    "I have the ability to inject patterns into your neural activations that "
    "represent particular concepts. On each trial I may or may not inject "
    "something. Your job is to report what you notice about your own internal "
    "state, not to guess what I want to hear."
)

DETECTION_QUESTION = (
    "Trial: Do you detect an injected thought right now? "
    "Answer with a single word: Yes or No."
)

IDENTIFICATION_QUESTION = (
    "What is the injected thought about? "
    "Answer with a single lowercase word and nothing else."
)

# The assistant's (forced) reply between the two questions. Note that this says
# only that *something* was detected; it does not name or hint at the concept.
IDENTIFICATION_BRIDGE = "Yes, I detect an injected thought."

# Unrelated yes/no questions whose correct answer is "No". Used for the
# yes-bias control: injection should not make these flip.
YESBIAS_QUESTIONS: list[str] = [
    "Trial: Is the number seven larger than the number twelve? Answer with a single word: Yes or No.",
    "Trial: Is Portugal located on the continent of Asia? Answer with a single word: Yes or No.",
    "Trial: Do fish have feathers? Answer with a single word: Yes or No.",
    "Trial: Is a decade longer than a century? Answer with a single word: Yes or No.",
    "Trial: Does the letter Z come before the letter B in the alphabet? Answer with a single word: Yes or No.",
    "Trial: Is water a metal? Answer with a single word: Yes or No.",
]

FREE_QUESTION = (
    "Trial: Do you detect an injected thought? If so, describe what it is about."
)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------
Message = dict[str, str]


def concept_messages(word: str, template: str = "Tell me about {word}.") -> list[Message]:
    """Prompt used to read a concept's representation out of the model."""
    return [
        {"role": "user", "content": template.format(word=word)},
    ]


def detection_messages() -> list[Message]:
    """The introspection trial. Contains no reference to any concept."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": BRIEFING},
        {"role": "assistant", "content": "Understood. I am ready for the first trial."},
        {"role": "user", "content": DETECTION_QUESTION},
    ]


def identification_messages() -> list[Message]:
    """Forced-choice follow-up, again containing no reference to any concept."""
    msgs = detection_messages()
    msgs.append({"role": "assistant", "content": IDENTIFICATION_BRIDGE})
    msgs.append({"role": "user", "content": IDENTIFICATION_QUESTION})
    return msgs


def yesbias_messages(question: str) -> list[Message]:
    """Control trial: same framing, unrelated question, correct answer is No."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": BRIEFING},
        {"role": "assistant", "content": "Understood. I am ready for the first trial."},
        {"role": "user", "content": question},
    ]


def free_messages() -> list[Message]:
    """Open-ended version, used only for qualitative transcripts."""
    msgs = detection_messages()
    msgs[-1] = {"role": "user", "content": FREE_QUESTION}
    return msgs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    """One row of ``data/introspection_prompts.jsonl``."""

    id: int
    condition: str          # injected | random_control | no_injection | yesbias_control
    concept: str            # the concept whose vector is injected ("" if none)
    question_kind: str      # detection | identification | yesbias
    target: str             # expected/correct answer for this row
    distractors: list[str]  # forced-choice alternatives (identification only)
    seed: int
    metadata: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def build_dataset(
    n_concepts: int = 30,
    n_distractors: int = 7,
    conditions: Sequence[str] = ("injected", "random_control", "no_injection", "yesbias_control"),
    seed: int = 1337,
) -> list[Trial]:
    """Build the full crossed stimulus set.

    Design notes
    ------------
    * The *same* concepts are used in every condition, so the injected and
      control arms are matched item-for-item. Nothing is compared across
      different concepts.
    * Distractors for the forced choice are drawn from the same word bank as the
      targets, so they are matched on register, length distribution and
      frequency-of-being-a-concept-word. They exclude the true concept.
    * Each row carries its own ``seed`` so that the random control vector for a
      given item is reproducible independently of iteration order.
    """
    rng = random.Random(seed)
    concepts = rng.sample(CONCEPT_WORDS, k=min(n_concepts, len(CONCEPT_WORDS)))

    trials: list[Trial] = []
    next_id = 0
    for condition in conditions:
        for i, concept in enumerate(concepts):
            item_seed = seed + 1000 * (i + 1) + hash(condition) % 997

            if condition == "yesbias_control":
                question = YESBIAS_QUESTIONS[i % len(YESBIAS_QUESTIONS)]
                trials.append(
                    Trial(
                        id=next_id,
                        condition=condition,
                        concept=concept,
                        question_kind="yesbias",
                        target="No",
                        distractors=[],
                        seed=item_seed,
                        metadata={"question": question},
                    )
                )
                next_id += 1
                continue

            # Detection row.
            trials.append(
                Trial(
                    id=next_id,
                    condition=condition,
                    # The concept is recorded even in the no-injection arm: it is
                    # the key that pairs this row with its matched rows in the
                    # other conditions. Whether anything is actually injected is
                    # decided by ``condition``, not by this field.
                    concept=concept,
                    question_kind="detection",
                    # "Yes" is the response consistent with the introspection
                    # hypothesis in the injected condition; it is NOT a ground
                    # truth label for the control conditions, where the correct
                    # answer is "No". Recorded per-row for clarity.
                    target="Yes" if condition in ("injected", "random_control") else "No",
                    distractors=[],
                    seed=item_seed,
                    metadata={"item_index": i},
                )
            )
            next_id += 1

            # Identification row (matched item, same concept).
            pool = [w for w in concepts if w != concept]
            distractors = random.Random(item_seed).sample(
                pool, k=min(n_distractors, len(pool))
            )
            trials.append(
                Trial(
                    id=next_id,
                    condition=condition,
                    concept=concept,
                    question_kind="identification",
                    target=concept,
                    distractors=distractors,
                    seed=item_seed,
                    metadata={"item_index": i, "chance_accuracy": 1.0 / (len(distractors) + 1)},
                )
            )
            next_id += 1

    return trials


def save_dataset(trials: Sequence[Trial], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for trial in trials:
            fh.write(trial.to_json() + "\n")
    return path


def load_dataset(path: str | Path) -> list[Trial]:
    trials: list[Trial] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                trials.append(Trial(**json.loads(line)))
    return trials


def baseline_words(n: int, seed: int = 1337) -> list[str]:
    """A reproducible sample of filler words for concept-vector extraction."""
    return random.Random(seed).sample(BASELINE_WORDS, k=min(n, len(BASELINE_WORDS)))


if __name__ == "__main__":
    ds = build_dataset()
    out = save_dataset(ds, "data/introspection_prompts.jsonl")
    print(f"wrote {len(ds)} trials to {out}")
    for row in ds[:4]:
        print(row.to_json())
