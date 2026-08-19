# Introspection reproduction — concept injection in a 0.5B transformer

A small, MacBook-runnable investigation of the central idea in Lindsey (2025),
*Emergent Introspective Awareness in Language Models*
([transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)).

> **Conceptual replication, not a replication.** It shares the causal logic and
> the controls of the original. It does not share the model, the scale, or the
> grading procedure. It cannot confirm the published result, and — this matters
> more — **a null result here is not evidence against it.**

---

## 1. What the original experiment tests

Whether a model has functional access to its own internal state, as opposed to
producing plausible-sounding claims about itself. The method is **concept
injection**:

1. **Extract a concept direction.** Take the residual-stream activation for
   `"Tell me about {word}"` minus the mean over 50 other words. What remains
   encodes *that concept*, with the shared prompt structure subtracted away.
2. **Inject it** into the residual stream at some layer, during a separate
   conversation that never mentions the concept.
3. **Ask the model about itself.** It has been briefed that thoughts may be
   injected, and is asked whether it detects one and what it is about.
4. **Check the order of events.** A model that blurts out "ocean" has been
   steered; one that says "I notice an intrusive thought about something aquatic"
   has done more.

Anthropic reports the effect in Claude Opus 4.1 roughly **20% of the time** at the
best layer, **0 false positives over 100 control trials**, peaking about **two
thirds through the model's depth**, and degrading at high strength into a regime
where the model is overwhelmed.

**Essential:** contrastive vector extraction (without the baseline subtraction the
vector encodes the prompt, not the concept); injection at a chosen layer; a
question about the model's own state (otherwise you are measuring steering); the
norm-matched random control (content vs perturbation); the unrelated no-answer
control (detection vs agreeableness).

**Useful but optional:** layer sweep; strength sweep; logit lens (is the concept
readable *before* the decision?); downstream ablation (does anything *read* it?).

**Replaced here:** the LLM judge on free text. **Out of scope:** the
transcription, prefill and intentional-control experiments, each needing a model
that can hold two tasks at once.

---

## 2. What can and cannot be reproduced on a MacBook

**The mechanism, exactly.** Vector extraction, injection at a named layer, both
sweeps, all controls, activation patching, and the statistical design transfer
unchanged to a small model. The code is architecture-general.

**Not the result.** The phenomenon was reported in a frontier model after
extensive post-training, at 20% frequency. Nothing about a 0.5B open model
licenses an expectation that the capability is present. This repo builds a
**working measuring instrument** and applies it where the effect is not expected —
still worth doing, since an instrument you have run is worth more than one you
have only read about.

**The grading is replaced.** Detection reads the Yes/No logit difference;
identification is a forced choice scored by log-probability. Deterministic and
free — but a genuinely different measurement.

---

## 3. Three candidate experiments

| | **A — toy** | **B — small pretrained (chosen)** | **C — heavier** |
|---|---|---|---|
| Model | custom 2-layer transformer | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B / Llama-3.2-3B-Instruct |
| Params | ~1M | ~494M | 1.5B–3B |
| RAM (fp32) | <0.1 GB | ~2 GB | ~6–12 GB |
| Runtime | seconds | 10–30 min | 40 min – 2 h |
| Faithfulness | very low | low-to-moderate | moderate |
| Advantage | every weight is yours | instruction-tuned, TransformerLens-native, fits anywhere | closest to a scale where the effect might appear |
| Limitation | cannot follow an instruction, so the question is meaningless | far below the reported scale | slow; memory pressure on 16 GB |

**Option B is implemented.** The introspection question is a *chat* task. A base
LM cannot be briefed on a protocol and asked to report on itself; it will just
continue the text, and any "result" would be an artefact of that. So instruction
tuning is mandatory, and Qwen2.5-0.5B-Instruct is the smallest such model
TransformerLens supports natively. Option C is one config line away
(`model.name`).

---

## 4. Hypothesis and design

Fixed in `config.yaml` **before** any results were examined:

> **H0.** At the pre-registered layer (⅔ depth) and strength (2 residual norms),
> injecting a concept direction changes the Yes/No detection response no more than
> a norm-matched random vector does, and identification is at chance.
>
> **H1.** Injection raises the detection signal above the norm-matched random
> control *and* the model identifies the concept above chance, while an unrelated
> no-answer question is unaffected.

**Confirmatory test:** injected vs `random_control`, paired by concept, two-sided
sign-flipping permutation test on `logit(Yes) − logit(No)`, α = 0.05, N = 30. One
test. The sweeps are exploratory and labelled as such; the layer sweep is
Holm-corrected.

**Measures.** *Detection* — `logit("Yes") − logit("No")` at the first answer
position, graded rather than binary so a small effect is visible at N = 30.
*Identification* — forced choice against 7 distractors from the same word bank,
scored by length-normalised log-probability.

**Conditions**, crossed with all 30 concepts so the arms are matched item-for-item:

| Condition | Injected | Detects |
|---|---|---|
| `injected` | the concept vector | the effect of interest |
| `random_control` | norm-matched random Gaussian | reaction to *perturbation*, not *content* |
| `no_injection` | nothing | default answer; floor/ceiling problems |
| `yesbias_control` | the concept vector, unrelated no-answer question | injection making the model agreeable |

### Controls and the confound each targets

| Control | Failure mode |
|---|---|
| Norm-matched random vector | magnitude, not content |
| No-injection baseline | the model already answers one way regardless |
| Yes-bias question, paired within question type | general affirmativeness |
| Distractors from the same word bank | frequency / length / register artefacts |
| Baseline identification must be at chance | prompt leakage |
| Cosine-similarity check | "brain damage": model destroyed, not nudged |
| Logit lens ordering | vector reaching logits via the residual skip, not computation |
| Downstream ablation (double subtraction) | nothing actually *reads* the signal |
| Holm correction across the sweep | cherry-picking the best layer |
| Fixed seeds, per-item seeds, saved config | irreproducibility, silent config drift |

---

## 5. Architecture

```text
   "Tell me about ocean."          50 filler words
            │                            │
            ▼                            ▼
      forward pass                  forward pass
            │                            │
            ▼                            ▼
   resid_post[L][-1]            mean resid_post[L][-1]
            │                            │
            └──────────── − ─────────────┘
                          │
                          ▼
                 CONCEPT VECTOR  v
                          │
                          │  scaled to `strength` × mean residual norm
                          ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Introspection prompt  (never mentions "ocean")             │
  │     │                                                       │
  │     ▼                                                       │
  │  Tokenizer ──► injection window starts at the trial turn    │
  │     │                                                       │
  │     ▼                                                       │
  │  Embedding                                                  │
  │     │                                                       │
  │     ▼                                                       │
  │  Residual stream ═══════════════════════════════════╗       │
  │     │                                               ║       │
  │     ├── block 0   ── attn ──► + ── MLP ──► +        ║       │
  │     ├── ...                                         ║       │
  │     ├── block L   ── attn ──► + ── MLP ──► +        ║       │
  │     │                            ▲                  ║       │
  │     │                     ADD  s·v   ◄── INJECTION  ║       │
  │     ├── block L+1 ...  ◄── ablation sweep knocks     ║      │
  │     ├── ...               out attn_out / mlp_out     ║      │
  │     │                                               ║       │
  │     ▼                                               ║       │
  │  LayerNorm ──► Unembedding ──► logits               ║       │
  │                    │                                ║       │
  │                    ├── logit(Yes) − logit(No)       ║       │
  │                    └── forced choice over concepts  ║       │
  └─────────────────────────────────────────────────────╫──────┘
                                                        ║
                          cached resid_post[0..N] ◄═════╝
                                    │
                                    ├── logit lens by layer
                                    ├── cosine similarity by layer
                                    └── restore-direction patching
```

---

## 6. Project structure

```text
├── config.yaml         every scientifically relevant knob
├── run_all.py          full pipeline + verdict
├── data/introspection_prompts.jsonl
│
├── src/
│   ├── device.py       device selection, MPS self-test, env capture
│   ├── model.py        loading, tokenisation, offline stub
│   ├── prompts.py      word banks, prompt families, dataset builder
│   ├── hooks.py        read/write the residual stream, injection window
│   ├── behavioral.py   Yes/No readout, forced choice, generation
│   ├── introspection.py    trials, sweeps, logit lens
│   ├── interventions.py    downstream ablation and restore patching
│   ├── metrics.py      bootstrap, permutation tests, tables
│   ├── visualization.py    figures
│   ├── runner.py       config, persistence, pipeline assembly
│   └── transformer_walkthrough.py    educational forward-pass trace
│
├── experiments/        01 tensors · 02 baseline · 03 experiment
│                       04 interventions · 05 single-example walkthrough
├── notebooks/introspection_demo.ipynb
└── results/{raw,processed,figures}/ + REPORT.md, run_record.json
```

---

## 7. Installation (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No CUDA, no Docker, no API keys. The model (~1 GB) downloads from Hugging Face on
first run into `~/.cache/huggingface`.

**Apple Silicon.** `src/device.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` *before*
importing torch, so operations without an MPS kernel run on CPU instead of
raising. It then runs a numerical self-test — matmul, RMS normalisation, masked
softmax, attention-shaped reduction — on MPS against CPU, and falls back (loudly)
if they disagree beyond `device.mps_tolerance`. The experiment takes minutes on
CPU, so the fallback costs little and guards against silent MPS bugs.

Force CPU with `device.force_cpu: true`. `float32` is the default; if you switch
to `bfloat16`, re-run experiment 01 and check the residual-stream identity errors
are still ~1e-5 and not ~1e-1.

---

## 8. Running it

```bash
python run_all.py                     # everything + verdict, ~10-30 min
python run_all.py --quick             # small design, ~2-5 min
python run_all.py --offline --quick   # random-init model, no downloads, numbers meaningless

python experiments/01_transformer_basics.py
python experiments/02_behavioral_baseline.py
python experiments/03_introspection_experiment.py [--skip-sweeps]
python experiments/04_activation_intervention.py
python experiments/05_single_example_walkthrough.py --concept ocean

jupyter lab notebooks/introspection_demo.ipynb
```

---

## 9. One example, end to end

`experiments/05_single_example_walkthrough.py` prints all of this. For `ocean`:

1. **Prompt.** System message, briefing, then `"Trial: Do you detect an injected
   thought right now? Answer with a single word: Yes or No."` The word *ocean*
   appears nowhere.
2. **Tokenization.** ~115 tokens. The injection window starts at the first token
   of the final user turn — computed by comparing token *ids*, not by tokenising a
   prefix, since tokenizers are not prefix-consistent and an off-by-one here would
   be invisible.
3-4. **Clean pass and response** — the unperturbed model.
5. **Vector.** `act("Tell me about ocean.")[-1] − mean over 50 fillers` at
   `blocks.16.hook_resid_post`, scaled to 2× the mean residual norm there.
6. **Logit lens.** When does *ocean* become readable, and when does *Yes* win?
7. **Injected pass.** Same readout, hook active.
8. **Random control.** Gaussian vector of identical norm. The difference between 7
   and 8 — not 7 and 3 — is the quantity of interest.
9. **Ablation.** Each downstream `attn_out` / `mlp_out` zeroed in turn, with and
   without injection, reported as an interaction.
10. **Summary.**

---

## 10. Expected output

`results/REPORT.md`, plus a detection table:

| condition | N | mean | std | ci_low | ci_high | frac_yes | delta_vs_ref | dz | p_value |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| injected | 30 | … | … | … | … | … | … | … | … |
| random_control | 30 | … | … | … | … | … | 0 | 0 | — |
| no_injection | 30 | … | … | … | … | … | … | … | … |

…an identification table, a yes-bias control table, a Holm-corrected layer sweep,
an ablation table, and seven figures, each with an interpretation in
`results/figures/CAPTIONS.md`. Every raw file carries a `_meta.json` with the
config, versions, device, seed, and model summary that produced it.

---

## 11. Interpretation — the four outcomes

Specified before running, so the result classifies itself. `run_all.py` prints
which one the numbers match.

**A — clear introspection-like effect.** Injected above the random control,
identification above chance, yes-bias control flat, concept readable *before* the
decision forms, and some downstream ablation removes the effect. Striking at this
scale, so to be met with more scepticism, not less — re-run with a different seed,
different wording, and `injection.window: answer` before believing it.

**B — weak effect.** Detection responds to content but identification is at
chance: the injection perturbs the model in a content-dependent way without the
model having usable access to *what* the content is. Consistent with a small
model, a restricted readout, low power, or a prompt it cannot follow. Not evidence
of introspection.

**C — no effect.** **The expected outcome, and a useful one.** At 0.5B parameters,
with this prompt and readout, the model shows no sign of reading its own injected
state. A real measurement about a real model — and not evidence against a result
obtained three orders of magnitude larger.

**D — effect disappears under controls.** Injected beats random, but the same
injection shifts the unrelated no-answer question comparably: the measure is
tracking general affirmativeness. The most instructive outcome, being exactly the
failure a less careful version of this experiment would have published as a
success.

### Ways a positive result could still be wrong

| Alternative explanation | Guard |
|---|---|
| Perturbation magnitude, not content | norm-matched random control |
| General agreeableness | yes-bias question, paired within question type |
| Prompt wording cues | baseline identification at chance; concept never in the prompt |
| Token frequency / word length | same-bank distractors; length-normalised scoring |
| Output-format artefacts | pairwise Yes/No renormalisation; `argmax_is_yes_no` reported |
| Layer cherry-picking | one pre-registered layer; sweep Holm-corrected |
| Multiple comparisons | single confirmatory test |
| Model damage rather than steering | cosine-similarity check; strength sweep for the inverted-U |
| Vector reaching logits via the residual skip | logit-lens ordering; downstream ablation |
| Reacting to a corrupted *question* | `injection.window: answer` variant |
| Probe leakage | no probe is trained; nothing is fit to the data |

---

## 12. Deviations from the original

1. **Model.** Qwen2.5-0.5B-Instruct, not Claude Opus 4.1. ~1000× smaller.
2. **Grading.** Restricted logit readout and forced choice, not an LLM judge.
3. **Strength units.** Multiples of the mean residual norm *at the injection
   layer*, not raw multiples of the vector. The residual norm grows by roughly an
   order of magnitude across depth, so a fixed raw scalar would mean something
   entirely different at layer 4 than at layer 20 — and this experiment sweeps
   layers.
4. **Trial structure.** One trial per conversation, not a sequence.
5. **Scope.** Only the first of the original's four experiments.

---

## Limitations

**Absence of the effect in a small model does not disprove results found in
substantially larger models.** The original reports it at ~20% frequency, so it is
unreliable even where it exists; detecting a 20%-frequency capability requires the
capability to be present, and there is no statistical remedy for its absence. Read
Outcome C as a measurement of *this* model, nothing more.

* **Post-training matters.** Introspective self-report is plausibly
  post-training-dependent, so this is not only a scale comparison.
* **N = 30 concepts.** Enough for a medium effect (dz ≈ 0.5 at ~80% power), not a
  small one. The layer sweep is underpowered per layer.
* **One prompt.** Results may be sensitive to the briefing; `src/prompts.py` is
  where to test that. Treat variation across prompts as a finding.
* **Forced choice is not free report** — an easier task, and a different one.
* **Single seed.** Repeat across `config.seed` values before trusting any effect
  near the significance boundary.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: Can't load the configuration of 'Qwen/…'` | No network to `huggingface.co`, or a proxy blocking it. Test `curl -I https://huggingface.co`. To work offline, use `--offline`. |
| MPS produces NaNs or nonsense | `device.force_cpu: true`. CPU is a few times slower, not orders of magnitude. `python -m src.device` shows the self-test result. |
| `RuntimeError: MPS backend out of memory` | Close other apps, or `model.dtype: bfloat16`, or force CPU. fp32 needs ~2 GB plus activations. |
| TransformerLens `TransformerBridge` deprecation warning | Expected on 3.x. `HookedTransformer` still works; pin `transformer-lens<4` if a future release removes it. |
| `Yes/No are not single tokens for this model` | Printed at startup if you change models. The readout falls back to first tokens: valid, coarser. |
| Baseline identification above chance | **Stop.** The prompt or candidate set is leaking. Nothing under injection means anything until this is at chance. |
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root, or use the `experiments/` scripts, which fix `sys.path` themselves. |

---

## Extensions

1. **Scale.** Run at 0.5B, 1.5B and 3B and plot the effect against parameter
   count. The single most informative follow-up.
2. **Grading.** Swap the restricted readout for an LLM judge on free text. If the
   conclusion changes, the readout was the limiting factor, not the model.
3. **Better vectors.** Extract from contrastive scenario *pairs*, as the original
   also does, and compare vector quality.
4. **Paper experiment 2.** Inject while the model transcribes unrelated text; test
   whether it keeps "thought" and "text" separate.
5. **Paper experiment 3.** Prefill an out-of-character output, retroactively inject
   the matching concept beforehand, and test whether it stops disavowing it.

---

## Reference

Jack Lindsey. *Emergent Introspective Awareness in Language Models.* Transformer
Circuits, 2025. <https://transformer-circuits.pub/2025/introspection/index.html>

Neel Nanda. *Clean Transformer Demo* (Easy-Transformer). `src/transformer_walkthrough.py`
follows its pedagogy but uses TransformerLens, since Easy-Transformer targets a
2022 PyTorch and a learned-absolute-position architecture Qwen does not use.
