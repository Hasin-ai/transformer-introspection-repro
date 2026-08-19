You are an expert in **mechanistic interpretability, transformer internals, PyTorch, TransformerLens/Easy-Transformer, and reproducible ML experimentation**.

I want you to study an introspection experiment from the Transformer Circuits work and design a **small, understandable, locally reproducible version** that I can run on a MacBook.

## Primary references

First, carefully read and analyze:

1. Transformer Circuits introspection article:
   https://transformer-circuits.pub/2025/introspection/index.html

2. Easy Transformer / Clean Transformer demo notebook:
   https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo_Template.ipynb

Do not merely summarize these pages. Understand the experimental idea, what "introspection" means in the paper, what behavior is being measured, what model internals are involved, and which parts can realistically be reproduced at small scale.

---

# Objective

Design and implement the **simplest scientifically meaningful experiment** that demonstrates the central introspection phenomenon from the Transformer Circuits article while being practical to run locally on a modern MacBook.

The experiment should prioritize:

- conceptual clarity;
- reproducibility;
- minimal compute requirements;
- transparent transformer internals;
- easy modification;
- interpretable outputs;
- no dependence on expensive GPUs or distributed infrastructure.

It does **not** need to reproduce the paper's full-scale results.

Instead, create a small proof-of-concept that captures the most important idea behind the original experiment.

---

# Hardware constraints

Assume the experiment will run on a MacBook with approximately:

- Apple Silicon: M1/M2/M3/M4
- 16–32 GB unified memory
- Python 3.10+
- PyTorch
- Apple MPS acceleration if available
- CPU fallback

The entire core experiment should ideally work in a few minutes to tens of minutes rather than hours.

Avoid models requiring large amounts of RAM/VRAM.

Prefer models roughly in the range of:

- tiny custom transformer;
- GPT-2 small;
- Pythia-70M / Pythia-160M;
- similarly small open models;

depending on which is scientifically appropriate.

Explain your choice.

---

# Important: distinguish reproduction from adaptation

Before writing code, explain:

1. What the original Transformer Circuits experiment actually tests.
2. What the key introspection hypothesis is.
3. What measurements constitute evidence for introspection.
4. Which parts of the original experiment require large models or infrastructure.
5. Which parts can be meaningfully reproduced on a MacBook.
6. What your simplified experiment changes.
7. What conclusions the simplified experiment can and **cannot** support.

Do not claim that the small experiment is an exact replication if it is actually an adaptation.

Label it clearly as one of:

- exact replication;
- partial replication;
- conceptual replication;
- toy demonstration.

---

# Step 1 — Analyze the original experiment

Give a concise but technically serious explanation of the experiment.

Cover, where relevant:

- prompt construction;
- task definition;
- model behavior;
- introspective reports;
- latent/internal representations;
- activations;
- residual stream;
- attention;
- MLPs;
- logits;
- logit lens;
- activation patching;
- interventions;
- controls;
- baselines;
- statistical comparisons;
- evaluation metrics.

For every major component, distinguish between:

**Essential to the phenomenon**

and

**Useful but optional for a small reproduction.**

---

# Step 2 — Select the smallest viable experiment

Propose **3 possible reproduction strategies**, ordered from easiest to most faithful.

For example:

### Option A — Tiny conceptual demonstration
Very small pretrained transformer or custom transformer.

### Option B — Small pretrained language model
For example GPT-2 Small or Pythia-70M/160M.

### Option C — More faithful but somewhat heavier reproduction
Still feasible on a high-memory MacBook.

For each option provide:

- model;
- approximate parameter count;
- expected RAM usage;
- expected runtime;
- difficulty;
- faithfulness to the original experiment;
- advantages;
- limitations.

Then choose **one recommended implementation** and explain why.

---

# Step 3 — Use the Clean Transformer notebook pedagogically

Study the Easy-Transformer Clean Transformer Demo and use its philosophy to make the implementation understandable.

I want to see how a transformer works internally rather than simply calling a black-box generation API.

Show clearly how data flows through:

`tokens`
→ `token embeddings`
→ `positional embeddings`
→ `residual stream`
→ `attention`
→ `MLP`
→ `residual stream`
→ `LayerNorm`
→ `unembedding`
→ `logits`

Where practical, expose/cache:

- residual-stream activations;
- attention patterns;
- attention outputs;
- MLP outputs;
- layer-normalized residuals;
- logits.

If Easy-Transformer is outdated or incompatible with modern PyTorch/macOS, explain this and use **TransformerLens** or an equivalent modern implementation while preserving the educational structure of the original notebook.

Do not unnecessarily depend on obsolete packages just to imitate the old notebook.

---

# Step 4 — Build the experiment

Create a small introspection experiment with four components.

## A. Stimulus / prompt generation

Create a controlled dataset of prompts capable of testing the intended introspection phenomenon.

The dataset should contain:

- experimental examples;
- negative controls;
- matched controls where possible;
- randomized seeds;
- enough examples for a small statistical comparison.

Save it in a human-readable format.

Example:

`data/introspection_prompts.jsonl`

Each line might contain fields such as:

```json
{
  "id": 1,
  "condition": "experimental",
  "prompt": "...",
  "target": "...",
  "metadata": {}
}
```

Use the schema actually appropriate to the experiment.

---

## B. Behavioral experiment

Measure the model's externally observable response.

Save:

- prompts;
- generated responses;
- token probabilities;
- target-token probabilities where applicable;
- predictions;
- experimental condition;
- evaluation scores.

Do not rely only on qualitative examples.

---

## C. Internal transformer analysis

Instrument the transformer.

At minimum, investigate the most relevant internal quantities for this experiment.

Potential analyses include:

- residual-stream activation;
- attention patterns;
- layer-by-layer logits;
- logit-lens predictions;
- MLP contributions;
- cosine similarity between representations;
- activation differences between conditions.

Only include analyses that genuinely help answer the introspection question.

Do not add interpretability techniques merely to make the project look sophisticated.

---

## D. Intervention

If technically reasonable, implement at least one causal intervention such as:

- activation patching;
- zero ablation;
- mean ablation;
- residual-stream replacement;
- attention-head ablation;
- MLP-output ablation.

Use the intervention to test whether an internal state associated with the introspection phenomenon actually affects the model's response.

Explain the causal logic.

---

# Step 5 — Add controls

The simplified experiment must include controls.

Possible controls could include:

- random prompts;
- semantically matched prompts;
- shuffled labels;
- random activation vectors;
- mismatched activation patches;
- unrelated layers;
- unrelated token positions;
- no-intervention baseline.

Explain what failure mode each control is intended to detect.

---

# Step 6 — Quantitative evaluation

Define clear metrics before examining the results.

Depending on the experiment, consider:

- accuracy;
- probability assigned to target answer;
- logit difference;
- KL divergence;
- cosine similarity;
- intervention effect size;
- introspection-vs-control performance;
- confidence intervals;
- permutation tests;
- simple bootstrap estimates.

Prefer a small number of interpretable metrics.

Produce a results table similar to:

| Condition | N | Mean score | Std | Effect vs control |
|---|---:|---:|---:|---:|
| Introspection | ... | ... | ... | ... |
| Matched control | ... | ... | ... | ... |
| Random control | ... | ... | ... | ... |

Explain what pattern would count as evidence and what pattern would falsify the simplified hypothesis.

---

# Step 7 — Create visualizations

Create simple plots such as:

1. introspection performance vs control;
2. target-token probability by condition;
3. layer-by-layer logit difference;
4. intervention effect by layer;
5. activation similarity by layer.

Avoid unnecessary visualization complexity.

Every graph must have:

- title;
- axis labels;
- legend where appropriate;
- explanation of its interpretation.

Save figures under:

```text
results/figures/
```

---

# Step 8 — Produce a complete project

Do not give me isolated code snippets.

Generate the experiment as a **complete small repository**.

Use approximately this structure:

```text
introspection_reproduction/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   └── introspection_prompts.jsonl
│
├── src/
│   ├── __init__.py
│   ├── device.py
│   ├── model.py
│   ├── transformer_walkthrough.py
│   ├── prompts.py
│   ├── hooks.py
│   ├── behavioral.py
│   ├── introspection.py
│   ├── interventions.py
│   ├── metrics.py
│   └── visualization.py
│
├── experiments/
│   ├── 01_transformer_basics.py
│   ├── 02_behavioral_baseline.py
│   ├── 03_introspection_experiment.py
│   └── 04_activation_intervention.py
│
├── notebooks/
│   └── introspection_demo.ipynb
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
└── run_all.py
```

You may modify the structure if another organization is clearly better.

---

# Step 9 — Write COMPLETE code

For every file, give the **complete contents**.

Do not write:

```python
# add implementation here
```

or:

```python
# same as above
```

or pseudocode where working Python is expected.

Every Python file should be runnable.

Use clear functions, type hints where useful, comments, and docstrings.

Prefer straightforward research code over elaborate software engineering.

---

# Step 10 — MacBook device handling

Create robust device selection.

For example:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

But verify whether individual operations required by TransformerLens/model code work properly on MPS.

If some operation is unreliable on MPS, provide a safe CPU fallback.

Avoid hard-coding CUDA.

Explain any environment variables needed for MPS fallback.

---

# Step 11 — Installation instructions

Provide exact commands starting from a clean environment.

Prefer something like:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Specify versions when version compatibility matters.

Then show:

```bash
python run_all.py
```

and commands for running experiments separately.

---

# Step 12 — Reproducibility

Set random seeds for:

- Python;
- NumPy;
- PyTorch.

Record:

- Python version;
- PyTorch version;
- model name;
- model revision if relevant;
- TransformerLens version;
- device;
- random seed;
- configuration.

Save the experiment configuration alongside results.

---

# Step 13 — Make the transformer mechanics understandable

Create:

```text
src/transformer_walkthrough.py
```

This should be an educational implementation inspired by the Clean Transformer Demo.

Walk through one forward pass and print or return tensor shapes such as:

```text
tokens            [batch, position]
embedding         [batch, position, d_model]
residual_pre      [batch, position, d_model]
q                 [batch, position, head, d_head]
k                 [batch, position, head, d_head]
v                 [batch, position, head, d_head]
attention_scores  [...]
attention_pattern [...]
attention_output  [...]
mlp_output        [...]
logits            [batch, position, vocab]
```

Explain what each tensor means.

Connect these quantities to the introspection experiment.

---

# Step 14 — Add an introspection walkthrough

After the general transformer walkthrough, provide a single-example walkthrough showing:

1. the input prompt;
2. tokenization;
3. normal forward pass;
4. model response;
5. relevant activation extraction;
6. layer-by-layer analysis;
7. introspective prediction/report;
8. control comparison;
9. intervention;
10. changed output after intervention.

Print enough intermediate information that someone learning mechanistic interpretability can follow what happened.

---

# Step 15 — Notebook

Create a clean:

```text
notebooks/introspection_demo.ipynb
```

It should act as the tutorial version of the project.

Organize it approximately as:

1. Introduction
2. What we are testing
3. Original experiment vs this reproduction
4. Setup
5. Load model
6. Transformer forward-pass walkthrough
7. Generate experimental prompts
8. Behavioral baseline
9. Cache activations
10. Inspect internal state
11. Introspection test
12. Controls
13. Intervention
14. Quantitative results
15. Visualization
16. Interpretation
17. Limitations
18. Further experiments

The notebook should call reusable functions from `src/` rather than duplicating the entire codebase.

---

# Step 16 — README

Write an excellent `README.md`.

Include:

- project motivation;
- original paper/article;
- what phenomenon is being reproduced;
- simplified hypothesis;
- experiment diagram;
- project structure;
- installation;
- MacBook instructions;
- how to run;
- expected output;
- how to interpret results;
- limitations;
- troubleshooting;
- ideas for extensions.

Clearly state that absence of the effect in a small model does not automatically disprove results found in substantially larger models.

---

# Step 17 — Explain expected outcomes

Before claiming success, specify possible outcomes.

### Outcome A — Clear introspection-like effect

Explain what result would support this interpretation.

### Outcome B — Weak effect

Explain why model size, prompt design, statistical power, or task construction could matter.

### Outcome C — No effect

Explain why that would still be a useful result for this reproduction.

### Outcome D — Apparent effect disappears under controls

Explain why this suggests a confound rather than genuine introspection.

Be scientifically conservative.

---

# Step 18 — Validate the implementation

Before presenting the final project, mentally/code-review the implementation for common errors, including:

- incorrect tensor dimensions;
- wrong hook names;
- wrong token positions;
- incorrect MPS device handling;
- accidentally comparing different prompts;
- data leakage;
- using generated text instead of logits incorrectly;
- incorrect activation patch direction;
- incorrect normalization;
- accidentally measuring prompt semantics rather than introspection;
- random seeds not being applied;
- outputs being overwritten.

Make fixes before providing the final code.

---

# Output format

Structure your final answer exactly in this order:

## 1. Original Experiment Explained

Explain the Transformer Circuits experiment.

## 2. What We Can Reproduce on a MacBook

Separate exact replication from conceptual replication.

## 3. Three Candidate Experiments

Compare three implementations.

## 4. Recommended Experiment

Give the exact hypothesis:

**H0:** ...

**H1:** ...

Then explain the experimental design.

## 5. Architecture

Show the transformer and experimental data flow using an ASCII diagram.

Example:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Embedding
  │
  ▼
Residual Stream ──────────────────────────┐
  │                                      │
  ├── Attention                          │
  │       │                              │
  │       ▼                              │
  ├── MLP                                │
  │                                      │
  ▼                                      │
Final Residual                           │
  │                                      │
  ▼                                      │
Unembedding                              │
  │                                      │
  ▼                                      │
Logits                                   │
                                         │
Cached activations ◄─────────────────────┘
      │
      ├── introspection analysis
      ├── controls
      └── causal interventions
```

## 6. Project Directory

Show the complete file tree.

## 7. Installation

Give exact macOS commands.

## 8. Complete Source Code

For each file use:

```text
FILE: src/model.py
```

followed by the complete code.

Do this for **every file**.

## 9. Experiment Walkthrough

Trace one example end-to-end.

## 10. Running the Experiment

Give exact commands.

## 11. Expected Results

Explain the tables/plots that will appear.

## 12. Interpretation

Explain what would and would not constitute evidence of introspection.

## 13. Limitations

Discuss limitations compared with the 2025 Transformer Circuits experiment.

## 14. Next Experiments

Suggest 5 increasingly ambitious extensions.

---

# Coding preferences

Use:

- Python;
- PyTorch;
- TransformerLens where useful;
- Hugging Face only where useful;
- NumPy;
- pandas;
- matplotlib;
- scipy if required;
- tqdm;
- PyYAML.

Avoid unnecessary frameworks.

Do not require:

- CUDA;
- Docker;
- cloud GPUs;
- paid APIs;
- proprietary models.

If the phenomenon fundamentally requires a model unavailable locally, say so explicitly and build the closest scientifically meaningful local approximation instead of pretending it is a faithful reproduction.

---

# Scientific standard

Be skeptical of your own experiment.

Specifically check whether a positive result could instead arise from:

- prompt wording;
- ordinary semantic inference;
- memorized associations;
- token-frequency effects;
- output-format cues;
- activation magnitude;
- layer-selection cherry-picking;
- multiple comparisons;
- classifier/probe leakage.

Where possible, design controls against these alternatives.

The goal is not to "prove that transformers are introspective."

The goal is to create the **smallest clean experiment that lets us investigate whether a transformer has access to information about its own internal state in a way analogous to the phenomenon studied in the Transformer Circuits article.**

---

# Final requirement

I should be able to:

```bash
git clone <project>
cd introspection_reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

on a MacBook and get:

1. a behavioral baseline;
2. an introspection-vs-control comparison;
3. cached transformer activations;
4. at least one layer-wise internal analysis;
5. at least one causal intervention;
6. quantitative metrics;
7. plots;
8. saved raw results;
9. a clear explanation of whether the simplified experiment showed evidence consistent with introspection.

Above all, favor **working, minimal, readable, reproducible code** over an overcomplicated approximation of the original experiment.