"""Small-scale reproduction of the Transformer-Circuits introspection experiment.

Modules
-------
device            device selection + MPS sanity checking
model             model loading (TransformerLens) and an offline stub model
prompts           stimulus construction and the prompt dataset
hooks             activation caching and residual-stream injection
behavioral        externally observable measurements (logits, generations)
introspection     the introspection experiment proper
interventions     causal ablations downstream of the injection site
metrics           bootstrap CIs, permutation tests, results tables
visualization     figures
transformer_walkthrough  educational forward-pass trace
"""

__version__ = "0.1.0"
