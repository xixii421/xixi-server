"""CPU/Gloo teaching implementation of the disaggregated serving path.

Model stages are simulated. Tensor transport is real inter-process Gloo traffic.
Importing this package does not load a model, CUDA, or the production backend.
"""
