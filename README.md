# `jax-raft`
A JAX/Flax implementation of the RAFT optical flow estimator [(https://arxiv.org/abs/2003.12039)](https://arxiv.org/abs/2003.12039), ported from PyTorch [(https://docs.pytorch.org/vision/main/models/raft.html)](https://docs.pytorch.org/vision/main/models/raft.html). Checkpoints have been ported, too. The implementation has been tested to reproduce the original results. 

## Reproducibility
With pre-trained checkpoints, `jax-raft` achieves the following metrics on Sintel (train), compared to the original PyTorch implementation. This comparison uses the `raft_large_C_T_SKHT_V2` and `raft_small_C_T_V2` checkpoints, respectively. FPS have been computed on a single RTX 3090 Ti.

| Model                   | EPE (clean) | EPE (final) | FPS  |
|-------------------------|-------------|-------------|------|
| raft_large (`jax-raft`) | 0.650       | 1.019       | 11.9 |
| raft_large (PyTorch)    | 0.649       | 1.020       | 8.1  |
| raft_small (`jax-raft`) | 1.993       | 3.268       | 36.9 |
| raft_small (PyTorch)    | 1.998       | 3.279       | 15.0 |

## Usage
As easy as that:
```python
from jax_raft import raft_large  # or raft_small
model, variables = raft_large(pretrained=True)
model.apply(variables, image1, image2, train=False)
```
`jax-raft` is fully compatible with `jax.jit`; RAFT's recurrent refinement process has been implemented using `jax.lax.scan`. 

## Installation
```python
pip install git+https://github.com/alebeck/jax-raft
```

## Additional resources
In the `scripts` directory, we provide scripts for converting official PyTorch RAFT checkpoints to Flax; and for validation on Sintel. The `examples` directory contains example usage scripts.