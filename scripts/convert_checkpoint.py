"""
Helper script for converting official PyTorch RAFT checkpoints to JAX/Flax.
"""

import sys
import numpy as np
import torch
from flax.serialization import to_bytes


def _convert(in_dict):
    top_keys = set([k.split('.')[0] for k in in_dict.keys()])
    leaves = set([k for k in in_dict.keys() if '.' not in k])

    # convert leaves
    out_dict = {}
    for l in leaves:
        if l == 'weight' and in_dict[l].ndim == 4:
            # conv kernels
            out_dict['kernel'] = np.asarray(in_dict[l]).transpose((2, 3, 1, 0))
        elif l == 'weight' and in_dict[l].ndim == 1:
            # normalization scales
            out_dict['scale'] = np.asarray(in_dict[l])
        else:
            out_dict[l] = np.asarray(in_dict[l])

    for top_key in top_keys.difference(leaves):
        new_top_key = 'layers_' + top_key if top_key.isdigit() else top_key
        out_dict[new_top_key] = _convert(
            {k[len(top_key) + 1:]: v for k, v in in_dict.items() if k.startswith(top_key + '.')})

    return out_dict


def convert_checkpoint(torch_checkpoint, output_file):
    state_dict = torch.load(torch_checkpoint, map_location='cpu')

    # Move batch norm stats to a separate dict
    params, batch_stats = {}, {}
    for k, v in state_dict.items():
        if k.endswith('.running_mean'):
            batch_stats[k.replace('.running_mean', '.mean')] = v
        elif k.endswith('.running_var'):
            batch_stats[k.replace('.running_var', '.var')] = v
        elif k.endswith('.num_batches_tracked'):
            pass
        else:
            params[k] = v

    # Recursively convert params and batch stats to Flax format
    params_flax = _convert(params)
    batch_stats_flax = _convert(batch_stats)
    state_flax = {'params': params_flax, 'batch_stats': batch_stats_flax}

    with open(output_file, 'wb') as f:
        f.write(to_bytes(state_flax))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    assert output_file.endswith('.msgpack')

    convert_checkpoint(input_file, output_file)
