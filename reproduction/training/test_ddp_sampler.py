"""Small torchrun smoke test for the weighted DDP sampler."""

import torch
import torch.distributed as dist

from esm.esmdynamic.training.train import DistributedWeightedSampler


dist.init_process_group("gloo")
sampler = DistributedWeightedSampler(torch.arange(1, 11, dtype=torch.double), 20, 123)
sampler.set_epoch(4)
indices = list(sampler)
gathered = [None] * dist.get_world_size()
dist.all_gather_object(gathered, indices)
assert len(indices) == 10
assert gathered[0] != gathered[1]
print(f"rank={dist.get_rank()} count={len(indices)} first={indices[:3]}")
dist.destroy_process_group()
