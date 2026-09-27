# Local reproduction artifacts

This directory contains the retained artifacts from the two-stage training
reproduction and the fixed 270-protein mdCATH test evaluation.

## Contents

- `01_rcsb_dynamic_batch8/`: stage-1 RCSB pretraining. The best checkpoint is
  epoch 36 (`val_loss=0.00028673794255906066`).
- `02_mdcath_finetune_from_rcsb_best/`: stage-2 mdCATH fine-tuning initialized
  from the stage-1 best checkpoint. The best checkpoint is epoch 97
  (`val_loss=0.1919394102692604`).
- `eval_gpu3_table1_v97/`: evaluation of the epoch-97 checkpoint on the fixed
  270-protein mdCATH test split at 320 K.

The repository's existing `*.pt` ignore rule keeps model checkpoints out of
regular Git because every retained checkpoint exceeds GitHub's 100 MB file
limit. The checkpoints remain available in this local working tree. Their
SHA-256 checksums are listed below for provenance.

```text
d61733959759f9c2c51c90c3adf45c0bde069d12fc1a9b4f10ec0a427e167fc4  01_rcsb_dynamic_batch8/checkpoint_best.pt
d7f22a02b603cf94dafae80eebc19ff09593e79b4e6f08058fb8dc9510fecad7  01_rcsb_dynamic_batch8/checkpoint_last.pt
473055f27b03a2c89bfc7704d2fb09b68d17887d30b0f6b05168a1f6ff020ab8  01_rcsb_dynamic_batch8/heads_best.pt
081accbc0bbab0d1b016213018db599f959c8b3af6b1eae6d13c30001975e2c2  01_rcsb_dynamic_batch8/heads_last.pt
9533d9d9383318d07e1bc925134eb245f8562701a2b11425b92a84beb7537d39  02_mdcath_finetune_from_rcsb_best/checkpoint_best.pt
56be33021ffa40dd573b6ccb2c1d848e2774e135fe7864f1c863a29a7e716675  02_mdcath_finetune_from_rcsb_best/checkpoint_last.pt
a333da10d0e62f9ec4a4ffcc1ee47696a4bfb5b4f31df290c3f7b96a5667a29f  02_mdcath_finetune_from_rcsb_best/heads_best.pt
74d8d71bb654b1ac733f12911ac6cb2539073e18c3159b39b35d6d217d1a8d27  02_mdcath_finetune_from_rcsb_best/heads_last.pt
```
