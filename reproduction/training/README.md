# ESMDynamic two-stage training

The training entry point supports the paper's two-stage procedure:

1. `rcsb_pretrain.txt` trains a one-condition dynamic-contact head on RCSB
   structural-cluster labels. RCSB strict-upper-triangle CSV/PT labels are
   restored to symmetric matrices in memory.
2. `mdcath_finetune.txt` creates the five-condition mdCATH heads. Compatible
   dynamic-head weights are loaded from stage 1, and its one-condition output
   layers are replicated to initialize all five conditions.

Run on physical GPU 1 from the repository root:

```bash
conda activate esmdynamic
cd /data/user/sunxc/Projects/esmdynamic
CUDA_VISIBLE_DEVICES=1 python esm/esmdynamic/training/train.py \
  @reproduction/training/rcsb_pretrain.txt

CUDA_VISIBLE_DEVICES=1 python esm/esmdynamic/training/train.py \
  @reproduction/training/mdcath_finetune.txt
```

Resume RCSB training on physical GPUs 0 and 1 with DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  esm/esmdynamic/training/train.py \
  @reproduction/training/rcsb_pretrain.txt \
  --resume /bigdat2/user/sunxc/esmdynamic_training/01_rcsb_dynamic/checkpoint_last.pt
```

`--batch-size=4 --batch-accum=16` continues to describe the desired global
effective batch of 64. In a two-rank run the script automatically uses eight
accumulation steps per rank: `4 x 8 x 2 = 64`. The configured 10,000 training
and 1,000 validation draws are global counts and are partitioned between ranks.
Only rank 0 writes history, TensorBoard data, and atomic checkpoints.

Inside the process, the selected physical GPU is exposed as logical `cuda:0`;
therefore the configuration correctly uses `--device=cuda`.

Each output directory contains `checkpoint_best.pt`, `checkpoint_last.pt`,
standalone head weights, `history.csv`, `run_metadata.json`,
`timing_summary.json`, and TensorBoard logs. Resume an interrupted run with
`--resume /path/to/checkpoint_last.pt`. Validation reports per-protein mean,
standard deviation, SEM, and sample count. Dynamic classification metrics and
frequency RMSE use the strict upper triangle without the diagonal; condition 0
corresponds to 320 K for mdCATH.

The supplied configurations follow the Methods settings: effective batch size
64 (batch 4 x accumulation 16), 10,000/1,000 sampled proteins per RCSB epoch,
1,000/100 per mdCATH epoch, Adam at 1e-4, and early stopping after 10 epochs
without validation improvement. Auxiliary confidence/residual losses begin at
epoch 11 during fine-tuning. Before a full run, copy a configuration and set
small values such as
`--train-samples-per-epoch=8`, `--val-samples-per-epoch=4`, and `--epochs=1` for
a smoke test.

The released RCSB splits contain 164 entries whose label count does not match
the FASTA length (151 train, 5 validation, 8 test). The integrity report is in
`reproduction/results/rcsb_integrity_20260919/`. `rcsb_pretrain.txt` uses the
auditable filtered splits in `rcsb_filtered/`; original release files remain
unchanged. Recreate them with:

```bash
python reproduction/training/filter_rcsb_splits.py \
  --source-dir /bigdat2/user/sunxc/rcsb_release/rcsb \
  --issues-csv reproduction/results/rcsb_integrity_20260919/issues.csv \
  --output-dir reproduction/training/rcsb_filtered
```
