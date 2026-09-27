# ESMDynamic mdCATH Table 1 Reproduction

## Experimental setup

- Dataset: mdCATH test set, 270 proteins at 320 K
- Model weights: `/bigdat2/user/sunxc/ESMDynamic/esmdynamic_model_weights_V2.pt`
- GPU: physical GPU 1, NVIDIA GeForce RTX 5090 (32,607 MiB)
- Python: 3.11.13
- PyTorch: 2.8.0+cu129
- CUDA: 12.9
- Batch size: 1
- Chunk size: 128
- Number of recycles: 3
- Classification threshold: 0.5
- Evaluation scope: upper triangle without the diagonal
- Run interval: 2026-09-18 09:41:08–09:50:32 UTC

## Table 1 metrics

Values following `±` are standard errors over 270 proteins.

| Metric | Reproduced | Paper | Mean difference |
|---|---:|---:|---:|
| Balanced Accuracy | 0.8048 ± 0.0062 | 0.796 ± 0.007 | +0.0088 |
| Precision | 0.5098 ± 0.0118 | 0.511 ± 0.012 | −0.0012 |
| Recall | 0.7713 ± 0.0095 | 0.767 ± 0.010 | +0.0043 |
| F1 | 0.5709 ± 0.0083 | 0.569 ± 0.008 | +0.0019 |
| AUROC | 0.8992 ± 0.0057 | 0.889 ± 0.006 | +0.0102 |
| RMSE | 0.0742 ± 0.0023 | 0.076 ± 0.002 | −0.0018 |

The reproduced values are close to the published Table 1 values. Hardware differs from the paper's RTX 4090, so runtime is not a direct hardware-matched comparison.

## Runtime and memory

| Measurement | Seconds | Human-readable |
|---|---:|---:|
| Model loading | 74.44 | 1 min 14.44 s |
| Sum of per-protein inference times | 399.59 | 6 min 39.59 s |
| Mean inference time per protein | 1.48 | 1.48 s |
| Complete inference-loop wall time | 489.05 | 8 min 9.05 s |
| Complete process wall time | 564.74 | 9 min 24.74 s |

- Model-load peak GPU memory: 8,300.23 MiB
- Overall peak GPU memory: 10,825.58 MiB

Per-protein inference time is synchronized with CUDA before and after each prediction. The complete inference-loop time additionally includes sample loading, metric calculation, and CSV writing. Complete process wall time also includes model construction and weight loading.

## Source result files

- `summary.json`: full-precision aggregate metrics
- `run_metadata.json`: environment, timing, and memory metadata
- `per_protein_metrics.csv`: per-protein metrics, inference time, length, and peak memory
- `table1_metrics_comparison.csv`: compact paper-versus-reproduction comparison
- `timing_summary.csv`: compact timing summary
