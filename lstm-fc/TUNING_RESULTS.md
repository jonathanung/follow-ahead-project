# LSTM-FC Hypertuning Results

## Summary

Systematic hyperparameter tuning of the LSTM-FC model for human action prediction
(left/straight/right) on walking trajectory data across three rounds of search.

## Best Configuration

| Parameter | Original Default | Round 1-2 Best | Round 3 Best |
|-----------|-----------------|----------------|--------------|
| hidden_size | 64 | 96 | **192** |
| num_layers | 1 | 1 | **2** |
| dropout | 0.0 | 0.1 | 0.1 |
| batch_size | 64 | 512 | 512 |
| learning_rate | 0.01 | 0.006 | 0.006 |
| optimizer | Adam | Adam | **AdamW** |
| scheduler | StepLR | OneCycleLR | OneCycleLR |
| training_steps | 100,000 | 50,000-100,000 | **200,000** |

## Results Comparison

| Model | MSE | Accuracy | Parameters | Checkpoint |
|-------|-----|----------|------------|------------|
| Original (500 steps) | 0.0441 | 69.5% | 17,603 | — |
| Tuned h=96 (50k steps) | 0.0140 | 87.8% | 38,691 | `outputs/final_best/` |
| Tuned h=128 (100k steps) | 0.0134 | 88.0% | 67,971 | `outputs/final_best_128h/` |
| **Tuned h=192 L=2 (200k steps)** | **0.0041** | **90.1%** | **447,555** | **`outputs/final_v3/`** |

**Round 3 improvement: 3.2x lower MSE, +2.1% accuracy over previous best.**

## Per-Class Performance

### Best Model (h=192, L=2, 200k steps)

| Class | Accuracy | Precision | Recall | F1 | Count |
|-------|----------|-----------|--------|-----|-------|
| Left | 75.8% | 89.4% | 75.8% | 82.0% | 24,021 |
| Straight | 96.1% | 90.4% | 96.1% | 93.2% | 109,580 |
| Right | 77.4% | 89.6% | 77.4% | 83.0% | 24,349 |

### Previous Best (h=128, L=1, 100k steps) for comparison

| Class | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Left | 70.9% | 87.0% | 70.9% | 78.1% |
| Straight | 95.3% | 88.4% | 95.3% | 91.7% |
| Right | 72.0% | 86.6% | 72.0% | 78.7% |

**Left/right accuracy improved by +5-6 percentage points.**

## Key Findings

### Round 1-2 (unchanged)
1. **Dropout (0.1)** provides significant regularization benefit
2. **Larger batch sizes (512)** dramatically improve convergence stability
3. **OneCycleLR scheduler** outperforms StepLR and CosineAnnealing
4. **Learning rate 0.005-0.006** is optimal
5. **MSE loss** outperformed KL-Divergence and Cross-Entropy with soft labels

### Round 3 (new findings)
6. **2-layer LSTM outperforms 1-layer** — this was missed in rounds 1-2 because
   2-layer configs were only tested with weak schedulers (StepLR, CosineAnnealing).
   OneCycleLR is required to train 2-layer LSTM effectively.
7. **Larger hidden sizes (192-384)** consistently beat smaller ones
8. **Gradient clipping** provided no benefit
9. **AdamW weight decay** provided no benefit (dropout is sufficient)
10. **Warmup ratio (pct_start)** insensitive — 0.1 is fine
11. **200k steps with OneCycleLR** dramatically improves over 100k — the model
    keeps improving as LR decays toward zero in the final phase

## Tuning Process

- **Round 1**: 50 trials across 4 architectures (base, dropout, bidir, deep),
  hidden sizes [32-256], LRs [0.0005-0.02], batch sizes [32-256],
  loss functions [MSE, KL, CE], schedulers, dropout rates, weight decay
- **Round 2**: 62+ trials with fine grid around best config from round 1,
  hidden sizes [96-256], LRs [0.003-0.008], batch sizes [128-512],
  schedulers [step, cosine, onecycle], 10k steps per trial
- **Round 3**: 59 trials exploring 2-layer LSTM with OneCycleLR,
  hidden sizes [64-384], gradient clipping [0.5-5.0], AdamW weight decay,
  warmup ratios [0.05-0.3], batch sizes [256-2048], 20k steps per trial.
  Top 7 results were all 2-layer LSTM configs.
- **Final**: 200k-step long training with best config (h=192, L=2, drop=0.1,
  lr=0.006, bs=512, OneCycleLR, AdamW)
