# LSTM-FC Hypertuning Results

## Summary

Systematic hyperparameter tuning of the LSTM-FC model for human action prediction
(left/straight/right) on walking trajectory data.

## Best Configuration

| Parameter | Original Default | Tuned Value |
|-----------|-----------------|-------------|
| hidden_size | 64 | 96 |
| num_layers | 1 | 1 |
| dropout | 0.0 | 0.1 |
| batch_size | 64 | 512 |
| learning_rate | 0.01 | 0.006 |
| scheduler | StepLR | OneCycleLR |
| training_steps | 100,000 | 50,000-100,000 |

## Results Comparison

| Model | MSE | Accuracy | Parameters |
|-------|-----|----------|------------|
| Original (500 steps) | 0.0441 | 69.5% | 17,603 |
| Tuned h=96 (50k steps) | 0.0140 | 87.8% | 38,691 |
| Tuned h=128 (100k steps) | 0.0134 | 88.0% | 67,971 |

## Key Findings

1. **Dropout (0.1)** provides significant regularization benefit
2. **Larger batch sizes (512)** dramatically improve convergence stability
3. **OneCycleLR scheduler** outperforms StepLR and CosineAnnealing
4. **Learning rate 0.005-0.006** is optimal for Adam optimizer
5. **Single LSTM layer** outperforms deeper architectures
6. **Bidirectional and Deep FC variants** showed no improvement
7. **MSE loss** outperformed KL-Divergence and Cross-Entropy with soft labels
8. **Weight decay** did not help (dropout was sufficient)

## Tuning Process

- **Round 1**: 50 trials across 4 architectures (base, dropout, bidir, deep),
  hidden sizes [32-256], LRs [0.0005-0.02], batch sizes [32-256],
  loss functions [MSE, KL, CE], schedulers, dropout rates, weight decay
- **Round 2**: 62+ trials with fine grid around best config from round 1,
  hidden sizes [96-256], LRs [0.003-0.008], batch sizes [128-512],
  schedulers [step, cosine, onecycle], 10k steps per trial
- **Final**: Long training runs (50k-100k steps) with best configuration

## Per-Class Performance (Best Model)

| Class | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Left | 70.9% | 87.0% | 70.9% | 78.1% |
| Straight | 95.3% | 88.4% | 95.3% | 91.7% |
| Right | 72.0% | 86.6% | 72.0% | 78.7% |
