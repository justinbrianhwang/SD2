# SD2 Synthetic Fault Injection Benchmark

## Primary Failure Stage Diagnosis Accuracy: 100.0%

This benchmark is a framework sanity check: synthetic clean/stress run pairs are generated with a known primary failure stage, then SD2 scores only the diagnosis returned by the real offline analysis pipeline.

## Per-class Accuracy

| Class | Accuracy | Samples |
| --- | --- | --- |
| Vision | 100.0% | 20 |
| Semantic | 100.0% | 20 |
| Reasoning | 100.0% | 20 |
| Planning | 100.0% | 20 |
| Control | 100.0% | 20 |

## Confusion Matrix

![Confusion matrix](confusion_matrix.png)

| True \ Predicted | Vision | Semantic | Reasoning | Planning | Control | No Failure |
| --- | --- | --- | --- | --- | --- | --- |
| Vision | 20 | 0 | 0 | 0 | 0 | 0 |
| Semantic | 0 | 20 | 0 | 0 | 0 | 0 |
| Reasoning | 0 | 0 | 20 | 0 | 0 | 0 |
| Planning | 0 | 0 | 0 | 20 | 0 | 0 |
| Control | 0 | 0 | 0 | 0 | 20 | 0 |

## Common Confusions

No off-diagonal confusions were observed. This indicates that the synthetic injections match the current diagnosis policy assumptions.
