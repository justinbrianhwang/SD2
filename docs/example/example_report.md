# SD2 Failure Diagnosis Report

| Field | Value |
| --- | --- |
| Model | transfuser |
| Scenario | Town10HD_Opt_spawn0_dest77 |
| Condition | stress |
| Stress Type | gaussian_noise |
| Severity | 3 |
| Seed | 42 |

## Summary Diagnosis

Under Gaussian Noise severity 3, the transfuser model completed 83.5% of the route and did not record a collision or lane invasion. No stage crossed the critical deviation threshold. Downstream deviation increases followed the Planning onset in the order Control (+0.020). The primary_failure_stage label is Planning. Planning had the highest observed mean deviation (0.234) across the pipeline stages.

Diagnosis type: temporal-correlational; this report identifies the earliest-collapsing stage by timing, not mechanistic proof.

## Final Outcome Comparison

| Metric | Clean | Stress | Delta |
| --- | --- | --- | --- |
| Collision | no | no | n/a |
| Lane invasion | no | no | n/a |
| Route progress | 84.8% | 83.5% | 1.3 pp |

## Stage-wise Mean Deviation

| Stage | Mean | Max | Status | Samples |
| --- | --- | --- | --- | --- |
| Vision | 0.121 | 0.279 | healthy | 120 |
| Semantic | 0.086 | 0.181 | healthy | 120 |
| Planning | 0.234 | 0.521 | healthy | 120 |
| Control | 0.102 | 0.269 | healthy | 120 |

## Collapse Onset Times

| Stage | Warning Onset | Critical Onset |
| --- | --- | --- |
| Vision | n/a | n/a |
| Semantic | n/a | n/a |
| Reasoning | n/a | n/a |
| Planning | t=0.700s, frame 14, score 0.456 | n/a |
| Control | n/a | n/a |

## Propagation Summary

| Edge | Legacy Ratio | Clipped Ratio | Log Ratio | Absolute Increase | Persistence | Lag |
| --- | --- | --- | --- | --- | --- | --- |
| Vision -> Semantic | 0.752 | 0.752 | -0.484 | n/a | n/a | 0 |
| Semantic -> Reasoning | n/a | n/a | n/a | n/a | n/a | 0 |
| Reasoning -> Planning | n/a | n/a | n/a | n/a | n/a | 0 |
| Planning -> Control | 0.561 | 0.561 | -1.169 | -0.135 | 0.000 | 0 |

## Robustness Fingerprint

![Robustness fingerprint](plots/robustness_fingerprint.png)

| Stage | Robustness |
| --- | --- |
| Vision | 0.879 |
| Semantic | 0.914 |
| Reasoning | n/a |
| Planning | 0.766 |
| Control | 0.898 |
| Mean | 0.864 |
| Run count | 1 |

```text
transfuser Robustness Fingerprint

Vision:      [#########-] 0.88
Semantic:    [#########-] 0.91
Reasoning:   [??????????] n/a
Planning:    [########--] 0.77
Control:     [#########-] 0.90
```

## Embedded Plots

![Stage-wise deviation timeline](plots/deviation_timeline.png)

![Propagation scores](plots/propagation_scores.png)
