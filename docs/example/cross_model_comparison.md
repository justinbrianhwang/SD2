# Cross-Architecture Failure Comparison (InterFuser vs TransFuser)

SD2 diagnoses two E2E driving models under the **same** visual stress
(Gaussian noise, severity 3, Town10HD, matched seed/route) and asks RQ3:
*do different architectures fail at different pipeline stages?*

## Stage-wise mean deviation (clean vs stress)

| Stage | InterFuser | TransFuser |
| --- | ---: | ---: |
| vision | 0.026 (robust) | 0.159 (sensitive) |
| semantic | 0.233 (collapses) | 0.096¹ |
| planning | 0.073 | 0.278 |
| control | 0.060 | 0.214 |
| **primary failure stage** | **semantic** | **planning** |

## Interpretation

- **InterFuser** has a noise-**robust visual encoder** (vision deviation only
  0.026), but the small visual change is **amplified at the semantic stage**
  (0.233, crossing critical) — its object-density decoder is the fragile point.
  Failure is *localized and amplified* downstream of a stable perception.
- **TransFuser**'s fused image+LiDAR feature is itself **noise-sensitive**
  (vision 0.159), and the perturbation propagates fairly uniformly into
  planning (0.278) and control (0.214). Failure *originates earlier* (vision)
  and moves through the whole chain together.

The two architectures fail differently under identical stress: InterFuser's
weak point is semantic scene understanding on top of a robust encoder;
TransFuser's weak point is the perception feature itself. This is the kind of
architecture-level robustness fingerprint SD2 is built to expose.

## Honest caveats

1. **¹ TransFuser semantic**: TransFuser's bounding-box detection head needs
   `mmdet` (not installed on the torch-2.11 stack), so the box-based `objects`
   list is empty and SD2's default `object_jaccard` semantic metric reads 0.
   The 0.096 figure is the total-variation distance of TransFuser's **BEV
   semantic-segmentation** class distribution (computed from the recorded
   `bev_seg_summary`), which does not need `mmdet`. A seg-distribution semantic
   metric should be added to SD2 so this is captured in the standard pipeline.
2. **TransFuser driving**: in this run TransFuser mostly held station
   (route_progress ≈ 0.01, control saturated to brake) — its planning/control
   deviations are therefore measured on a near-stationary ego and should be
   read as model-output sensitivity, not closed-loop driving divergence. The
   vision and BEV-seg deviations are unaffected by this (the model still
   processes every camera frame). Getting TransFuser to drive through the route
   is follow-up work before the planning/control comparison is fully clean.
