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
2. **TransFuser driving (resolved)**: earlier runs held station
   (route_progress ≈ 0.01, control saturated to brake), so planning/control
   deviations were measured on a near-stationary ego. Live `--debug-driving`
   diagnosis traced this to a cold-start crawl limit-cycle (short predicted
   waypoints → low desired speed → brake), *not* the LiDAR safety brake
   (`emergency_stop=False` throughout) or a wrong target-point frame. Engaging
   TransFuser's own creep controller in the crawl regime
   (`--tf-creep-speed 2.5 --tf-stuck-threshold 5 --tf-creep-duration 60`) makes it drive
   the route at ~4 m/s and complete ~85% of it (matching NEAT), so the
   planning/control comparison is now on a properly moving ego. See the README
   "Making TransFuser drive (anti-crawl creep)" section. The same cold-start
   crawl affects the AIM/CILRS/TCP camera baselines (NEAT escapes it on its own).
