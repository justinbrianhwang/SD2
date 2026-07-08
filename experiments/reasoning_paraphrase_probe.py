"""Reasoning metric ablation and paraphrase robustness probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from sd2.core.schema import ReasoningState
from sd2.core.stage import Stage
from sd2.metrics import build_metric


METRIC_VARIANTS = (
    "text_embedding_and_intent",
    "reasoning_intent_only",
    "reasoning_text_only",
    "reasoning_critical_object_only",
)


@dataclass(frozen=True)
class ProbePair:
    label: str
    clean: ReasoningState
    stress: ReasoningState


@dataclass(frozen=True)
class ProbeRow:
    metric_type: str
    paraphrase_mean: float
    semantic_change_mean: float
    separation: float


def build_probe_pairs() -> list[ProbePair]:
    """Return a deterministic hand-built reasoning-text probe set."""

    return [
        ProbePair(
            label="paraphrase",
            clean=ReasoningState(
                text=(
                    "A pedestrian is near the crosswalk, so the ego vehicle "
                    "should slow down."
                ),
                intent="slow_down",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="A walker is close to the crossing; reduce speed smoothly.",
                intent="slow_down",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="paraphrase",
            clean=ReasoningState(
                text="The lane ahead is clear, so the ego vehicle should follow the lane.",
                intent="follow_lane",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The roadway ahead is open; keep centered and continue in lane.",
                intent="follow_lane",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="paraphrase",
            clean=ReasoningState(
                text="The traffic light is red, so the vehicle must stop.",
                intent="stop",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The signal requires yielding; bring the car to a halt.",
                intent="stop",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="paraphrase",
            clean=ReasoningState(
                text="A vehicle ahead is slowing, so reduce speed and keep distance.",
                intent="slow_down",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The lead car is decelerating; ease off and leave room.",
                intent="slow_down",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="semantic_change",
            clean=ReasoningState(
                text="The pedestrian is near the crosswalk, so slow down.",
                intent="slow_down",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The pedestrian is near the crosswalk, so maintain speed.",
                intent="maintain_speed",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="semantic_change",
            clean=ReasoningState(
                text="The lane ahead is clear, so follow the lane.",
                intent="follow_lane",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The lane ahead is clear, so stop in the lane.",
                intent="stop",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="semantic_change",
            clean=ReasoningState(
                text="The traffic light is red, so stop before the line.",
                intent="stop",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="The traffic light is red, so proceed before the line.",
                intent="proceed",
                critical_object_mentioned=True,
            ),
        ),
        ProbePair(
            label="semantic_change",
            clean=ReasoningState(
                text="A vehicle ahead is slowing, so slow down.",
                intent="slow_down",
                critical_object_mentioned=True,
            ),
            stress=ReasoningState(
                text="A vehicle ahead is slowing, so accelerate.",
                intent="accelerate",
                critical_object_mentioned=True,
            ),
        ),
    ]


def evaluate_probe(
    pairs: list[ProbePair] | None = None,
    metric_variants: tuple[str, ...] = METRIC_VARIANTS,
) -> list[ProbeRow]:
    """Evaluate each metric variant on paraphrase and semantic-change pairs."""

    probe_pairs = build_probe_pairs() if pairs is None else pairs
    rows: list[ProbeRow] = []
    for metric_type in metric_variants:
        metric = build_metric(Stage.REASONING, {"type": metric_type})
        scores_by_label = {
            "paraphrase": [
                metric.compute(pair.clean, pair.stress).normalized_score
                for pair in probe_pairs
                if pair.label == "paraphrase"
            ],
            "semantic_change": [
                metric.compute(pair.clean, pair.stress).normalized_score
                for pair in probe_pairs
                if pair.label == "semantic_change"
            ],
        }
        paraphrase_mean = mean(scores_by_label["paraphrase"])
        semantic_change_mean = mean(scores_by_label["semantic_change"])
        rows.append(
            ProbeRow(
                metric_type=metric_type,
                paraphrase_mean=paraphrase_mean,
                semantic_change_mean=semantic_change_mean,
                separation=semantic_change_mean - paraphrase_mean,
            )
        )
    return rows


def render_markdown(rows: list[ProbeRow]) -> str:
    """Render probe results as a Markdown report."""

    lines = [
        "# Reasoning Metric Ablation and Paraphrase Probe",
        "",
        "This hand-built probe compares same-intent paraphrases against "
        "same-wording semantic changes. It intentionally exposes the lexical "
        "token-Jaccard limitation in the current `text_embedding` component.",
        "",
        "| Metric variant | Paraphrase mean | Semantic-change mean | Separation |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.metric_type}` | "
            f"{row.paraphrase_mean:.3f} | "
            f"{row.semantic_change_mean:.3f} | "
            f"{row.separation:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: token-set lexical distance is fragile to paraphrase "
            "because same-meaning rewrites can share few words, while intent "
            "mismatch cleanly separates the decision-changing examples. The "
            "default full metric is unchanged; this probe motivates the existing "
            "intent weighting and a future semantic embedding or judge upgrade.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown(rows: list[ProbeRow], output_path: str | Path) -> Path:
    """Write the rendered probe report and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows), encoding="utf-8")
    return path


def main() -> None:
    rows = evaluate_probe()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "docs" / "example" / "reasoning_ablation.md"
    write_markdown(rows, output_path)
    print(render_markdown(rows).strip())
    print(f"\nWrote {output_path.relative_to(repo_root).as_posix()}")


if __name__ == "__main__":
    main()
