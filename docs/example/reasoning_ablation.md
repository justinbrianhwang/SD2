# Reasoning Metric Ablation and Paraphrase Probe

This hand-built probe compares same-intent paraphrases against same-wording semantic changes. It intentionally exposes the lexical token-Jaccard limitation in the current `text_embedding` component.

| Metric variant | Paraphrase mean | Semantic-change mean | Separation |
| --- | ---: | ---: | ---: |
| `text_embedding_and_intent` | 0.430 | 0.458 | 0.028 |
| `reasoning_intent_only` | 0.000 | 1.000 | 1.000 |
| `reasoning_text_only` | 0.861 | 0.317 | -0.544 |
| `reasoning_critical_object_only` | 0.000 | 0.000 | 0.000 |

Interpretation: token-set lexical distance is fragile to paraphrase because same-meaning rewrites can share few words, while intent mismatch cleanly separates the decision-changing examples. The default full metric is unchanged; this probe motivates the existing intent weighting and a future semantic embedding or judge upgrade.
