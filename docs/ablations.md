# Ablations

This repo is built around measurable retrieval changes rather than feature accumulation.

## Lexical fusion

Two variants were tested:

- additive lexical bonus
- multiplicative lexical fusion

Multiplicative fusion consistently performed better on our internal evaluation and on LongMemEval.

Representative LongMemEval numbers:

| Variant | R@5 | R@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Raw | 92.8 | 96.4 | 0.846 |
| Hybrid | 96.8 | 98.0 | 0.920 |
| Hybrid + topic | 97.0 | 98.4 | 0.922 |

## Topic boost

Topic signals were mixed.

- On one internal evaluation, topic boost hurt entity-heavy queries.
- On LongMemEval, a small topic boost improved the final score slightly.

The takeaway is not that topic signals are always good or always bad. They are workload-dependent and should stay configurable.

## Temporal penalty

Removing the temporal penalty made current-fact retrieval worse. Keeping historical facts in the store is useful, but they should not outrank the current version when the query is asking about the present.

## Ideas deliberately rejected

- spellcheck before storage
- Wikipedia-based entity disambiguation
- always-on LLM reranking in production

These were either too expensive, too noisy, or not worth the operational complexity.
