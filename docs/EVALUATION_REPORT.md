# PointTwo evaluation report

The classifier ranked the correct matter first in **211 of 240 answerable held-out cases (87.9%)**, and in its top five in **238 (99.2%)**. However, **all 340 held-out cases required manual review**. These results support candidate retrieval as a review aid; they do not establish a calibrated automatic-assignment system.

## Evaluation after removing the availability bonus (`random-42-v2`)

Removed the fixed `has_example_match` score contribution while preserving every other coefficient. Example availability remains diagnostic context for review rules; it does not add ranking points. Saved v1 traces remain immutable, and existing unapproved entries are queued for rescoring when the demo scorer upgrades.

A new real-Qwen run on the same isolated synthetic corpus produced **87.9% held-out top-one accuracy, 99.2% top-five accuracy, 100% ambiguous-case abstention, and 0% coverage without low confidence**. These blind metrics are unchanged because the blind evaluation starts without confirmed examples. They do not measure the quality of feedback-based ranking.

| Feedback matter | Before feedback | v1 after (with bonus) | v2 after (similarity only) | v2 increase |
| --- | ---: | ---: | ---: | ---: |
| Alder Robotics | 0.0802 | 0.3894 | 0.1658 | +0.0856 |
| Birch Energy | 0.0702 | 0.3802 | 0.1565 | +0.0863 |
| Cobalt Analytics | 0.0842 | 0.3942 | 0.1705 | +0.0864 |

For Alder, example similarity remains **0.7485**, contributing **0.085574** at weight **0.114333**. The **0.223675** availability bonus is gone. This reduces the final probe score from **0.3894** to **0.1658**, without re-embedding or retraining. This mechanism check does not establish that the remaining random weights rank feedback correctly in every case.

The new held-out cached run measured **33.5 ms median / 52.5 ms p95**; concurrent local work and cache state prevent treating timing differences as a scorer speed regression. Full output is saved in `.pointtwo/python/evaluation-v2.json`.

**The detailed measurements and original worked example below record v1**, retained for comparison. Their bonus-based scores are historical, not the active scorer's output.

## Setup

- Local Apple M1 development machine, 8 GB unified memory; Python, Postgres 17 and pgvector 0.8.7. Other development processes were running.
- Qwen3-Embedding-0.6B, 1,024 dimensions; fixed `random-42-v1` scoring weights. No model or scorer training in this quality run.
- 108 fictional catalog matters and 556 synthetic time entries: 532 blind cases, plus 12 feedback source/probe pairs. Three pairs were exercised separately.
- The blind split contains 192 development cases and 340 held-out cases. Held-out explicit-client narratives use different templates. This is a synthetic template split, not a chronological or unseen-client production test.
- Expected labels are used only by evaluation. The classifier receives the prompt fields; imported assignments are deliberately unreliable. Quality evaluation uses an isolated workspace without personal review history.

## Held-out metrics explained

“Held out” means these cases were reserved for final evaluation rather than used to choose weights or confidence thresholds. This demo uses fixed random weights, so there was no classifier training; the split separates development examples from the final check. It is not evidence that Qwen never encountered similar text during pretraining.

The held-out set has **340 entries: 240 with a single expected matter and 100 ambiguous or unsupported entries**.

| Metric | Result | Meaning |
| --- | ---: | --- |
| Top-one accuracy | 87.9% (211/240) | The first-ranked matter was correct on answerable entries, even if flagged as low confidence. |
| Top-five accuracy | 99.2% (238/240) | The correct matter appeared among the first five candidates on answerable entries. |
| Ambiguous/unsupported abstention | 100% (100/100) | Every deliberately unresolved case was flagged for review. |
| Coverage without a low-confidence flag | 0% (0/340) | No held-out entry cleared the confidence policy. This includes all 240 answerable cases. |

The former UI label “ready-for-review coverage” meant the last row; it did not mean that other entries were unavailable for review. Ranking quality and the review policy are separate: the classifier often ranked the right matter first, but its conservative policy flagged **every** held-out result. Perfect ambiguous-case abstention therefore does not establish useful confidence discrimination. These are synthetic results, not production legal-billing accuracy.

## Classification quality

Top-one/top-five accuracy measures ranking on answerable cases, including results flagged for review. Candidate coverage asks whether the correct matter appeared anywhere in the retrieved candidate set. Abstention means the result was flagged as low confidence; ranked alternatives may still be shown.

| Measure | Development | Held out |
| --- | ---: | ---: |
| Cases | 192 | 340 |
| Answerable / ambiguous or unsupported | 192 / 0 | 240 / 100 |
| Top-one accuracy | 79.2% (152/192) | 87.9% (211/240) |
| Top-five accuracy | 96.4% (185/192) | 99.2% (238/240) |
| Correct-matter candidate coverage | 100% | 100% |
| Ambiguous/unsupported flagged for review | — | 100% (100/100) |
| Cases not flagged as low confidence | 4.2% (8/192) | 0% (0/340) |

The review policy is too conservative for useful automatic triage on this held-out set. Perfect abstention on ambiguous cases must be read alongside zero coverage on answerable cases. No weights or thresholds were tuned on held-out results.

### Breakdown across both blind splits

| Case type | Cases | Top-one | Top-five | Ambiguous/unsupported flagged |
| --- | ---: | ---: | ---: | ---: |
| Cross client ambiguous | 32 | — | — | 100.0% |
| Explicit client | 288 | 78.8% | 97.2% | — |
| French | 12 | 91.7% | 100.0% | — |
| Matter reference | 96 | 99.0% | 100.0% | — |
| Multiple clients | 12 | — | — | 100.0% |
| Out of domain | 8 | — | — | 100.0% |
| Same client ambiguous | 12 | — | — | 100.0% |
| Same client specific | 12 | 100.0% | 100.0% | — |
| Spanish | 12 | 100.0% | 100.0% | — |
| Typo | 12 | 50.0% | 91.7% | — |
| Unknown client | 24 | — | — | 100.0% |
| Vague | 12 | — | — | 100.0% |

Typos were a clear weakness: 6 of 12 correct at rank one. Spanish and French results come from only 12 examples each and do not establish broad multilingual quality. Missing client identity, multiple clients, unknown clients, unrelated work, and indistinguishable same-client matters were evaluated as unresolved cases rather than assigned arbitrary ground truth.

## Feedback effects

For each experiment, save a probe result, confirm a **different related source entry** for the same timekeeper, then reevaluate the original probe. Weights and embedding model stay fixed.

| Confirmed matter | Source → probe | Before | After | Score change |
| --- | --- | ---: | ---: | ---: |
| Alder Robotics / Series A financing | corpus-0040 → corpus-0041 | 0.0802 | 0.3894 | +0.3092 |
| Birch Energy / Series A financing | corpus-0081 → corpus-0082 | 0.0702 | 0.3802 | +0.3100 |
| Cobalt Analytics / Series A financing | corpus-0122 → corpus-0123 | 0.0842 | 0.3942 | +0.3100 |

### Worked example: Alder Robotics

**Fixed probe (`corpus-0041`, timekeeper `feedback-0`):**

> Updated memo 1 about vesting schedules and founders share options.

**Different source entry confirmed as Alder Robotics / Series A financing (`corpus-0040`):**

> Prepared the 1th founders memo on option vesting and equity grants.

These are the exact synthetic fixture narratives. Before feedback, Alder’s score for the probe was **0.0802**, supported only by catalog similarity. Accepting the source entry made its embedding available as a labeled example for that timekeeper. Reclassifying the unchanged probe retrieved it at cosine similarity **0.7485** (distance **0.2515**).

| Feature | Before → after | Fixed weight | Weighted contribution before → after |
| --- | ---: | ---: | ---: |
| Catalog similarity | 0.397754 → 0.397754 | 0.201625 | 0.080197 → 0.080197 |
| Confirmed-example similarity | 0.000000 → 0.748459 | 0.114333 | 0.000000 → 0.085574 |
| Confirmed example available | 0.000000 → 1.000000 | 0.223675 | 0.000000 → 0.223675 |
| Client name, matter reference, timekeeper context, and bias | Unchanged at zero | Unchanged | 0 → 0 |

Using the unrounded values, the score changed from **0.080197** to **0.389445**: catalog contribution + example-similarity contribution (**0.085574**) + example-available contribution (**0.223675**). The increase was **0.309248**, displayed as **+0.3092**. Rounded table values may differ slightly when summed.

In the debugger, the confirmed example appeared as a new point connected to Alder’s catalog record. Switching between baseline and feedback reused the same projection axes; the original embeddings and scoring weights stayed unchanged. Restoring the source’s previous unassigned state removed its support and returned Alder’s probe score to **0.0802**. The probe itself was never accepted or reassigned.

This is an observed feedback mechanism, not retraining or a calibrated probability increase. The narrative omits client identity, so a related example can raise Alder’s ranking without proving the probe belongs to Alder. The demo still flags it for review.

Three successful mechanism checks do not measure whether feedback improves accuracy generally.

## Performance and ANN scale

The repeated 532-case quality run measured **28.4 ms median / 37.2 ms p95** inside the classifier, with cached embeddings and a small catalog. Postgres chose exact search for that small dataset. These measurements exclude UI/network overhead and do not represent uncached inference or HNSW performance at scale.

A separate benchmark reused **50,100 prepared synthetic catalog vectors**. Building the workspace HNSW index took **94.9 seconds**; the index occupied **391 MB** (excluding vector tables, cache, and model storage). Six cached queries were compared with exhaustive exact search.

| HNSW search effort | Mean recall@30 |
| --- | ---: |
| `ef_search=200` | 93.3% |
| `ef_search=400` (selected) | 95.0% |
| `ef_search=1000` | 97.2% |

At 400, individual recall ranged from **90% to 100%**, with **276.3–473.5 ms** measured query times. Cache/order effects make these timings unsuitable for comparing effort settings as a controlled speed benchmark. HNSW use was verified in query plans. Neighbor recall measures agreement with exact neighbors, not correct-matter assignment.

One complete 50k application query took **959 ms overall**, including **700 ms retrieval/scoring**. Its correct Acme candidate was recovered through explicit-reference retrieval after ANN omitted it. That sample **missed the proposed 250 ms retrieval/ranking target**. It is one query, not a p95 measurement.

## Functional and visualization checks

The updated suite passes **31 tests** using isolated Postgres workspaces and deterministic embeddings. Coverage includes score arithmetic, import/cache reuse, accept/correct/reject/undo, duplicate evidence, timekeeper/workspace isolation, stale revisions, model changes, immutable runs, and candidate training. Projection checks verify full-dimensional cosine distance, fixed baseline axes, relabeling without coordinate changes, historical evidence after undo, missing-vector handling, and API authorization. Additional checks cover fixture initialization without overwriting edits/reviews, exact EntryId lookup, open-entry refresh without changing accepted assignments, and rejecting stale in-flight scores after feedback. Checks also cover removing presence-only score contributions, one-time scorer upgrades preserving decisions/history, and rebuild-triggered refresh of open entries. The TypeScript production build passes. In the native application, the synthetic Alder source/probe workflow reproduced the 0.0802 → 0.3894 score change, displayed the new example and its 0.7485 cosine similarity, and restored the original assignment and 0.0802 score. The Demo guide navigation item is removed.

The debugger uses a bounded PCA view of saved evidence. It shows up to eight ranked matters and three retrieved confirmed examples per matter. Two-dimensional distance is approximate; inspected cosine values use all 1,024 dimensions. Assignment lines are not HNSW edges. These are visualization correctness checks, not a new accuracy study.

## Limits and reproduction

These are fictional, template-generated cases. The 108-matter accuracy study and 50,100-vector recall study are separate; no 50k classification-accuracy result is claimed. There was no controlled OS cold-cache reset, concurrency/load study, representative production feedback training, or complete process-memory measurement. Weekly training and calibrated release gates remain proposed work.

Measured source artifacts are `.pointtwo/python/evaluation.json` and `.pointtwo/python/scale.json`; private local artifacts are ignored by Git. The JSON includes per-case results, model identity, and scale query details. The report records those runs; adding the debugger did not rerun or change the quality scores.

```sh
# With DATABASE_URL configured privately:
.pointtwo/classifier-venv/bin/python -m classifier.evaluate
.pointtwo/classifier-venv/bin/python -m classifier.benchmark --workspace local-review
.pointtwo/classifier-venv/bin/python -m pytest tests
npm --prefix web run build
```

The benchmark command measures the selected workspace and reports its actual size. Repeating the historical 50,100-vector measurement requires an equivalently prepared catalog; the machine-local scale fixture is not included in a fresh clone. The retained `scale.json` records the original build and effort comparison; the portable benchmark writes `benchmark.json`. See [setup](../README.md), [demonstration](DEMO.md), and [design](../DESIGN.md).
