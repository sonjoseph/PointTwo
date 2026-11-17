# PointTwo evaluation report

## Jev versus similarity and weighted scoring

On November 17, 2026, the paired synthetic evaluation improved held-out top-one ranking from **87.9% with the weighted baseline to 95.0% with Jev**. Jev cleared the review gates for **65.0%** of held-out entries, including **one unsupported assignment**. These results support further evaluation of text-based decisions after retrieval; they do not justify automatic assignment or establish production accuracy. The application now supports the same Jev decision protocol after explicit opt-in, with the weighted method retained as a local baseline.

### Shared retrieval, different decisions

Both decision pipelines use the same Qwen embeddings, retrieved catalog records, and confirmed-example evidence. Retrieval requests **50 catalog neighbors and 100 confirmed-example neighbors**, unions their matter labels with explicit references, then selects **10 matters** independently of the linear weights. Jev receives their descriptions and up to three distinct confirmed narratives per matter, plus the original query. It never receives the expected answer, imported assignment, or baseline score.

The production design searches indexed collections of 50,000+ records and potentially thousands of distinct client–matter pairs, but **Jev judges only the shortlist**, not every record or pair. Neighbor counts are search budgets, not exhaustive comparisons. This quality experiment uses **108 matters**, so Postgres selected exact scans for the small catalog; HNSW performance at 50,100 vectors is a separate measurement below. We have not measured Jev quality with 1,000+ candidate classes or 50,000 real catalog records.

| Approach | Uses the same ten candidates | How the final order is determined |
| --- | --- | --- |
| Catalog cosine only | Yes | Catalog embedding similarity; ranking diagnostic without a confidence policy |
| Weighted baseline (`random-42-v3`) | Yes | Catalog/example similarity plus explicit client-name and matter-reference features |
| Jev (`jev-1.14.0`) | Yes | Original narrative, candidate descriptions, and confirmed-example text; Choice probabilities plus an evidence-sufficiency check |

The 532 no-history cases comprise 192 development and 340 held-out cases. Three separate before/after feedback probes test retrieved examples. The held-out set has already been inspected in earlier evaluations, so it is a **regression set, not a fresh release test**. Model, prompt (`client-matter-choice-v1`), and gates (0.7 probability, 0.1 margin, 0.7 sufficiency) were fixed before the full run. No weights were trained and no thresholds were tuned to these results.

### Paired results

| Held-out measure | Catalog cosine only | Weighted baseline | Jev |
| --- | ---: | ---: | ---: |
| Top-one, 240 answerable cases | 50.8% (122/240) | 87.9% (211/240) | **95.0% (228/240)** |
| Top-five, 240 answerable cases | 75.0% (180/240) | 99.2% (238/240) | **100% (240/240)** |
| Correct matter retained in shortlist | 100% | 100% | 100% |
| Ambiguous/unsupported abstention, 100 cases | — | 100% (100/100) | 99% (99/100) |
| Coverage without low confidence, 340 cases | — | 0% (0/340) | 65.0% (221/340) |
| Incorrect suggestions clearing review gates | — | 0 | 1 |
| Cached classifier + API, median / p95 | Not timed separately | 40.0 / 60.8 ms | 123.9 / 168.7 ms |

Rank accuracy includes cases flagged for review. Jev's ranking excludes the insufficient-evidence option, while its review decision includes it. Returned probabilities often round alternatives to zero, so deterministic tie-breaking orders those alternatives; top-five accuracy should not be read as five well-differentiated choices. Every expected answer in this run had positive probability. Neither weighted scores nor Jev probabilities are calibrated correctness estimates.

Across all 432 answerable development/held-out cases, top-one accuracy was **63.4% cosine-only, 84.0% weighted, and 97.2% Jev**. Jev fixed 68 weighted-baseline top-one errors and introduced 11; both missed one case. The full weighted candidate set and ten-candidate weighted shortlist had identical top-one/top-five accuracy. All 432 expected matters survived retrieval and shortlisting. That does not guarantee recall on a larger catalog.

The paired run completed with **zero API/protocol failures**. Timing sums the cached local classifier duration and, for Jev, the HTTP request. It excludes shortlist/prompt assembly, response validation, UI overhead, and uncached embedding inference; it is not a full user-visible response-time benchmark. Three blind responses reused the smoke-test cache and were excluded from live latency aggregates. The detailed replay reused all responses and is not a second speed measurement. This compares the seeded baseline with Jev, not a trained logistic-regression scorer or a strong cross-encoder baseline.

### Where reading the text beats embedding order

These are actual cases from the run, selected afterward to illustrate behavior. The text explanations below are our interpretation of the evidence, not Jev's internal reasoning trace. Neither example had confirmed history.

**Employee compensation — `corpus-0009`:** “Reviewed employee compensation agreements for Alder Robotics.”

| Candidate | Catalog cosine | Weighted score | Jev probability |
| --- | ---: | ---: | ---: |
| Alder / Series A financing | **0.5981** | **0.3023** | 0.00 |
| Alder / Employment advice | 0.5562 | 0.2938 | **1.00** |

The financing description mentions equity investment and subscription agreements; the employment description explicitly covers employee agreements and compensation. Both candidates match the client name and have no example support, so the weighted scorer preserves their cosine order. Jev selects the employment matter from the text, with sufficiency **0.92**. Its reported 1.00 is not a guarantee of correctness.

**Borrowing versus equity — `corpus-0029`:** “Revised financial covenants in the credit facility for Alder Robotics.”

| Candidate | Catalog cosine | Weighted score | Jev probability |
| --- | ---: | ---: | ---: |
| Alder / Series A financing | **0.6561** | **0.3140** | 0.00 |
| Alder / Credit facility | 0.6520 | 0.3131 | **0.99** |

Embedding closeness slightly favors the equity-financing matter. The credit-facility description covers secured borrowing, credit covenants, lender security, and loan agreements, which specifically matches the work described. Jev selects that matter, with sufficiency **0.95**. These examples demonstrate useful text discrimination; a separate controlled set is still needed for negation, conflicting references, and harder relational reasoning.

### Confirmed examples still matter

The unchanged Alder probe (`corpus-0041`) says: “Updated memo 1 about vesting schedules and founders share options.” A reviewer then confirms a different entry (`corpus-0040`) as Alder / Series A financing: “Prepared the 1th founders memo on option vesting and equity grants.” The synthetic wording is preserved exactly.

Its confirmed-example cosine is **0.7485**, versus **0.3978** for the catalog description. Before feedback, Alder's financing matter was outside the ten-candidate shortlist; afterward, example retrieval brings it into the shortlist and supplies the source narrative to Jev. No embedding model or decision model is retrained.

| Fixed probe | Expected matter in shortlist, before → after | Weighted target score, before → after¹ | Jev response, before → after |
| --- | --- | ---: | --- |
| Alder (`corpus-0041`) | No → Yes | 0.0802 → 0.1658 | Insufficient evidence (0.66) → Alder (0.99) |
| Birch (`corpus-0082`) | No → Yes | 0.0702 → 0.1565 | Insufficient evidence (0.90) → Birch (0.99) |
| Cobalt (`corpus-0123`) | Yes → Yes | 0.0842 → 0.1705 | Insufficient evidence (0.83) → Cobalt (1.00) |

¹ Baseline scores refer to the full retrieved set, including matters omitted from the shortlist. All three post-feedback Jev decisions cleared the experimental gates. Both the weighted scorer and Jev ranked the confirmed target first afterward. This shows feedback reaches both pipelines; **it does not demonstrate Jev outperforming weighted scoring on feedback**. Alder and Birch change the shortlist as well as the evidence, so this is not an isolated example-text ablation. Similar narratives without explicit client identity remain fallible evidence, even after approval of a related entry.

### Failures and limits

- **Unsupported identity (`corpus-0494`):** “Reviewed the capitalization schedule.” Jev chose Cobalt / Series A financing with probability **0.93**, sufficiency **0.81**, and no low-confidence flag. The work type fits, but the narrative never identifies Cobalt. Only one Series A matter survived this shortlist, which can make local certainty misleading. The weighted baseline abstained. Evaluate shortlist diversity and missing-client cases before production; a model cannot recover omitted competing matters.
- **Regression against the fixture (`corpus-0003`):** “Negotiated protections for incoming shareholders for Alder Robotics.” The fixture expects Series A; the weighted baseline ranks it first. Jev ranks Series B first (**0.44**, versus **0.20** for Series A) and abstains. The text does not identify the financing round, exposing a possible fixture-label ambiguity. All 11 baseline-correct/Jev-wrong regressions use this template; retain their errors in the reported metrics and audit the labels separately.
- **Generalization:** Synthetic templates repeat across fictional clients. The three feedback probes are mechanism checks. No matched-coverage comparison, probability calibration study, candidate-order robustness study, 1,000-class Jev evaluation, or fresh legal-domain test has been completed.

### Reproduce and inspect

Run `python -m classifier.compare_jev` with the local database configured to prepare requests without network inference. Add `--execute` with `TYPESAFE_API_KEY` configured privately to send **only synthetic fixture data**. `--limit 3` runs three blind cases plus the six before/after feedback requests. The app's review data is not sent.

Reports contain per-case narratives, candidate descriptions, cosine/features, baseline scores, Jev responses, and retrieval coverage. `.requests.json` files preserve the exact model inputs; successful responses are cached by request/model/prompt identity. The original live run is `.pointtwo/python/jev-comparison-live.json`; its cached evidence replay is `.pointtwo/python/jev-comparison-detailed.json`. These artifacts and credentials remain outside Git. The source corpus, comparator, and protocol tests are in the repository.

The comparison-stage suite passed **47 tests**, including shared-shortlist behavior, exclusion of expected/imported labels, malformed-response rejection, explicit review overrides, retrieval misses, API-error accounting, and redacted errors. The provider's [Choice](https://docs.typesafe.ai/primitives/choice) and [Noul](https://docs.typesafe.ai/primitives/noul) documentation describes the response types; those outputs do not expose a reasoning trace.

### Application integration checks

The Jev-enabled app preserves ranked alternatives on abstention, exposes model probabilities separately from evidence sufficiency, and sends retrieved confirmed narratives with the candidate descriptions. Hosted inference runs outside database locks; changed evidence or edited source records invalidate the response before publication. Pipeline changes refresh unapproved entries once and preserve accepted assignments and historical runs. Protocol failures are retryable processing errors, without a weighted fallback. The integration suite uses a deterministic Jev stub with real Postgres, so these checks measure application behavior rather than model quality. The complete suite passes **55 tests**, and the TypeScript/Vite production build passes. A live app restart refreshed **563 unapproved entries** with Jev while retaining **4 accepted assignments**; no entries remained failed or queued after processing. In the native review UI, `demo-004` selected insufficient evidence (0.96) while retaining five alternatives and remaining unassigned. The debugger exposed Jev probabilities, supplied texts, and the embedding neighborhood. These are functional checks in a workspace with existing review history, not a new accuracy evaluation.

---

The following sections preserve earlier baseline evaluations and the separate ANN scale measurements.

## Evaluation without catalog timekeeper involvement (`random-42-v3`)

Re-ran all **532 blind cases** and three feedback probes on November 17, 2026. The scorer now has four features: catalog similarity, confirmed-example similarity, explicit client name, and matter reference. `InvolvedTimekeepers` remains catalog metadata; it contributes no score. The remaining coefficients and review thresholds are unchanged.

| Held-out measure | v3 result |
| --- | ---: |
| Top-one / top-five accuracy (240 answerable cases) | 87.9% / 99.2% |
| Correct-matter candidate coverage | 100% |
| Ambiguous / unsupported cases flagged (100 cases) | 100% |
| Ready-for-review coverage (340 cases) | 0% |
| Cached end-to-end median / p95 | 32.7 / 53.9 ms |

The corpus has empty `InvolvedTimekeepers` lists, so unchanged accuracy is a regression check, not evidence that removing this feature improves accuracy. The three feedback score increases also match v2. The confidence policy remains too conservative on the held-out cases. Raw results: `.pointtwo/python/evaluation-v3.json`; these are local cached timings, not cold-cache measurements.

A separate live demo probe with existing Acme involvement fell from **0.6504 to 0.3963**, removing the **0.254161** involvement contribution while retaining catalog, example, and client-name evidence. The [demo screenshot](DEMO.md#inspect-a-score) shows the four-feature breakdown. Regression tests check that listing the timekeeper changes neither features nor score, and that upgrading v1/v2 queues unapproved entries while preserving accepted assignments and saved runs.

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

The benchmark command measures the selected workspace and reports its actual size. Repeating the historical 50,100-vector measurement requires an equivalently prepared catalog; the machine-local scale fixture is not included in a fresh clone. The retained `scale.json` records the original build and effort comparison; the portable benchmark writes `benchmark.json`. See [setup](../README.md), [demonstration](DEMO.md), and [design](../CLASSIFICATION_DESIGN.md).
