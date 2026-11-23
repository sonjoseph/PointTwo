# PointTwo evaluation report

## Pool without a full-catalog name search

On November 22, 2026, the candidate pool stopped searching every matter's client name and matter ID. The pool is now only:

- the **50** nearest catalog matters (HNSW, `CATALOG_NEIGHBORS` in `classifier/engine.py`);
- matters of this timekeeper's 100 nearest confirmed examples (HNSW);
- matters with an earlier accept or reject for the same normalized narrative.

The capped shortlist slots are unchanged, but they apply only to pool members whose client name or matter ID appears in the narrative. The removed search could not use an index, so it read every catalog row on every query. It also stopped at 1,000 matches in arbitrary ID order, which could drop the right matter when a common-word matter ID such as `employment` matched every client.

### Corpus results (`jev-comparison-v5.json`)

Model, prompt, gates, and baseline weights are unchanged. Only **1 of 568** shortlists changed, so every Jev response came from cache; no new requests were sent.

| Held-out measure (376 cases, 276 answerable) | Catalog cosine only | Weighted, same shortlist | Jev | Jev before (full-catalog name search) |
| --- | ---: | ---: | ---: | ---: |
| Top-one | 54.7% (151) | 87.0% (240) | **93.5% (258)** | 93.8% (259) |
| Top-five | 77.9% (215) | 98.6% (272) | **99.6% (275)** | 100% (276) |
| Correct matter in shortlist | 99.6% (275) | 99.6% | 99.6% | 100% |
| Ambiguous/unsupported abstention, 100 cases | — | 100% | 99% | 99% |
| Coverage without low confidence, 376 cases | — | 0% | 66.0% (248) | 66.2% (249) |
| Incorrect suggestions clearing review gates | — | 0 | 2 | 2 |

The one change is `corpus-0139`, “Updated contractual obligations of the data processor; file PT-03-03.” The referenced Delta Foods privacy matter fell just outside the 50 nearest, so no matter-ID slot could take it. Jev chose insufficient evidence (0.97) and the entry was flagged for review rather than misassigned.

### Scale results (`shortlist-scale-v2.json`)

Expected matter among the ten options, for queries that name the client exactly (480 per size; 96 at 12 clients). Real local Qwen embeddings, exact cosine search, no Jev.

| Clients (matters) | Similarity only | Pool 50 (current) | Pool 100 | Pool 200 | Pool 400 | Full-catalog name search |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 (108) | 95 | 96 | 96 | 96 | 96 | 96 |
| 100 (900) | 475 | 479 | 480 | 480 | 480 | 480 |
| 300 (2,700) | 467 | 477 | 479 | 480 | 480 | 480 |
| 1,000 (9,000) | 457 | **473** | 477 | 479 | 479 | 480 |

- All seven misses at 1,000 clients with a 50-matter pool use one template, “Amended obligations owed to the commercial landlord for <client>”, with similarity ranks 54–421.
- A 200-matter pool recovers all but one; the remaining miss is Lark Analytics / Office lease at similarity rank **421**, which even a 400-matter pool misses.
- Misspelled client names were unchanged under every rule (95, 478, 470, 442), since exact matching never helped them.

**Recommendation:** raise `CATALOG_NEIGHBORS` from 50 to 200. On these catalogs it matches the full-catalog name search at up to 2,700 matters and loses one query in 480 at 9,000, using only indexed search. HNSW already explores 400 candidates per query (`ef_search = 400`). Coverage at 50,000 matters and query latency at 200 are not yet measured.

The sections below record earlier runs, which used the full-catalog name search.

## Shortlist with capped exact client and matter matches

On November 22, 2026, the shortlist changed twice. A similarity-only rule was tested first and lost referenced matters; the selected rule now guarantees **capped slots for exact matches** and fills the rest by similarity:

1. a reviewer-accepted matter;
2. up to **3** matters whose matter ID appears in the narrative as a whole word;
3. up to **7** matters whose client name appears;
4. the highest catalog or confirmed-example similarity.

Each group is ordered by similarity. In these runs, exact matches also joined the retrieved pool, so a named client's matters could enter even when they were outside the 50 nearest catalog neighbors. The current code no longer does this; see [Pool without a full-catalog name search](#pool-without-a-full-catalog-name-search). The caps differ from the November 17 rule, which put every exact match ahead of all similarity-ranked matters. The [worked example](../CLASSIFICATION_EXAMPLE.md) traces one entry through the pipeline.

Two tests show why exact matching is needed.

### Evidence 1: matter IDs are invisible to embeddings

The similarity-only rule was run on the full corpus (`synthetic-v3`, below). **21 of 96** narratives citing a matter ID lost the correct matter from the shortlist. Catalog text excludes matter IDs, and every client has an identical matter of each type, so similarity cannot tell `PT-03-03` from the other privacy matters.

**Example (`corpus-0139`):** “Updated contractual obligations of the data processor; file PT-03-03.”

| Shortlist rule | Delta Foods / Privacy compliance (`PT-03-03`) | Ten options sent to Jev | Jev result |
| --- | --- | --- | --- |
| Similarity only | Similarity rank **50**; not sent | Privacy compliance for 10 other clients | Insufficient evidence |
| Capped exact matches | Matter-ID slot | The referenced matter plus nine similar matters | Correct matter ranked first |

Under the similarity-only rule, Jev chose insufficient evidence in all 21 cases, so each entry needed manual review. On the 240 earlier held-out answerable cases, Jev top-one accuracy fell from **95.0% to 86.3%**, entirely from these cases. The capped rule restores all 21.

### Evidence 2: client names stop being enough for similarity at scale

The 108-matter corpus cannot show this, because a named client's nine matters always dominate similarity there. [`classifier.shortlist_scale`](../classifier/shortlist_scale.py) builds larger catalogs of fictional clients with overlapping names (“Lark Analytics”, “Lark Holdings”, “Lark Partners”). Every client has the same nine matters, so only the client name identifies the right one. For each catalog size, 60 clients × 8 held-out narrative templates = **480 queries** name the client exactly. The test uses real local Qwen embeddings and exact cosine search; Jev is not called.

Expected matter among the ten options sent to Jev:

| Clients (matters) | Similarity only | Expected matter in 50-matter pool | Capped exact matches | November 17 exact-first |
| ---: | ---: | ---: | ---: | ---: |
| 12 (108) | 95/96 | 96/96 | **96/96** | 96/96 |
| 100 (900) | 475/480 | 479/480 | **480/480** | 480/480 |
| 300 (2,700) | 467/480 | 477/480 | **480/480** | 480/480 |
| 1,000 (9,000) | 457/480 (95.2%) | 473/480 | **480/480** | 480/480 |

Similarity-only coverage falls as the catalog grows. At 1,000 clients, 7 expected matters are outside even the 50-neighbor pool; exact client matching brings them in.

**Example at 1,000 clients:** “Amended obligations owed to the commercial landlord for Lark Analytics.” The expected matter is Lark Analytics / Office lease.

| Similarity rank | Matter | Cosine |
| ---: | --- | ---: |
| 1 | Lark Analytics / Supply contract dispute | 0.5883 |
| 2 | Lark Holdings / Office lease | 0.5836 |
| 3 | Lark Partners / Office lease | 0.5632 |
| 4–9 | Office leases and other matters of other “Lark” clients | 0.5477–0.5254 |
| 10 | Lark Analytics / Series B financing | 0.5175 |
| **421** | **Lark Analytics / Office lease** ✔ | **0.4342** |

The embedding captured “Lark” and “landlord” but did not keep the full client name together with the right matter. The expected office lease is only the **sixth** most similar of Lark Analytics' own nine matters. The client-name slots send Lark Analytics' seven most similar matters, including the office lease. The remaining three similarity slots go to other “Lark” clients' office leases.

**Why up to seven client slots.** In an exploratory run on the same 1,000-client catalog, reserving fewer slots for the named client did not fix the misses. With 1, 3, 5, 6, or 7+ reserved slots, coverage was 456, 458, 469, 475, and 480 of 480 respectively. The expected matter ranked as low as seventh within its own client. Seven slots cover that while still leaving three for similarity, for counterparties and other clients.

**Limits of this evidence.**
- Identical descriptions across clients are a worst case; real catalogs vary more.
- Exact search was used; HNSW can only lose further neighbors.
- One narrative template caused 18 of the 23 similarity-only misses.
- Misspelled client names get no help from exact matching. At 1,000 clients, every rule covered 442 of 480 misspelled queries. Those would need fuzzy or alias matching.
- A client with more than seven relevant matters can still lose the expected one; that is untested.

### Results with the capped rule

The corpus (`synthetic-v3`) adds **36 held-out cases** to the 532 earlier blind cases. Every client has the same nine matter names and descriptions, so only the client reference identifies the matter.

| New category | Example | Cases |
| --- | --- | ---: |
| Misspelled client | “…for Alder Robtoics.” | 12 |
| Shortened client | “…for Alder.” | 12 |
| Counterparty also named | “…for Alder Robotics, responding to comments from Birch Energy.” | 12 |

Model, prompt (`client-matter-choice-v1`), gates (0.7 probability, 0.1 margin, 0.7 sufficiency), and baseline weights (`random-42-v3`) are unchanged.

| Held-out measure (376 cases, 276 answerable) | Catalog cosine only | Weighted, same shortlist | Jev |
| --- | ---: | ---: | ---: |
| Top-one | 54.7% (151/276) | 87.0% (240/276) | **93.8% (259/276)** |
| Top-five | 77.9% (215/276) | 98.9% (273/276) | **100% (276/276)** |
| Correct matter in shortlist | 100% | 100% | 100% |
| Ambiguous/unsupported abstention, 100 cases | — | 100% | 99% |
| Coverage without low confidence, 376 cases | — | 0% | 66.2% (249/376) |
| Incorrect suggestions clearing review gates | — | 0 | 2 |

**The three shortlist rules on the 240 earlier held-out answerable cases:**

| Rule | Correct matter in shortlist | Weighted top-one | Jev top-one |
| --- | ---: | ---: | ---: |
| November 17 exact-first | 100% | 87.9% (211/240) | 95.0% (228/240) |
| Similarity only | 91.3% (219/240) | 79.6% (191/240) | 86.3% (207/240) |
| **Capped exact matches** | **100%** | **87.9% (211/240)** | **95.0% (228/240)** |

The capped rule matches the earlier exact-first accuracy on this corpus. It changes the ten options in 286 of the 532 earlier cases, usually by swapping the named client's two least similar matters for similarity-ranked ones. No Jev top-one answer changed; five explicit-client cases changed only whether they cleared the review gates.

**New same-matter-name cases:** Jev ranked the correct **client** first in all 36 cases and the correct matter first in 31; the weighted baseline ranked 29 correctly.
- All five Jev misses chose the same client's Series B matter for “Negotiated protections for incoming shareholders…”. That narrative does not name the round, the same fixture ambiguity described under `corpus-0003` below.
- One of these (`corpus-0592`, a counterparty case) cleared the gates with probability 0.88.
- The other incorrect suggestion that cleared the gates is the unsupported `corpus-0494` case, unchanged from November 17.

| Category | Weighted top-one | Jev top-one | Jev cleared gates |
| --- | ---: | ---: | ---: |
| Misspelled client | 11/12 | 10/12 | 10/12 |
| Shortened client | 10/12 | 11/12 | 11/12 |
| Counterparty also named | 8/12 | 10/12 | 7/12 |

In the counterparty cases, both named clients' 18 matters compete for the seven client slots. The counterparty took 1–4 of those slots in 11 of 12 cases, but the correct matter reached the shortlist in all 12.

**Timing.** The Jev comparison ran alongside the scale benchmark on the same Ollama and Postgres instances, so its timings are not reported as performance measurements. 148 of 568 blind requests were new; the rest reused cached responses from earlier runs.

Raw results, all under `.pointtwo/python/`:
- capped rule: `jev-comparison-v4.json` (with `.requests.json`) and `evaluation-v5.json`;
- similarity-only rule: `jev-comparison-v3.json` and `evaluation-v4.json`;
- scale test: `shortlist-scale.json`.

## Jev versus similarity and weighted scoring (November 17 shortlist)

This section records the earlier shortlist, which put every exact client-name and matter-ID match ahead of similarity-ranked matters. On November 17, 2026, the paired synthetic evaluation improved held-out top-one ranking from **87.9% with the weighted baseline to 95.0% with Jev**. Jev cleared the review gates for **65.0%** of held-out entries, including **one unsupported assignment**. These results support further evaluation of text-based decisions after retrieval; they do not justify automatic assignment or establish production accuracy. The application now supports the same Jev decision protocol after explicit opt-in, with the weighted method retained as a local baseline.

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

## Setup

- Local Apple M1 development machine, 8 GB unified memory; Python, Postgres 17 and pgvector 0.8.7. Other development processes were running.
- Qwen3-Embedding-0.6B, 1,024 dimensions. No model training; the weighted baseline uses fixed weights.
- 108 fictional catalog matters and 556 synthetic time entries: 532 blind cases, plus 12 feedback source/probe pairs. Three pairs were exercised separately.
- The blind split contains 192 development cases and 340 held-out cases. Held-out explicit-client narratives use different templates. This is a synthetic template split, not a chronological or unseen-client production test.
- Expected labels are used only by evaluation. The classifier receives the prompt fields; imported assignments are deliberately unreliable. Quality evaluation uses an isolated workspace without personal review history.

## Performance and ANN scale

A benchmark reused **50,100 prepared synthetic catalog vectors**. Building the workspace HNSW index took **94.9 seconds**; the index occupied **391 MB** (excluding vector tables, cache, and model storage). Six cached queries were compared with exhaustive exact search.

| HNSW search effort | Mean recall@30 |
| --- | ---: |
| `ef_search=200` | 93.3% |
| `ef_search=400` (selected) | 95.0% |
| `ef_search=1000` | 97.2% |

At 400, individual recall ranged from **90% to 100%**, with **276.3–473.5 ms** measured query times. Cache/order effects make these timings unsuitable for comparing effort settings as a controlled speed benchmark. HNSW use was verified in query plans. Neighbor recall measures agreement with exact neighbors, not correct-matter assignment.

One complete 50k application query took **959 ms overall**, including **700 ms retrieval/scoring**. ANN omitted its correct Acme candidate, which was recovered by the full-catalog name search that has since been removed. That sample **missed the proposed 250 ms retrieval/ranking target**. It is one query, not a p95 measurement.

## Limits and reproduction

These are fictional, template-generated cases. The 108-matter accuracy study and 50,100-vector recall study are separate; no 50k classification-accuracy result is claimed. There was no controlled OS cold-cache reset, concurrency/load study, representative production feedback training, or complete process-memory measurement. Weekly training and calibrated release gates remain proposed work.

The scale measurement is in `.pointtwo/python/scale.json`; private local artifacts are ignored by Git.

```sh
# With DATABASE_URL configured privately:
.pointtwo/classifier-venv/bin/python -m classifier.evaluate
.pointtwo/classifier-venv/bin/python -m classifier.benchmark --workspace local-review
.pointtwo/classifier-venv/bin/python -m pytest tests
npm --prefix web run build
```

The benchmark command measures the selected workspace and reports its actual size. Repeating the historical 50,100-vector measurement requires an equivalently prepared catalog; the machine-local scale fixture is not included in a fresh clone. The retained `scale.json` records the original build and effort comparison; the portable benchmark writes `benchmark.json`. See [setup](../README.md), [demonstration](DEMO.md), and [design](../CLASSIFICATION_DESIGN.md).
