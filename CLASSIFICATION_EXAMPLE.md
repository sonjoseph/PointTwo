# Classification example: one entry end to end

This walkthrough follows one time entry through the Jev pipeline: embedding, similarity search, shortlisting, and the Jev decision. All values come from the November 22, 2026 synthetic run (`.pointtwo/python/jev-comparison-v4.json`), case `corpus-0559`. See the [classification design](CLASSIFICATION_DESIGN.md) for the rationale and the [evaluation report](docs/EVALUATION_REPORT.md#similarity-only-shortlist) for aggregate results.

## The entry

```json
{
  "UserId": "corpus-reviewer",
  "EntryId": "corpus-0559",
  "Narrative": "Amended obligations owed to the commercial landlord for Alder Robotics, responding to comments from Birch Energy.",
  "Hours": 0.5,
  "ClientName": "Incorrect imported client",
  "MatterName": "Incorrect imported matter",
  "EntryDate": "2026-11-17"
}
```

The expected answer is **Alder Robotics / Office lease** (`PT-00-05`). This case is hard in three ways:

- All 12 synthetic clients have a matter named "Office lease" with the same description. Only the client reference distinguishes them.
- The narrative names a second client, Birch Energy, as the other party. Birch also has an "Office lease" matter.
- Alder's office lease is **Closed**, which always requires review.

The imported `ClientName`/`MatterName` are deliberately wrong. They are kept for display and are never used as evidence.

## 1. Embeddings

Qwen3-Embedding-0.6B runs locally through Ollama and produces 1,024-dimensional vectors. Three kinds of text are embedded:

| Text | When | Embedded input | Not embedded |
| --- | --- | --- | --- |
| Catalog matter | Catalog preparation ([`Matter.text`](classifier/models.py)) | `ClientName MatterName MatterDescription RelatedKeywords PracticeArea MatterType`, joined with spaces | `ClientId`, `MatterId`, `Status`, `InvolvedTimekeepers` |
| Time-entry query | Each classification ([`engine.py`](classifier/engine.py)) | Instruction prefix + `Narrative` | `UserId`, `EntryId`, `Hours`, `EntryDate`, imported assignment |
| Confirmed example | When a reviewer accepts an assignment ([`store.decide`](classifier/store.py)) | `Narrative` only, no prefix | Everything else; the vector is stored with the matter and timekeeper |

The relevant catalog record is embedded as this string:

```text
Alder Robotics Office lease Office tenancy, rent review, landlord covenants and commercial lease negotiation. property Real Estate Transactional
```

The query is embedded as:

```text
Instruct: Retrieve the client and matter matching this work description.
Query: Amended obligations owed to the commercial landlord for Alder Robotics, responding to comments from Birch Energy.
```

Vectors are cached in `pt_embedding_cache` under `sha256(model identity + exact text)`. Unchanged catalog records and repeated narratives are not embedded again.

### Why only the query has an instruction

Qwen3-Embedding is **instruction-aware** and built for **asymmetric retrieval**. In asymmetric retrieval, the query and the stored documents are different kinds of text. Here, a short past-tense work narrative is matched against a longer description of a matter's scope. The task instruction tells the model what the query is looking for: the matter this work belongs to, not text that merely looks similar. Documents are embedded as plain text. They are prepared once and reused even if the query instruction changes. Qwen's model card reports a 1–5% retrieval drop without an instruction; PointTwo has not measured that effect on its own corpus.

Because the catalog text includes the client name, the embedding captures client identity as well as the type of work. Matter IDs are excluded, so the embedding does not represent references such as `file PT-03-03`.

## 2. Similarity search

Retrieval runs in one database transaction ([`Engine.classify`](classifier/engine.py)). The values below are from the November 22 run, which also searched the whole catalog for named clients; the current code does not, but for this entry the pool is the same because all named matters are already among the 50 nearest.

1. **Catalog search:** the 50 catalog vectors nearest the query by cosine distance (`ORDER BY embedding <=> query LIMIT 50`). pgvector uses HNSW for large catalogs; this 108-matter catalog uses an exact scan.
2. **Confirmed-example search:** the 100 nearest accepted narratives **from the same `UserId`**. This entry has no history, so this search returns nothing.
3. **Earlier review decisions:** matters this timekeeper accepted or rejected for the same normalized narrative. There are none here.

The union is the **candidate pool**: 50 matters. Each candidate records `catalog_similarity` (exact cosine to its catalog text), `example_similarity` (best cosine among its confirmed examples, or 0), and whether its `ClientName` or `MatterId` appears in the narrative as a whole word, ignoring case. “Alder Robotics” and “Birch Energy” both appear, and all **18** of their matters are among the 50, so all 18 are flagged as named. Names and IDs are not searched for outside the pool.

Top of the pool by catalog similarity:

| Rank | Client / matter | Catalog similarity |
| ---: | --- | ---: |
| 1 | Alder Robotics / Supply contract dispute | 0.5573 |
| 2 | Alder Robotics / Series A financing | 0.5556 |
| 3 | Alder Robotics / Credit facility | 0.5463 |
| 4 | Alder Robotics / Series B financing | 0.5387 |
| **5** | **Alder Robotics / Office lease** ✔ | **0.5072** |
| 6 | Alder Robotics / Privacy compliance | 0.4925 |
| 7 | Alder Robotics / Patent prosecution | 0.4803 |
| 8 | Birch Energy / Office lease | 0.4560 |
| 9 | Alder Robotics / Employment advice | 0.4551 |
| 10 | Alder Robotics / Corporate tax advice | 0.4334 |
| 11 | Birch Energy / Employment advice | 0.4053 |
| 12 | Elm Medical / Office lease | 0.4030 |

The client name dominates the embedding. The top ten are all nine Alder matters plus Birch's office lease, which ranks 8th because Birch is named and the work matches. Other clients' identical office leases start at rank 12. Cosine similarity alone puts the correct matter **5th**: it identifies the client but cannot tell which Alder matter the work belongs to.

## 3. Shortlist

[`jev.shortlist`](classifier/jev.py) removes rejected matters and fills **10** slots in this order. Each group is ordered by `max(catalog_similarity, example_similarity)`, with stable ID tie-breaking.

| Step | Rule | This entry |
| --- | --- | --- |
| 1 | A reviewer-accepted matter | None |
| 2 | Up to **3** matters whose matter ID appears in the narrative | None cited |
| 3 | Up to **7** matters whose client name appears | 18 qualify (Alder and Birch). The 7 most similar are Alder's, similarity ranks 1–7, **including the office lease** |
| 4 | Remaining slots by similarity | Birch / Office lease (0.4560), Alder / Employment advice (0.4551), Alder / Corporate tax advice (0.4334) |

The shortlist does not use the weighted scorer, imported assignments, or expected labels. For this entry it matches the top ten by similarity: nine Alder matters plus Birch's office lease. The caps matter when similarity alone would miss an explicitly referenced matter; see [Alternatives considered](#alternatives-considered).

## 4. Jev request

[`jev.request`](classifier/jev.py) assigns each shortlisted matter an opaque key (`matter_0` … `matter_9`). The order is a stable shuffle derived from `sha256(EntryId, ClientId, MatterId)`, so Jev cannot infer the similarity rank. An `insufficient_evidence` option is added. Each option is described like this:

```json
"matter_0": {
  "client_id": "fictional-00",
  "matter_id": "PT-00-05",
  "client_name": "Alder Robotics",
  "matter_name": "Office lease",
  "catalog_description": "Alder Robotics Office lease Office tenancy, rent review, landlord covenants and commercial lease negotiation. property Real Estate Transactional",
  "status": "Closed",
  "confirmed_narratives_for_this_matter": []
}
```

The request sent to TypeSafe (`jev-1.14.0`) contains:

- `state.narrative`: the original narrative, unchanged;
- `state.candidates`: the ten options plus `insufficient_evidence`;
- `questions.assignment`: a **Choice** question over those options, with fixed instructions. The instructions say to read the work in its transaction context, not to invent client identity, to treat examples as supporting evidence, and to choose insufficient evidence when several assignments remain plausible;
- `questions.sufficient_evidence`: a **Noul** question asking whether the evidence identifies one specific client–matter.

Each option carries up to three confirmed narratives, excluding duplicates of the query; there are none for this entry. Jev never receives `UserId`, `EntryId`, hours, date, the imported assignment, similarity scores, weighted scores, or evaluation labels.

## 5. Jev decision

| Option | Probability |
| --- | ---: |
| **Alder Robotics / Office lease** | **0.92** |
| Birch Energy / Office lease | 0.03 |
| Insufficient evidence | 0.05 |
| Other eight matters | 0.00 |

The sufficiency answer is **0.80**. Jev ranks the correct matter first, ahead of four Alder matters that embedding similarity ranked higher. It also gives the counterparty's identically named matter only 0.03.

[`jev.interpret`](classifier/jev.py) validates the response and applies the review gates:

| Gate | Observed | Minimum | Result |
| --- | ---: | ---: | --- |
| Top matter probability | 0.92 | 0.70 | Pass |
| Margin over the next option | 0.87 | 0.10 | Pass |
| Sufficient evidence | 0.80 | 0.70 | Pass |
| Matter status | Closed | Open | **Needs review** |

The result is saved with **low confidence: "Closed matter needs explicit review"**. Up to five ranked suggestions are kept for the reviewer. For comparison, the weighted baseline ranked Alder / Supply contract dispute first on the same shortlist.

If the reviewer accepts Alder / Office lease, the narrative is embedded without the instruction and stored as a confirmed example for `corpus-reviewer`. That timekeeper's future similar entries can then retrieve it, and Jev receives it under `confirmed_narratives_for_this_matter`.

## Alternatives considered

The selected rule combines similarity with **capped exact-match slots, applied within the similarity pool**. Each alternative below failed one of the measured cases. Details are in the [evaluation report](docs/EVALUATION_REPORT.md#shortlist-with-capped-exact-client-and-matter-matches).

### Similarity only (no exact matching)

Ranking purely by embedding similarity is simple and handles misspelled and shortened client names. It misses two kinds of explicit reference.

**Matter IDs.** Catalog text omits matter IDs, so embeddings cannot connect “file PT-03-03” to its matter. For `corpus-0139`, “Updated contractual obligations of the data processor; file PT-03-03”:
- similarity ranked Delta Foods / Privacy compliance **50th**, behind ten other clients' identical privacy matters;
- Jev received only those ten and chose insufficient evidence.

Across the corpus, **21 of 96** matter-ID references missed the shortlist, and held-out Jev top-one accuracy fell from 95.0% to 86.3%. With the capped rule, the referenced matter takes an ID slot and Jev ranks it first. The current code, which searches names and IDs only within the 50-matter pool, misses this case: the referenced matter falls just outside the 50 nearest, and Jev chooses insufficient evidence (0.97). It is the only held-out case the change affected.

**Client names in large catalogs.** With 12 clients, the named client's matters dominate similarity, as in this example. With 1,000 fictional clients sharing matter names and overlapping client names, similarity omitted the expected matter for **23 of 480** queries that named the client exactly. For “Amended obligations owed to the commercial landlord for Lark Analytics.”:
- Lark Analytics / Office lease ranked **421st** by similarity (0.4342);
- the top ten were mostly office leases of *other* “Lark” clients (Lark Holdings, Lark Partners, Lark Foods);
- the office lease was only the sixth most similar of Lark Analytics' own matters.

When names were also searched across the whole catalog, the client-name slots included it and all 480 queries reached the shortlist. The current code searches names only within the 50-matter pool, so this case is no longer found; 473 of 480 queries reached the shortlist (479 with a 200-matter pool, which still misses this rank-421 case). See [Pool without a full-catalog name search](docs/EVALUATION_REPORT.md#pool-without-a-full-catalog-name-search).

### Exact-first without caps (November 17 rule)

The earlier rule put **every** exact client-name or matter-ID match ahead of all similarity-ranked matters. It covered both failures above on the measured sets, but it is fragile in cases those sets did not stress:

| Edge case | What uncapped exact-first matching does | Capped rule |
| --- | --- | --- |
| **Counterparty named** (this example) | All 18 Alder and Birch matters outrank every unnamed matter | Named clients share at most 7 slots; similarity keeps the rest |
| **Client with more than ten matters** | That client's matters fill every slot, however weakly related | At most 7 client slots |
| **Common-word matter IDs** (`employment`, `series-a` in [`fixtures/catalog.json`](fixtures/catalog.json)) | “employment” in a narrative promotes *every* client's `employment` matter | At most 3 ID slots |
| **Misspelled or shortened client** (“Alder Robtoics”, “Alder”) | No match, so no help | Same; similarity and Jev handle these |

On the synthetic corpus, both rules reach 95.0% Jev top-one on the earlier held-out cases, because no client there has more than nine matters. The capped rule limits the edge cases without losing measured coverage.

### Fewer reserved client slots

Reserving only the named client's top 1–3 matters did not fix the large-catalog misses. At 1,000 clients, coverage was 456–458 of 480 versus 480 with seven slots. The expected matter can rank as low as seventh among its own client's matters (sixth in the Lark Analytics example), so a small reservation keeps the wrong matters of the right client.

### Other options

| Option | Status |
| --- | --- |
| Add `MatterId` to the embedded catalog text | Embeddings capture opaque identifiers poorly, so this would be unreliable. Not adopted. |
| Fuzzy or alias client matching | Would help misspelled names, which no exact rule covers. At 1,000 clients, 38 of 480 misspelled queries missed under every rule. Not yet built. |
| Hybrid lexical (BM25) and vector retrieval | Adds identifier and keyword recall as a ranked signal instead of fixed slots. See [database choice](CLASSIFICATION_DESIGN.md#database-choice). |
| Search names across the whole catalog | Recovers named matters outside the 50 nearest (480/480 instead of at most 473/480 at 9,000 matters), but needed a full table scan per query with a 1,000-row cap. Removed; a 200-matter pool reaches 479/480, and an indexed name lookup would cover the rest. |
| Send the full pool to Jev instead of ten | Larger requests and more distractors; the pool size is a search budget, not a decision set. |
