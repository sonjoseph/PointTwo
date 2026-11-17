# PointTwo demonstration

Start the application using the [README](../README.md#run-locally), then open `http://127.0.0.1:8794` or run `npm --prefix desktop start`. With the same private database/workspace configuration, run `python -m classifier.seed` to import the fictional demo data. Existing IDs and review decisions are preserved.

These screenshots were captured from the running application with synthetic records and existing review history. Scores and review states can differ in a newly seeded workspace. Screenshots are cropped or resized for readability.

## Review assignments

Search Time entries for `demo-`. Open `demo-001`: its imported assignment is Northstar Labs / Patent dispute, while its narrative describes Acme's financing work. Compare the imported assignment, suggestions, and approved assignment separately. In this capture, the entry already has an accepted decision.

![Time-entry workspace filtered to demo records, showing imported assignments and review states](images/review-workspace.png)

## Handle ambiguity

Open `demo-004`: “Reviewed a term sheet for an equity financing round.” Several clients have financing matters, but the narrative names no client. The application shows alternatives and a **Low confidence** warning; it leaves the entry unassigned until review. Each suggestion offers **Accept** and **Reject**, and the reviewer can search the full catalog for a different matter.

![Low-confidence review of demo-004 with Acme and Northstar financing alternatives](images/low-confidence-review.png)

## Inspect a score

Choose **Open in score debugger** from entry details, **Open debugger** on a row, or search an exact EntryId in the debugger. Select a candidate to inspect its feature values, weights, contributions, catalog text, and retrieved confirmed narratives.

For the captured example, use timekeeper `joseph` and analyze this exploratory probe:

> Revised investor rights provisions and financing documents for Acme Ventures.

With `demo-001` already confirmed as Acme / Series A financing, the probe retrieves that distinct narrative as supporting evidence. The resulting Acme score is **0.6504**: approximately 0.1338 catalog similarity contribution + 0.0808 example similarity contribution + 0.1817 client-name contribution + 0.2542 timekeeper-context contribution. There is no example-availability bonus. These are ranking scores, not probabilities.

![Weighted score contributions for the Acme financing probe and alternative matters](images/score-contributions.png)

## Explore the embedding neighborhood

Click a point in the graph to inspect its full-dimensional cosine similarity and distance. The diamond is the probe, squares are catalog descriptions, and circles are confirmed narratives. Here, the selected confirmed Acme narrative has cosine similarity **0.7066** to the probe.

![Embedding neighborhood showing the probe, catalog matters, and a selected confirmed Acme narrative](images/embedding-neighborhood.png)

The map is a 2D PCA approximation; the displayed cosine values use all 1,024 dimensions. Lines connect confirmed examples to their matters, not HNSW neighbors. The graph keeps baseline axes fixed when comparing feedback. This capture shows existing feedback evidence, not a measured before/after experiment.

## Try a feedback experiment

1. Enter a narrative and timekeeper, then choose **Analyze & save baseline**. The saved run is immutable.
2. In **Correct another entry**, select a different source narrative for the same timekeeper and its correct matter. Choose **Apply correction & reevaluate probe**. This saves a real, reversible review decision.
3. Compare the baseline and feedback graph, score deltas, and example-similarity contributions. The probe's assignment and scoring weights stay unchanged; an identical normalized narrative can separately trigger an explicit decision override.
4. Choose **Restore previous assignment & reevaluate** to undo the experiment, or retain the correction. Change the probe's timekeeper to check that confirmed history is scoped separately.

After feedback, **Refresh open suggestions** queues outdated unapproved entries; accepted assignments stay fixed. **Rebuild & refresh scores** also rebuilds catalog preparation while reusing unchanged embeddings.

## Evaluate the result

Explore the synthetic corpus's same-client matters, missing client names, conflicting references, unknown clients, and unrelated work. A plausible nearest neighbor does not establish a unique assignment.

The [evaluation report](EVALUATION_REPORT.md) records measured accuracy, feedback effects, performance, and limitations in an isolated workspace. Compare top-one/top-five accuracy with abstention and ready-for-review coverage: good ranking does not establish calibrated confidence. Expected labels are evaluation data and never classifier inputs.
