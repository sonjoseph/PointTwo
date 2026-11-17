# PointTwo demonstration

Start the Python application as described in [README](../README.md). Open the local browser at `http://127.0.0.1:8794` or use `npm --prefix desktop start`.

1. **Review an assignment.** Open `demo-001`: its imported assignment is Northstar / Patent dispute, while the narrative names Acme's financing work. Imported and accepted assignments remain separate.
2. **Inspect the score.** Choose **Open in score debugger** in entry details or **Open debugger** on its row. You can also search its exact EntryId inside the debugger. Select a candidate to see its feature values, weights, contribution table, catalog text, and supporting confirmed narratives.
3. **Save a related probe.** In Score debugger, enter a narrative and timekeeper, then **Analyze & save baseline**. A run is immutable.
4. **Apply feedback.** In the debugger’s **Correct another entry** panel, select a different source narrative for the same timekeeper, select its matter, and choose **Apply correction & reevaluate probe**. Toggle the embedding graph between baseline and feedback; click a point for exact cosine distance. The graph keeps baseline axes fixed. The table shows score and example-similarity deltas without changing weights. An identical narrative can also trigger an explicit decision override.
5. **Test undo and isolation.** Choose **Restore previous assignment & reevaluate**, or undo the source assignment in Time entries and compare again. Change the probe's timekeeper: their history is scoped separately.
6. **Explore ambiguity.** The seeded synthetic corpus includes same-client financing matters, missing client names, multiple clients in one narrative, unknown clients, and unrelated work. A plausible nearest neighbor is not sufficient evidence of a unique assignment.
7. **Read the report.** Compare top-one/top-five accuracy with abstention and ready-for-review coverage. All cases can require review even when ranking is often correct; the seeded random scorer is not calibrated.

Initialize the synthetic corpus with `python -m classifier.seed`. Existing IDs are skipped. The measured report uses an isolated workspace with no personal review history, and its expected labels are never classifier inputs. See [evaluation report](EVALUATION_REPORT.md) for limitations.

For repeatable initialization, run `python -m classifier.seed` with the same private database/workspace configuration as the app. Existing IDs and review decisions are preserved. After feedback, **Refresh open suggestions** queues outdated unapproved entries; accepted assignments stay fixed.
