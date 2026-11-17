# Embedding distance lab

[embedding_distances.ipynb](embedding_distances.ipynb) visualizes the fictional catalog and time-entry fixtures using the same query/document text as the Python classifier. It includes a cosine-distance heatmap, interactive PCA map, nearest-matter and pair comparisons, projection-distortion plot, and optional custom narrative.

The notebook reads compatible cached vectors from Postgres when `DATABASE_URL` is provided. Otherwise, or for cache misses, it uses local Ollama with `qwen3-embedding:0.6b`. It never changes assignments or writes to the application database. Generated vectors and plots stay under ignored `.pointtwo/notebooks/`.

```sh
python3.12 -m venv .pointtwo/notebook-venv
.pointtwo/notebook-venv/bin/python -m pip install -r notebooks/requirements.txt
.pointtwo/notebook-venv/bin/jupyter lab notebooks/embedding_distances.ipynb
```

Choose **Run → Run All Cells**. An optional `DATABASE_URL` must be configured privately before starting Jupyter. Database access is read-only and restricted to hashes of the requested fixture text; personal entry records are not loaded.

Cosine values use all 1,024 dimensions. The 2D PCA view loses information and is not a search index or ranking score. For live feedback comparisons, use the application's **Score debugger**.

The committed notebook has no outputs. If you enter real work as a custom narrative, clear all outputs before sharing it. Model inference accepts only literal loopback endpoints and rejects cloud aliases and redirects.
