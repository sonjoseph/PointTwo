"""Local-only Qwen inference with persistent model/input-specific caching."""

import hashlib
import os
import threading
from urllib.parse import urlparse
import httpx
import numpy as np
from .database import connect

INSTRUCTION = (
    "Instruct: Retrieve the client and matter matching this work description.\nQuery: "
)


class Embedder:
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        if urlparse(self.url).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Local mode requires a loopback Ollama endpoint")
        self.name = os.getenv("LOCAL_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        self._model = None
        self.lock = threading.RLock()

    @property
    def model(self):
        with self.lock:
            if not self._model:
                with httpx.Client(trust_env=False, timeout=15) as client:
                    result = client.get(self.url + "/api/tags")
                    result.raise_for_status()
                record = next(
                    (m for m in result.json()["models"] if m["name"] == self.name), None
                )
                if not record or record.get("remote_host"):
                    raise RuntimeError(
                        "Install the local Qwen embedding model before preparing data"
                    )
                self._model = f"qwen3-0.6b:1024:v1:{self.name}:{record['digest']}"
            return self._model

    def embed(self, text):
        return self.many([text])[0]

    def many(self, texts):
        with self.lock:
            model = self.model
            keys = [hashlib.sha256(f"{model}:{t}".encode()).hexdigest() for t in texts]
            with connect() as conn:
                cached = {
                    r["content_hash"]: r["embedding"]
                    for r in conn.execute(
                        "SELECT content_hash,embedding FROM pt_embedding_cache WHERE model_id=%s AND content_hash=ANY(%s)",
                        (model, keys),
                    )
                }
                missing = list(dict.fromkeys(k for k in keys if k not in cached))
                mapping = dict(zip(keys, texts))
                for offset in range(0, len(missing), 16):
                    batch = missing[offset : offset + 16]
                    with httpx.Client(trust_env=False, timeout=180) as client:
                        response = client.post(
                            self.url + "/api/embed",
                            json={
                                "model": self.name,
                                "input": [mapping[k] for k in batch],
                                "truncate": False,
                                "keep_alive": "10m",
                            },
                        )
                        response.raise_for_status()
                    vectors = response.json()["embeddings"]
                    if len(vectors) != len(batch):
                        raise ValueError("Embedding batch length mismatch")
                    with conn.transaction():
                        for key, vector in zip(batch, vectors):
                            v = np.asarray(vector, dtype=np.float32)
                            if (
                                v.shape != (1024,)
                                or not np.isfinite(v).all()
                                or not np.any(v)
                            ):
                                raise ValueError(
                                    "Embedding must be 1,024 finite, nonzero values"
                                )
                            conn.execute(
                                "INSERT INTO pt_embedding_cache VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                                (model, key, v),
                            )
                            cached[key] = v
                return [cached[k] for k in keys]
