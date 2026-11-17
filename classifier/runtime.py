"""Private runtime settings; cloud decisions require an explicit Jev selection."""

import json
import os
from pathlib import Path

from .engine import Engine
from .jev import JevClient
from .jev_engine import JevEngine


def create_engine(embedder, data):
    path = data / "runtime.json"
    settings = json.loads(path.read_text()) if path.exists() else {}
    mode = os.getenv("POINTTWO_DECISION", settings.get("decision_pipeline", "weighted"))
    if mode == "weighted":
        return Engine(embedder)
    if mode != "jev":
        raise ValueError("POINTTWO_DECISION must be jev or weighted")
    key = os.getenv("TYPESAFE_API_KEY")
    keyfile = os.getenv("TYPESAFE_API_KEY_FILE", settings.get("typesafe_api_key_file"))
    if not key and keyfile:
        key = Path(keyfile).expanduser().read_text().strip()
    return JevEngine(embedder, JevClient(key))
