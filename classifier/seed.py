"""Import the bundled fictional demo catalog and entries without approving labels."""

import argparse
import json
import os
from pathlib import Path

from . import store
from .database import connect
from .models import Matter, TimeEntry

ROOT = Path(__file__).resolve().parents[1]


def seed(workspace, *, small=False):
    catalog = json.loads((ROOT / "fixtures/catalog.json").read_text())
    entries = json.loads((ROOT / "fixtures/time-entries.json").read_text())
    if not small:
        corpus = json.loads((ROOT / "fixtures/classification-corpus.json").read_text())
        catalog += corpus["catalog"]
        # Ground-truth answers are evaluation metadata, never imported as decisions.
        entries += [case["entry"] for case in corpus["cases"]]
    store.migrate()
    with connect() as conn:
        store.workspace(conn, workspace)
        existing_catalog = {
            (r["client_id"], r["matter_id"])
            for r in conn.execute(
                "SELECT client_id,matter_id FROM matters WHERE workspace_id=%s",
                (workspace,),
            )
        }
        existing_entries = {
            r["external_id"]
            for r in conn.execute(
                "SELECT external_id FROM pt_entries WHERE workspace_id=%s", (workspace,)
            )
        }
    new_catalog = [
        Matter.model_validate(m)
        for m in catalog
        if (m["ClientId"], m["MatterId"]) not in existing_catalog
    ]
    new_entries = [
        TimeEntry.model_validate(e)
        for e in entries
        if e["EntryId"] not in existing_entries
    ]
    # Skip existing fixture IDs: reruns preserve edited sources and review decisions.
    if new_catalog:
        store.import_catalog(workspace, new_catalog)
    if new_entries:
        store.import_entries(workspace, new_entries)
    return {
        "workspace": workspace,
        "catalog_added": len(new_catalog),
        "entries_added": len(new_entries),
        "catalog_skipped": len(catalog) - len(new_catalog),
        "entries_skipped": len(entries) - len(new_entries),
        "message": "Start the application with this workspace to prepare vectors and classify queued entries.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", default=os.getenv("POINTTWO_WORKSPACE", "local-review")
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Only the six-matter / six-entry starter dataset",
    )
    args = parser.parse_args()
    print(json.dumps(seed(args.workspace, small=args.small), indent=2))


if __name__ == "__main__":
    main()
