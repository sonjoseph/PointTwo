"""Shortlist coverage as the number of same-name matters grows. Offline; no Jev calls.

Builds synthetic catalogs in which every fictional client has the fixture's nine matter
templates, so only the client name distinguishes them. Client names share words
("Alder Robotics", "Alder Energy", "Birch Robotics"). Queries reuse the fixture's
held-out explicit-client phrasing. Each shortlist rule is measured on whether the
expected matter reaches the ten options sent to Jev. Search is exact cosine; HNSW
can only lose further neighbors. Requires DATABASE_URL for the embedding cache.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .embeddings import INSTRUCTION, Embedder
from .jev import shortlist
from .models import Matter
from .scoring import contains

ROOT = Path(__file__).resolve().parents[1]
FIRST = [
    "Alder",
    "Birch",
    "Cobalt",
    "Delta",
    "Elm",
    "Fjord",
    "Granite",
    "Harbor",
    "Indigo",
    "Juniper",
    "Kestrel",
    "Linden",
    "Maple",
    "Nimbus",
    "Onyx",
    "Pine",
    "Quartz",
    "Raven",
    "Sierra",
    "Tidal",
    "Umber",
    "Vale",
    "Willow",
    "Xenon",
    "Yarrow",
    "Zephyr",
    "Amber",
    "Basalt",
    "Cedar",
    "Dune",
    "Ember",
    "Flint",
    "Glacier",
    "Hazel",
    "Iris",
    "Jade",
    "Krypton",
    "Lark",
    "Mesa",
    "Nova",
]
SECOND = [
    "Robotics",
    "Energy",
    "Analytics",
    "Foods",
    "Medical",
    "Logistics",
    "Solar",
    "Systems",
    "Materials",
    "Mobility",
    "Devices",
    "Networks",
    "Capital",
    "Pharma",
    "Holdings",
    "Labs",
    "Ventures",
    "Retail",
    "Biotech",
    "Aerospace",
    "Media",
    "Studios",
    "Partners",
    "Foundry",
    "Telecom",
]


def client_names():
    names = [f"{a} {b}" for a in FIRST for b in SECOND]
    random.Random(7).shuffle(names)
    return names


def misspell(name):
    first, last = name.split()
    i = 3 if len(last) >= 6 else 1
    return f"{first} {last[:i]}{last[i + 1]}{last[i]}{last[i + 2 :]}"


def templates():
    data = json.loads((ROOT / "fixtures/classification-corpus.json").read_text())
    matters = [m for m in data["catalog"] if m["ClientId"] == "fictional-00"]
    phrases = {}
    for case in data["cases"]:
        if (
            case["category"] == "explicit_client"
            and case["split"] == "heldout"
            and case["expected"][0] == "fictional-00"
        ):
            suffix = " for Alder Robotics."
            phrases[case["expected"][1]] = case["entry"]["Narrative"].removesuffix(
                suffix
            )
    return matters, phrases


POOL_SIZES = (50, 100, 200, 400)
IN_POOL_RULES = tuple(f"capped_in_pool_{k}" for k in POOL_SIZES)


def rules(sims, catalog, text):
    """Return each rule's ten selected catalog indexes.

    `capped_literal` searches the whole catalog for literal matches (the former
    Engine.classify pool). `capped_in_pool_<k>` mirrors the current engine: the
    pool is only the k nearest catalog matters, and literal matches within it get
    the capped shortlist slots.
    """
    order = np.argsort(-sims, kind="stable")
    literal = {
        k
        for k, m in enumerate(catalog)
        if contains(text, m.ClientName) or contains(text, m.MatterId)
    }

    def capped(pool):
        candidates = [
            {
                "CandidateId": str(k),
                "ClientId": catalog[k].ClientId,
                "MatterId": catalog[k].MatterId,
                "decision": None,
                "features": {
                    "catalog_similarity": float(sims[k]),
                    "example_similarity": 0.0,
                    "client_name_match": int(
                        k in literal and contains(text, catalog[k].ClientName)
                    ),
                    "matter_reference_match": int(
                        k in literal and contains(text, catalog[k].MatterId)
                    ),
                },
            }
            for k in pool
        ]
        return [int(c["CandidateId"]) for c in shortlist({"candidates": candidates})]

    literal_first = sorted(literal, key=lambda k: -sims[k]) + [
        k for k in order if k not in literal
    ]
    return order, {
        "similarity_only": order[:10].tolist(),
        "literal_first": literal_first[:10],
        "capped_literal": capped(set(order[:50].tolist()) | literal),
        **{f"capped_in_pool_{k}": capped(set(order[:k].tolist())) for k in POOL_SIZES},
    }


def run(sizes, query_clients, output):
    names = client_names()
    matters, phrases = templates()
    embedder = Embedder()
    report = {"model": embedder.model, "query_clients": query_clients, "sizes": {}}
    for size in sizes:
        catalog = [
            Matter.model_validate(
                {
                    **m,
                    "ClientId": f"client-{c:04d}",
                    "ClientName": names[c],
                    "MatterId": f"S-{c:04d}-{j}",
                    "InvolvedTimekeepers": [],
                }
            )
            for c in range(size)
            for j, m in enumerate(matters)
        ]
        vectors = np.asarray(embedder.many([m.text() for m in catalog]), np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        variants = {}
        for variant, label in (("exact", lambda n: n), ("misspelled", misspell)):
            queries = [
                (c, j, f"{phrases[m['MatterId']]} for {label(names[c])}.")
                for c in range(min(size, query_clients))
                for j, m in enumerate(matters)
                if m["MatterId"] in phrases
            ]
            q = np.asarray(
                embedder.many([INSTRUCTION + text for _, _, text in queries]),
                np.float32,
            )
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            rows = []
            for (c, j, text), sims in zip(queries, q @ vectors.T):
                target = c * len(matters) + j
                order, selected = rules(sims, catalog, text)
                rows.append(
                    {
                        "narrative": text,
                        "similarity_rank": int(np.where(order == target)[0][0]) + 1,
                        **{rule: target in ks for rule, ks in selected.items()},
                    }
                )
            variants[variant] = {
                "queries": len(rows),
                **{
                    f"in_pool_{k}": sum(r["similarity_rank"] <= k for r in rows)
                    for k in POOL_SIZES
                },
                **{
                    rule: sum(r[rule] for r in rows)
                    for rule in ("similarity_only", "literal_first", "capped_literal")
                    + IN_POOL_RULES
                },
                # Every case where some rule omitted the expected matter.
                "disagreements": [
                    r
                    for r in rows
                    if not all(
                        r[rule]
                        for rule in ("similarity_only", "capped_literal")
                        + IN_POOL_RULES
                    )
                ],
            }
            v = variants[variant]
            print(
                f"{size:>5} clients {variant:<10} similarity-only {v['similarity_only']}/{v['queries']}"
                f"  literal-first {v['literal_first']}  capped {v['capped_literal']}"
                + "".join(
                    f"  in-pool@{k} {v[f'capped_in_pool_{k}']}" for k in POOL_SIZES
                ),
                flush=True,
            )
        report["sizes"][size] = {"matters": len(catalog), **variants}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, nargs="+", default=[12, 100, 300, 1000])
    parser.add_argument("--query-clients", type=int, default=60)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".pointtwo/python/shortlist-scale.json"
    )
    args = parser.parse_args()
    if max(args.clients) > len(FIRST) * len(SECOND):
        parser.exit(2, f"At most {len(FIRST) * len(SECOND)} clients are available\n")
    run(args.clients, args.query_clients, args.output)
