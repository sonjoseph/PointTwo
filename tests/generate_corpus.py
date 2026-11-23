"""Generate fictional, labeled cases; ambiguity is a set, never an arbitrary label."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = [
    "Alder Robotics",
    "Birch Energy",
    "Cobalt Analytics",
    "Delta Foods",
    "Elm Medical",
    "Fjord Logistics",
    "Granite Solar",
    "Harbor Systems",
    "Indigo Materials",
    "Juniper Mobility",
    "Kestrel Devices",
    "Linden Networks",
]
# Each row has catalog vocabulary plus distinct phrasing for development / held-out cases.
WORK = [
    (
        "financing",
        "Series A financing",
        "Corporate",
        "Transactional",
        "Equity investment, investor rights, capitalization, subscription agreements and board approval.",
        [
            "Drafted investor rights provisions",
            "Reviewed the capitalization schedule",
            "Negotiated protections for incoming shareholders",
            "Prepared board approval of the equity issuance",
        ],
    ),
    (
        "dispute",
        "Supply contract dispute",
        "Litigation",
        "Dispute",
        "Breach of supply contract, document discovery, damages and settlement.",
        [
            "Reviewed discovery in the supply contract lawsuit",
            "Drafted the breach-of-contract complaint",
            "Assessed damages from missed deliveries",
            "Prepared a settlement position for the vendor litigation",
        ],
    ),
    (
        "employment",
        "Employment advice",
        "Employment",
        "Advisory",
        "Employee agreements, compensation, contractor classification and workplace policies.",
        [
            "Reviewed employee compensation agreements",
            "Updated contractor classification guidance",
            "Advised on onboarding and worker benefits",
            "Revised termination provisions in staff contracts",
        ],
    ),
    (
        "privacy",
        "Privacy compliance",
        "Privacy",
        "Advisory",
        "Personal information processing, data protection, retention and processor agreements.",
        [
            "Revised the data processing agreement",
            "Reviewed personal data retention rules",
            "Assessed safeguards for customer information",
            "Updated contractual obligations of the data processor",
        ],
    ),
    (
        "tax",
        "Corporate tax advice",
        "Tax",
        "Advisory",
        "Corporate income tax, allowable deductions, filing obligations and tax planning.",
        [
            "Reviewed corporate income tax deductions",
            "Prepared advice on tax filing obligations",
            "Analyzed the tax treatment of business expenses",
            "Checked allowable offsets in the corporate return",
        ],
    ),
    (
        "property",
        "Office lease",
        "Real Estate",
        "Transactional",
        "Office tenancy, rent review, landlord covenants and commercial lease negotiation.",
        [
            "Negotiated office lease covenants",
            "Reviewed the rent escalation schedule",
            "Amended obligations owed to the commercial landlord",
            "Advised on break rights for the premises tenancy",
        ],
    ),
    (
        "patent",
        "Patent prosecution",
        "Intellectual Property",
        "Transactional",
        "Patent applications, invention claims, examiner responses and prior art.",
        [
            "Drafted a patent examiner response",
            "Reviewed prior art for the invention claims",
            "Amended claim language in the patent application",
            "Addressed objections from the patent office",
        ],
    ),
    (
        "loan",
        "Credit facility",
        "Finance",
        "Transactional",
        "Secured borrowing, credit covenants, lender security and loan agreements.",
        [
            "Revised financial covenants in the credit facility",
            "Reviewed lender security documents",
            "Negotiated borrowing conditions with the bank",
            "Prepared amendments to the secured loan agreement",
        ],
    ),
]


def build():
    catalog = []
    cases = []
    pairs = []

    def add(
        narrative,
        category,
        expected=None,
        acceptable=None,
        split="heldout",
        user="corpus-reviewer",
        group=None,
    ):
        ident = f"corpus-{len(cases) + 1:04d}"
        record = {
            "UserId": user,
            "EntryId": ident,
            "Narrative": narrative,
            "Hours": 0.5,
            "ClientName": "Incorrect imported client",
            "MatterName": "Incorrect imported matter",
            "EntryDate": "2026-11-16" if split == "development" else "2026-11-17",
        }
        cases.append(
            {
                "entry": record,
                "category": category,
                "split": split,
                "group": group or ident,
                "expected": expected,
                "acceptable": acceptable or [],
                "should_abstain": expected is None,
            }
        )
        return record

    for i, client in enumerate(CLIENTS):
        cid = f"fictional-{i:02d}"
        for j, (key, name, area, kind, description, phrases) in enumerate(WORK):
            mid = f"PT-{i:02d}-{j:02d}"
            catalog.append(
                {
                    "ClientId": cid,
                    "ClientName": client,
                    "MatterId": mid,
                    "MatterName": name,
                    "MatterDescription": description,
                    "PracticeArea": area,
                    "MatterType": kind,
                    "Status": "Closed" if key == "property" and i % 3 == 0 else "Open",
                    "RelatedKeywords": [key],
                    "InvolvedTimekeepers": [],
                }
            )
            expected = [cid, mid]
            for k, phrase in enumerate(phrases):
                category = "explicit_client" if k < 3 else "matter_reference"
                narrative = (
                    f"{phrase} for {client}." if k < 3 else f"{phrase}; file {mid}."
                )
                add(
                    narrative,
                    category,
                    expected,
                    split="development" if k < 2 else "heldout",
                    group=f"{key}-template-{k}",
                )
        # Same client and practice area, but a separate financing matter.
        catalog.append(
            {
                "ClientId": cid,
                "ClientName": client,
                "MatterId": f"PT-{i:02d}-08",
                "MatterName": "Series B financing",
                "MatterDescription": "Later Series B investment round, preferred share rights, new investor negotiations and equity financing.",
                "PracticeArea": "Corporate",
                "MatterType": "Transactional",
                "Status": "Open",
                "RelatedKeywords": ["Series B", "preferred shares"],
                "InvolvedTimekeepers": [],
            }
        )
        add(
            f"Reviewed financing terms for {client}.",
            "same_client_ambiguous",
            acceptable=[[cid, f"PT-{i:02d}-00"], [cid, f"PT-{i:02d}-08"]],
        )
        add(
            f"Reviewed Series B preferred share terms for {client}.",
            "same_client_specific",
            [cid, f"PT-{i:02d}-08"],
        )
        add(
            f"Prepared advice for {client}.",
            "vague",
            acceptable=[[cid, f"PT-{i:02d}-{j:02d}"] for j in range(9)],
        )
        other = CLIENTS[(i + 1) % len(CLIENTS)]
        add(
            f"Reviewed the privacy agreement for {client} and {other}.",
            "multiple_clients",
            acceptable=[
                [cid, f"PT-{i:02d}-03"],
                [f"fictional-{(i + 1) % 12:02d}", f"PT-{(i + 1) % 12:02d}-03"],
            ],
        )
        add(
            f"Revisé las obligaciones de protección de datos personales para {client}.",
            "spanish",
            [cid, f"PT-{i:02d}-03"],
        )
        add(
            f"Examiné les déductions fiscales de {client}.",
            "french",
            [cid, f"PT-{i:02d}-04"],
        )
        add(
            f"Reveiwed patnt examner objections for {client}.",
            "typo",
            [cid, f"PT-{i:02d}-06"],
        )
        # Feedback probes are not part of the blind accuracy split.
        source = add(
            f"Prepared the {i + 1}th founders memo on option vesting and equity grants.",
            "feedback_source",
            [cid, f"PT-{i:02d}-00"],
            split="feedback",
            user=f"feedback-{i}",
        )
        related = add(
            f"Updated memo {i + 1} about vesting schedules and founders share options.",
            "feedback_probe",
            None,
            split="feedback",
            user=f"feedback-{i}",
        )
        pairs.append(
            {"source": source, "probe": related, "assignment": [cid, f"PT-{i:02d}-00"]}
        )
    for j, (key, name, area, kind, desc, phrases) in enumerate(WORK):
        for k, phrase in enumerate(phrases):
            add(
                phrase + ".",
                "cross_client_ambiguous",
                acceptable=[
                    [f"fictional-{i:02d}", f"PT-{i:02d}-{j:02d}"] for i in range(12)
                ],
                group=f"{key}-anonymous-{k}",
            )
        for k in range(3):
            add(
                f"{phrases[k]} for Unknown Orchard {j}-{k} LLC.",
                "unknown_client",
                group=f"unknown-{key}-{k}",
            )
    for phrase in [
        "Watered office plants",
        "Booked a personal holiday",
        "Organized lunch delivery",
        "Checked weekend weather",
        "Ordered kitchen supplies",
        "Sorted family photographs",
        "Repaired my bicycle",
        "Selected music for a party",
    ]:
        add(phrase + ".", "out_of_domain")
    # Same matter name at every client: only the client reference identifies the
    # matter. These variants break exact client-name matching but stay answerable.
    # Appended last so earlier case IDs remain stable across corpus versions.
    for i, client in enumerate(CLIENTS):
        cid = f"fictional-{i:02d}"
        other = CLIENTS[(i + 1) % len(CLIENTS)]
        first, *rest = client.split()
        last = rest[-1]
        misspelled = " ".join(
            [first, *rest[:-1], last[:3] + last[4] + last[3] + last[5:]]
        )
        for category, j, narrative in (
            ("client_misspelled", i % 8, f"{WORK[i % 8][5][2]} for {misspelled}."),
            (
                "client_short_name",
                (i + 3) % 8,
                f"{WORK[(i + 3) % 8][5][2]} for {first}.",
            ),
            (
                "counterparty_named",
                (i + 5) % 8,
                f"{WORK[(i + 5) % 8][5][2]} for {client}, responding to comments from {other}.",
            ),
        ):
            add(
                narrative,
                category,
                [cid, f"PT-{i:02d}-{j:02d}"],
                group=f"same-matter-name-{category}-{j}",
            )
    return {
        "version": "synthetic-v3",
        "catalog": catalog,
        "cases": cases,
        "feedback_pairs": pairs,
        "notes": [
            "Fictional, template-generated cases; not a claim of real legal-domain accuracy.",
            "Ambiguous cases have multiple plausible labels or no supported label; arbitrary top-one choices are not counted as correct.",
            "Development and held-out explicit-client templates differ; no training is performed on this corpus by default.",
            "Feedback examples are separate from blind accuracy evaluation; imported assignments are deliberately wrong.",
            "Every client shares the same matter names and descriptions; misspelled, shortened, and counterparty client references test same-name disambiguation without exact client-name matching.",
            "The conservative demo review policy is uncalibrated; closed matters intentionally require review.",
        ],
    }


if __name__ == "__main__":
    data = build()
    path = ROOT / "fixtures/classification-corpus.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    from collections import Counter

    print(
        json.dumps(
            {
                "catalog": len(data["catalog"]),
                "cases": len(data["cases"]),
                "categories": dict(Counter(c["category"] for c in data["cases"])),
            },
            indent=2,
        )
    )
