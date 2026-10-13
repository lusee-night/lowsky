"""Read the audited observational evidence index shipped with the repository.

The index is intentionally an inventory, not a likelihood: heterogeneous measurements
require their stated sky, beam, aperture and calibration operators before comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    """Return the checkout root for this source-tree package."""
    return Path(__file__).resolve().parents[2]


def index_path() -> Path:
    return repository_root() / "evidence" / "index.json"


def load_index(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the portable evidence inventory."""
    target = Path(path) if path is not None else index_path()
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported evidence schema: {data.get('schema_version')!r}")
    records = data.get("records")
    sources = data.get("sources")
    if not isinstance(records, list) or not isinstance(sources, list):
        raise ValueError("Evidence index requires records and sources lists")
    record_ids = [item["id"] for item in records]
    source_ids = [item["id"] for item in sources]
    if len(record_ids) != len(set(record_ids)) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Evidence IDs must be unique")
    known_sources = set(source_ids)
    for item in records:
        if not set(item.get("source_ids", ())) <= known_sources:
            raise ValueError(f"Unknown source ID in {item['id']}")
    known_records = set(record_ids)
    for alias in data.get("aliases", ()):
        if alias["canonical_id"] not in known_records:
            raise ValueError(f"Unknown canonical alias target: {alias['canonical_id']}")
    return data


def list_records(
    *, group: str | None = None, comparison_status: str | None = None,
    evidence_kind: str | None = None, index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter records without implying that their values are directly comparable."""
    records = (index if index is not None else load_index())["records"]
    return [
        item for item in records
        if (group is None or item["group"] == group)
        and (comparison_status is None or item["comparison_status"] == comparison_status)
        and (evidence_kind is None or item["evidence_kind"] == evidence_kind)
    ]


def get_record(record_id: str, *, index: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get one canonical record by stable ID."""
    for item in (index if index is not None else load_index())["records"]:
        if item["id"] == record_id:
            return item
    raise KeyError(record_id)


def resolve_origin(record: dict[str, Any], *, root: Path | None = None) -> Any:
    """Read the source JSON/CSV value named by a record's pointer."""
    base = root if root is not None else repository_root()
    origin = record["origin"]
    path = base / origin["path"]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            source: Any = {"csv_rows": list(csv.DictReader(stream))}
    else:
        source = json.loads(path.read_text(encoding="utf-8"))
    for token in origin["pointer"].split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        source = source[int(token)] if isinstance(source, list) else source[token]
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    listing = command.add_parser("list", help="List indexed evidence records")
    listing.add_argument("--group")
    listing.add_argument("--status", dest="comparison_status")
    listing.add_argument("--kind", dest="evidence_kind")
    listing.add_argument("--json", action="store_true", help="Print full JSON records")
    showing = command.add_parser("show", help="Show one canonical record")
    showing.add_argument("record_id")
    command.add_parser("summary", help="Count records by group and readiness")
    args = parser.parse_args(argv)
    index = load_index()
    if args.command == "show":
        print(json.dumps(get_record(args.record_id, index=index), indent=2, ensure_ascii=False))
    elif args.command == "list":
        matches = list_records(group=args.group, comparison_status=args.comparison_status,
                               evidence_kind=args.evidence_kind, index=index)
        if args.json:
            print(json.dumps(matches, indent=2, ensure_ascii=False))
        else:
            for item in matches:
                print(f"{item['id']}\t{item['group']}\t{item['comparison_status']}\t{item['observable']}")
    else:
        print(json.dumps({"total": len(index["records"]),
                          "by_group": dict(Counter(x["group"] for x in index["records"])),
                          "by_status": dict(Counter(x["comparison_status"] for x in index["records"])),
                          "sources": len(index["sources"]), "aliases": len(index["aliases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
