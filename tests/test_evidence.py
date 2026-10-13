"""Evidence index provenance and deduplication checks."""

from collections import Counter

from lowsky.evidence import get_record, list_records, load_index, resolve_origin


def test_index_coverage_and_deduplication() -> None:
    index = load_index()
    by_group = Counter(record["group"] for record in index["records"])
    assert by_group["hii_absorption"] == 152
    assert by_group["spacecraft_angular_spectrum"] == 56
    assert by_group["absolute_spectrum"] == 15
    assert by_group["reduced_antenna_spectrum"] == 128
    assert by_group["source_catalogue"] == 1
    assert by_group["source_context"] >= 12
    assert len(list_records(group="radio_loops", index=index)) == 5  # 4 points + 1 dependent fit
    assert len(index["aliases"]) == 4
    assert {alias["canonical_id"] for alias in index["aliases"]} == {
        f"borka2011_loop_{suffix}_t22" for suffix in ("i", "ii", "iii", "iv")
    }


def test_source_trace_and_comparison_semantics() -> None:
    index = load_index()
    for record in index["records"]:
        assert resolve_origin(record) == record["data"]
    assert get_record("panopoulou2021_ic", index=index)["data"]["uncertainty_type"] == "upper_limit_inclusive"
    assert get_record("edges2018_003", index=index)["comparison_status"] == "out_of_band"
    assert get_record("lodess_dr1_catalogue", index=index)["observable"] != "brightness_temperature_map"
    assert list_records(comparison_status="operator_required", index=index)
