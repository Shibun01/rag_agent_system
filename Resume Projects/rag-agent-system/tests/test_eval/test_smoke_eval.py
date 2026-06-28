from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).with_name("smoke_eval_dataset.json")


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("item", _load_dataset(), ids=lambda item: item["id"])
def test_smoke_eval_dataset_shape(item):
    assert item["query"].strip()
    assert item["collection_name"].strip()
    assert item["strategy"] in {
        "naive",
        "advanced",
        "corrective",
        "self_rag",
        "hyde",
        "multi_query",
        "graph_rag",
    }
    assert item["expected_keywords"]