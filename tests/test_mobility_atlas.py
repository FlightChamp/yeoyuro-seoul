"""
tests/test_mobility_atlas.py
============================
v2.4 mobility_atlas_mart 불변식. mart 가 없으면 건너뛴다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "marts" / "v2" / "mobility_atlas_mart.parquet"
SHARES = ["exposed_traversal_share", "structural_share", "alternative_opportunity", "temporal_shift_benefit"]


@pytest.fixture(scope="module")
def atlas():
    if not P.exists():
        pytest.skip("mobility_atlas_mart 없음")
    return pd.read_parquet(P)


def test_no_single_score_column(atlas):
    assert not any("stress" in c.lower() or "score" in c.lower() for c in atlas.columns)
    assert not any("stress" in str(d).lower() for d in atlas.atlas_dimension.unique())


def test_share_ranges(atlas):
    s = atlas[atlas.atlas_dimension.isin(SHARES)].value.dropna()
    assert ((s >= 0) & (s <= 1)).all()


def test_low_support_ratios_are_blank(atlas):
    ratio = ["structural_share", "alternative_opportunity", "temporal_shift_benefit"]
    low = atlas[atlas.low_support & atlas.atlas_dimension.isin(ratio)]
    assert low.value.isna().all()
    assert (atlas.n_hot_traversals <= atlas.n_traversals).all()


def test_groups_and_keys(atlas):
    assert set(atlas.time_group) == {"all", "am", "pm"}
    assert not atlas.duplicated(["from_node", "to_node", "time_group", "atlas_dimension"]).any()
