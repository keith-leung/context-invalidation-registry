"""D1 — Critical Events Registry staleness logic (three branches).

SPEC §3 D1 requires the staleness check to have three correct branches:
  1. matching industry+region AND case_date < event_date → stale
  2. non-matching industry or region → fresh
  3. matching industry+region + no vintage + recent event (<2y) → stale
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from context_invalidation_registry.models import Case, CriticalEvent
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.registry.staleness import check_case_staleness


@pytest.fixture
def food_event() -> CriticalEvent:
    return CriticalEvent(
        event_id="CE-TEST",
        event_name="Test Food Event",
        event_date="2025-09-10",
        affected_industries=["food_local_services"],
        affected_regions=["CN"],
        impact_description="Test",
        invalidated_strategies=["old-strategy"],
        new_required_strategies=["new-strategy"],
    )


@pytest.fixture
def store_with_event(tmp_path, food_event):
    store = JsonEventStore(tmp_path / "events.json")
    store.register_event(food_event)
    return store


class TestBranch1PreEventStale:
    """Branch 1: matching industry+region, case_date < event_date."""

    def test_pre_event_case_flagged_stale(self, store_with_event):
        case = Case(
            case_id="C1",
            industry="food_local_services",
            region="CN",
            data_collected_at="2024-06-15",  # before 2025-09-10
        )
        report = check_case_staleness(case, store_with_event)
        assert report.is_stale is True
        assert report.event_id == "CE-TEST"
        assert report.event_date == "2025-09-10"

    def test_post_event_case_fresh(self, store_with_event):
        case = Case(
            case_id="C1",
            industry="food_local_services",
            region="CN",
            data_collected_at="2026-01-15",  # after 2025-09-10
        )
        report = check_case_staleness(case, store_with_event)
        assert report.is_stale is False


class TestBranch2NonMatching:
    """Branch 2: no industry/region overlap → fresh regardless of date."""

    def test_wrong_industry_is_fresh(self, store_with_event):
        case = Case(
            case_id="C2",
            industry="fitness_equipment",  # event affects food only
            region="CN",
            data_collected_at="2020-01-01",  # very old
        )
        report = check_case_staleness(case, store_with_event)
        assert report.is_stale is False

    def test_wrong_region_is_fresh(self, store_with_event):
        case = Case(
            case_id="C3",
            industry="food_local_services",
            region="US",  # event affects CN only
            data_collected_at="2020-01-01",
        )
        report = check_case_staleness(case, store_with_event)
        assert report.is_stale is False


class TestBranch3ConservativeNoVintage:
    """Branch 3: matching + no vintage + recent event → stale (conservative)."""

    def test_no_vintage_recent_event_flagged_stale(self, store_with_event):
        case = Case(
            case_id="C4",
            industry="food_local_services",
            region="CN",
            # no data_collected_at
        )
        report = check_case_staleness(case, store_with_event)
        # 2025-09-10 is < 2 years old as of test time in 2026-07-15
        assert report.is_stale is True

    def test_no_vintage_old_event_is_fresh(self, tmp_path):
        old_event = CriticalEvent(
            event_id="CE-OLD",
            event_name="Old Event",
            event_date=str(date.today() - timedelta(days=1000)),  # ~2.7y ago
            affected_industries=["food_local_services"],
            affected_regions=["CN"],
            impact_description="Old",
            invalidated_strategies=[],
            new_required_strategies=[],
        )
        store = JsonEventStore(tmp_path / "events2.json")
        store.register_event(old_event)
        case = Case(
            case_id="C5",
            industry="food_local_services",
            region="CN",
        )
        report = check_case_staleness(case, store)
        assert report.is_stale is False
