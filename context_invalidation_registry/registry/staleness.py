"""
Staleness logic — event-driven semantic invalidation (SPEC §3 D1).

Three branches:
1. industry+region overlap AND case_date < event_date -> stale
2. non-matching -> fresh
3. matching+no-vintage+recent-event (<2 years) -> stale (conservative)
"""
from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional

from context_invalidation_registry.models import Case, CriticalEvent, StalenessReport


def check_case_staleness(case: Case, store: "EventStore") -> StalenessReport:
    """
    Check if a case is potentially invalidated by a critical event.

    Returns StalenessReport with is_stale=True/False.
    """
    events: List[CriticalEvent] = store.load_events()
    industry = case.industry
    region = case.region

    for event in events:
        if industry not in event.affected_industries:
            continue
        if region not in event.affected_regions:
            continue

        event_date = event.parsed_date()
        if event_date is None:
            continue

        case_date_str = case.data_collected_at or case.created_at
        if not case_date_str:
            # No date info — conservative: flag if event is recent (< 2 years)
            if (datetime.now().date() - event_date).days < 730:
                return StalenessReport(
                    is_stale=True,
                    reason=event.event_name,
                    event_id=event.event_id,
                    event_date=event.event_date,
                    invalidated_strategies=event.invalidated_strategies,
                )
            continue

        case_date = _parse_date(case_date_str)
        if case_date is None:
            continue

        if case_date < event_date:
            return StalenessReport(
                is_stale=True,
                reason=event.event_name,
                event_id=event.event_id,
                event_date=event.event_date,
                invalidated_strategies=event.invalidated_strategies,
            )

    return StalenessReport(is_stale=False)


def _parse_date(val) -> Optional[date]:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if not isinstance(val, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val, fmt).date()
        except (ValueError, TypeError):
            continue
    return None
