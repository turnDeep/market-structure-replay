"""Gate scheduled NQ/ES refreshes on the previous CME equity trade date.

At 14:10 JST Tuesday-Saturday (just after midnight in New York),
yesterday's calendar date is the US trade date that has finished. Japanese public holidays are irrelevant. Do not use
an NYSE calendar: CME can have shortened sessions on US stock-market holidays.
This gate controls scheduling only; it does not relax the price-finality,
append-only, verification or publication rules in cloud.refresh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

JST = ZoneInfo('Asia/Tokyo')
CALENDAR_NAME = 'CME_Equity'


def refresh_decision(now: datetime, event_name: str = 'schedule', *, calendar=None) -> dict:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('A timezone-aware execution clock is required')
    if event_name not in ('schedule', 'workflow_dispatch'):
        raise ValueError('Only scheduled or explicitly dispatched refreshes are supported')
    local = now.astimezone(JST)
    session_day = local.date() - timedelta(days=1)
    result = {
        'should_refresh': False,
        'event_name': event_name,
        'checked_at_utc': now.astimezone(timezone.utc).isoformat(),
        'update_date_jst': local.date().isoformat(),
        'session_date': session_day.isoformat(),
        'calendar': CALENDAR_NAME,
        'calendar_version': mcal.__version__,
        'scheduled_time_jst': '14:10',
    }
    if event_name == 'workflow_dispatch':
        return result | {'should_refresh': True, 'reason': 'explicit_manual_refresh'}
    if local.weekday() not in (1, 2, 3, 4, 5):
        return result | {'reason': 'not_tuesday_to_saturday_jst'}
    if local.time() < time(9):
        return result | {'reason': 'before_0900_jst'}
    cal = calendar if calendar is not None else mcal.get_calendar(CALENDAR_NAME)
    schedule = cal.schedule(start_date=session_day, end_date=session_day)
    if schedule.empty:
        return result | {'reason': 'previous_trade_date_is_fully_closed'}
    if len(schedule) != 1 or schedule.index[0].date() != session_day:
        raise ValueError('Calendar returned an unexpected trade date')
    session_close = schedule.iloc[0]['market_close'].to_pydatetime()
    if session_close.tzinfo is None:
        raise ValueError('Calendar close must be timezone-aware')
    result['calendar_close_utc'] = session_close.astimezone(timezone.utc).isoformat()
    if session_close > now:
        return result | {'reason': 'previous_session_has_not_closed'}
    return result | {'should_refresh': True, 'reason': 'previous_cme_equity_session_closed'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-name', choices=['schedule', 'workflow_dispatch'], default='schedule')
    parser.add_argument('--now', help='Optional timezone-aware ISO timestamp for deterministic dry checks')
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    # Exceptions intentionally fail closed: no publishing on calendar failures.
    decision = refresh_decision(now, args.event_name)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if args.github_output:
        with args.github_output.open('a', encoding='utf-8') as handle:
            handle.write(f"should_refresh={str(decision['should_refresh']).lower()}\n")
            handle.write(f"session_date={decision['session_date']}\n")


if __name__ == '__main__':
    main()
