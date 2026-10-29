from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from cloud.schedule import refresh_decision


@pytest.mark.parametrize('jst,expected,session', [
    ('2026-10-27T09:00:00+09:00', True, '2026-10-26'),
    ('2026-10-31T09:00:00+09:00', True, '2026-10-30'),
    ('2026-11-01T09:00:00+09:00', False, '2026-10-31'),
    ('2026-11-02T09:00:00+09:00', False, '2026-11-01'),
    ('2026-10-27T08:59:59+09:00', False, '2026-10-26'),
    ('2026-10-27T09:41:00+09:00', True, '2026-10-26'),
    ('2026-07-16T09:00:00+09:00', True, '2026-07-15'),
    ('2026-01-16T09:00:00+09:00', True, '2026-01-15'),
    ('2026-03-10T09:00:00+09:00', True, '2026-03-09'),
    ('2026-11-03T09:00:00+09:00', True, '2026-11-02'),
    ('2026-12-26T09:00:00+09:00', False, '2026-12-25'),
    ('2027-01-02T09:00:00+09:00', False, '2027-01-01'),
    ('2026-12-25T09:00:00+09:00', True, '2026-12-24'),
    # US equities closed, but CME equity futures have shortened sessions.
    ('2026-11-27T09:00:00+09:00', True, '2026-11-26'),
    ('2026-11-28T09:00:00+09:00', True, '2026-11-27'),
    ('2026-04-04T09:00:00+09:00', True, '2026-04-03'),
    # Japan's public holidays do not govern US futures updates.
    ('2026-09-22T09:00:00+09:00', True, '2026-09-21'),
])
def test_cme_following_day_gate(jst, expected, session):
    result = refresh_decision(datetime.fromisoformat(jst))
    assert result['should_refresh'] is expected
    assert result['session_date'] == session


def test_utc_and_jst_clocks_agree():
    a = refresh_decision(datetime.fromisoformat('2026-10-27T00:00:00+00:00'))
    b = refresh_decision(datetime.fromisoformat('2026-10-27T09:00:00+09:00'))
    assert a == b


def test_manual_refresh_is_available_on_weekends():
    result = refresh_decision(datetime.fromisoformat('2026-11-01T08:00:00+09:00'), 'workflow_dispatch')
    assert result['should_refresh'] and result['reason'] == 'explicit_manual_refresh'


def test_naive_clock_and_unrelated_event_rejected():
    with pytest.raises(ValueError, match='timezone-aware'):
        refresh_decision(datetime(2026, 10, 27, 9))
    with pytest.raises(ValueError, match='explicitly dispatched'):
        refresh_decision(datetime.fromisoformat('2026-10-27T09:00:00+09:00'), 'push')


def test_calendar_failure_is_not_treated_as_a_holiday():
    class BrokenCalendar:
        def schedule(self, **kwargs):
            raise RuntimeError('calendar unavailable')
    with pytest.raises(RuntimeError, match='unavailable'):
        refresh_decision(datetime.fromisoformat('2026-10-27T09:00:00+09:00'), calendar=BrokenCalendar())


def test_future_session_does_not_run():
    class FutureClose:
        def schedule(self, **kwargs):
            return pd.DataFrame({'market_close': [pd.Timestamp('2026-10-28', tz='UTC')]},
                                index=pd.DatetimeIndex(['2026-10-26']))
    result = refresh_decision(datetime.fromisoformat('2026-10-27T09:00:00+09:00'), calendar=FutureClose())
    assert not result['should_refresh']


def test_cli_only_writes_safe_outputs(tmp_path):
    output = tmp_path/'github_output'
    result = subprocess.run([sys.executable, '-m', 'cloud.schedule', '--now',
                             '2026-12-26T09:00:00+09:00', '--github-output', str(output)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'should_refresh=false\nsession_date=2026-12-25\n'
