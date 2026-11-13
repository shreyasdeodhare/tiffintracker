from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from flask import current_app

from app.domain.orders import FinalizedPeriod as FinalizedPeriod
from app.domain.orders import week_bounds as week_bounds
from app.infrastructure.finance import (
    ensure_mutable as ensure_mutable,
)
from app.infrastructure.finance import (
    lock_member as lock_member,
)
from app.infrastructure.finance import (
    report_snapshot as report_snapshot,
)
from app.infrastructure.finance import (
    weekly_totals as weekly_totals,
)


def local_today():
    return datetime.now(ZoneInfo(current_app.config["APP_TIMEZONE"])).date()


def previous_week(day: date):
    return week_bounds(day - timedelta(days=7))


def completed_week(start: date, end: date, today=None):
    today = today or local_today()
    return week_bounds(start) == (start, end) and end < week_bounds(today)[0]
