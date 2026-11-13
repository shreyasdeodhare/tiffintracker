"""Business rules independent of Flask, SQLAlchemy and WhatsApp transport."""

from datetime import date, timedelta
from decimal import Decimal


class BusinessError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class FinalizedPeriod(BusinessError):
    def __init__(self, message):
        super().__init__(message, 409)


def validate_quantity(quantity):
    if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= 20:
        raise BusinessError("Quantity must be an integer from 1 to 20")


def validate_day(day, today):
    if not today - timedelta(days=14) <= day <= today:
        raise BusinessError("Order date must be within the preceding 14 days, not in the future")


def amount(quantity, price):
    price = Decimal(str(price))
    if not price.is_finite() or price <= 0 or price.as_tuple().exponent < -2:
        raise BusinessError("Historical unit price must be positive with at most two decimals")
    return (price * quantity).quantize(Decimal("0.01"))


def week_bounds(day: date):
    start = day - timedelta(days=day.weekday())
    return start, start + timedelta(days=6)


def month_bounds(day):
    start = day.replace(day=1)
    following = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start, following - timedelta(days=1)


def reporting_week(day):
    """Participant/group reporting week; legacy payment weeks stay Mon–Sun."""
    start = day - timedelta(days=(day.weekday() - 5) % 7)
    return start, start + timedelta(days=6)


def publication_period(kind, day):
    if kind == "weekly" and day.weekday() == 4:
        return day - timedelta(days=6), day  # Saturday through Friday, inclusive.
    raise BusinessError("Weekly publication is due on Friday and covers Saturday–Friday")


def validate_category(category):
    if category not in {"lunch", "dinner", "unspecified"}:
        raise BusinessError("Choose lunch, dinner or unspecified")
    return category
