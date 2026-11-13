"""Host cron entry point: Friday 14:00 IST only; never schedules daily messages."""

from app import create_app
from app.application.publications import prepare_weekly
from app.extensions import db


def main():
    with create_app().app_context():
        try:
            identity = prepare_weekly()
            print(
                f"weekly_summary_prepared id={identity}; delivery depends on enabled connected receiver"
            )
        except Exception as exc:
            db.session.rollback()
            print(f"weekly_summary_failed type={type(exc).__name__}")
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
