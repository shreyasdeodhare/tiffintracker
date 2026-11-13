"""A single supervised worker drains durable PostgreSQL events, with no extra container."""

import time

from app import create_app
from app.extensions import db
from app.whatsapp_service import process_next


def main():
    app = create_app()
    while True:
        with app.app_context():
            try:
                if not process_next():
                    time.sleep(1)
            except Exception as exc:
                db.session.rollback()
                app.logger.error("WhatsApp worker unavailable type=%s", type(exc).__name__)
                time.sleep(5)


if __name__ == "__main__":
    main()
