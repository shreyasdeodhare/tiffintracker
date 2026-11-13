"""Weekly invoices; intentionally fail-closed on ambiguous SMTP outcomes."""

import argparse
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage

from sqlalchemy.exc import IntegrityError

from app import create_app
from app.extensions import db
from app.models import EmailLog, User
from app.services import local_today, lock_member, previous_week, weekly_totals


def send_email(recipient, subject, body):
    required = ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"]
    if not all(os.getenv(k) for k in required):
        raise ValueError("SMTP configuration incomplete")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = recipient
    msg.set_content(body)
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    mode = os.getenv("SMTP_SECURITY", "starttls").lower()
    if mode not in {"starttls", "ssl"}:
        raise ValueError("SMTP_SECURITY must be starttls or ssl")
    if mode == "ssl":
        transport = smtplib.SMTP_SSL(host, port, timeout=20, context=ssl.create_default_context())
    else:
        transport = smtplib.SMTP(host, port, timeout=20)
    with transport as smtp:
        smtp.ehlo()
        if mode == "starttls":
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)


def run(send=False, retry_failed=False):
    start, end = previous_week(local_today())
    totals = {row["member_id"]: row for row in weekly_totals(start, end)}
    counts = {"sent": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    # A PostgreSQL streaming cursor cannot survive the per-recipient commits.
    member_ids = db.session.scalars(
        db.select(User.id)
        .filter_by(role="member", is_active=True, billing_email_enabled=True)
        .order_by(User.id)
    ).all()
    for member_id in member_ids:
        user = db.session.get(User, member_id)
        item = totals.get(user.id)
        if not item:
            continue
        # Avoid exposing recipient addresses in cron logs.
        if not send:
            print(f"DRY_RUN member_id={user.id} amount={item['amount']}")
            counts["dry_run"] += 1
            continue
        user = lock_member(user.id)
        db.session.refresh(user)
        if not user.is_active or user.role != "member" or not user.billing_email_enabled:
            db.session.rollback()
            counts["skipped"] += 1
            continue
        fresh = weekly_totals(start, end, user.id)
        if not fresh:
            db.session.rollback()
            continue
        item = fresh[0]
        existing = (
            EmailLog.query.filter_by(member_id=user.id, week_start=start, week_end=end)
            .order_by(EmailLog.id.desc())
            .first()
        )
        if existing and (existing.status != "failed" or not retry_failed):
            db.session.rollback()
            counts["skipped"] += 1
            continue
        # Commit 'sending' before communicating with SMTP. An interrupted send becomes
        # ambiguous and requires manual reconciliation, NOT automatic retry.
        if existing:
            log = existing
            log.status = "sending"
            log.error = None
            log.recipient = user.email
        else:
            log = EmailLog(
                member_id=user.id,
                week_start=start,
                week_end=end,
                recipient=user.email,
                status="sending",
            )
            db.session.add(log)
        # Freeze the exact recipient and body used by this claim. Later ORM
        # refreshes must not redirect a claimed invoice to a different address.
        recipient = log.recipient
        body = (
            f"Hello {user.name},\n\nYour tiffin bill for {start} to {end}:\n"
            f"Tiffins: {item['orders']}\nTotal: INR {item['amount']:.2f}\n\nThank you."
        )
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            counts["skipped"] += 1
            continue
        try:
            send_email(recipient, f"Tiffin bill: {start} to {end}", body)
        except (smtplib.SMTPException, OSError, ValueError) as exc:
            db.session.rollback()
            # Some SMTP failures happen after remote acceptance; leave ambiguous state
            # to prevent an unintended duplicate email.
            log = db.session.get(EmailLog, log.id)
            log.status = (
                "failed"
                if isinstance(
                    exc,
                    (
                        ValueError,
                        smtplib.SMTPRecipientsRefused,
                        smtplib.SMTPAuthenticationError,
                        smtplib.SMTPConnectError,
                    ),
                )
                else "unknown"
            )
            log.error = type(exc).__name__
            db.session.commit()
            counts["failed"] += 1
            print(f"DELIVERY_{log.status.upper()} member_id={user.id} type={type(exc).__name__}")
            continue
        db.session.rollback()
        log = db.session.get(EmailLog, log.id)
        log.status = "sent"
        log.sent_at = datetime.now(timezone.utc)
        db.session.commit()
        counts["sent"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    parser.add_argument(
        "--status", action="store_true", help="Inspect previous-week delivery states"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retries only explicitly failed rows, not unknown SMTP outcomes",
    )
    args = parser.parse_args()
    app = create_app()
    with app.app_context():
        if args.status:
            start, end = previous_week(local_today())
            for log in EmailLog.query.filter_by(week_start=start, week_end=end).order_by(
                EmailLog.member_id
            ):
                print(f"member_id={log.member_id} state={log.status} error={log.error or '-'}")
            return
        counts = run(send=args.send, retry_failed=args.retry_failed)
        print(counts)
        if counts["failed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
