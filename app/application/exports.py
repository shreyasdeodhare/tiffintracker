"""Snapshot export use cases, shared with participant dashboard projections."""

from io import BytesIO

from openpyxl import Workbook

from app.domain.orders import month_bounds, reporting_week
from app.infrastructure.inbox import InboxRepository
from app.infrastructure.reporting import ReportingRepository
from app.reports import excel_safe, money_cell
from app.services import report_snapshot


def summaries(day, source="all", member_id=None, session=None):
    if session is None:
        with report_snapshot() as snapshot:
            return summaries(day, source, member_id, snapshot)
    identity = None
    if member_id is not None:
        participant = InboxRepository(session).member_participant(member_id)
        if participant is None:
            return {key: [] for key in ["daily", "weekly", "monthly"]}
        identity = participant.id
    repo = ReportingRepository(session)
    return {
        "daily": repo.totals(day, day, source, identity),
        "weekly": repo.totals(*reporting_week(day), source, identity),
        "monthly": repo.totals(*month_bounds(day), source, identity),
    }


def counts_workbook(day, source):
    snapshot = summaries(day, source)
    book = Workbook(write_only=True)
    for period, rows in snapshot.items():
        sheet = book.create_sheet(period.title())
        sheet.append(
            ["Participant ID", "Participant", "Order records", "Tiffin quantity", "Amount INR"]
        )
        for row in rows:
            sheet.append(
                [
                    row["id"],
                    excel_safe(row["name"]),
                    row["orders"],
                    row["quantity"],
                    money_cell(sheet, row["amount"]),
                ]
            )
    output = BytesIO()
    book.save(output)
    output.seek(0)
    return output
