import re
from io import BytesIO

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from sqlalchemy.orm import joinedload

from app.models import Order, User
from app.whatsapp_models import WAParticipant

from .finance import report_snapshot, weekly_totals
from .reporting import ReportingRepository


def excel_safe(value):
    if not isinstance(value, str):
        return value
    # XML 1.0 cannot represent these characters; untrusted display names must
    # not turn an otherwise valid report into an HTTP 500.
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return (
        "'" + value
        if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r"))
        else value
    )


def weekly_workbook(start, end):
    with report_snapshot() as session:
        return _weekly_workbook(start, end, session)


def _weekly_workbook(start, end, session):
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Weekly Summary")
    ws.title = "Weekly Summary"
    ws.append(["Member", "Email", "Tiffin quantity", "Amount (INR)", "Payment Status"])
    users = {u.id: u for u in session.query(User).filter_by(role="member").all()}
    payments = {
        item["member_id"]: item["payment_status"]
        for item in weekly_totals(start, end, session=session)
    }
    mapping = dict(session.query(WAParticipant.id, WAParticipant.member_id).all())
    for item in ReportingRepository(session).totals(start, end):
        member_id = mapping[item["participant_id"]]
        user = users.get(member_id)
        ws.append(
            [
                excel_safe(item["name"]),
                excel_safe(user.email) if user else "",
                item["quantity"],
                money_cell(ws, item["amount"]),
                payments.get(member_id, "pending"),
            ]
        )
    detail = wb.create_sheet("Order Details")
    detail.append(
        ["Order ID", "Member", "Email", "Date", "Quantity", "Unit Price", "Status", "Source"]
    )
    for order in (
        session.query(Order)
        .options(joinedload(Order.participant))
        .filter(Order.order_date >= start, Order.order_date <= end)
        .order_by(Order.order_date, Order.id)
        .yield_per(200)
    ):
        user = users.get(order.member_id)
        detail.append(
            [
                order.id,
                excel_safe(order.participant.display_name),
                excel_safe(user.email) if user else "",
                order.order_date.isoformat(),
                order.quantity,
                money_cell(detail, order.unit_price),
                order.status,
                order.source,
            ]
        )
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def money_cell(sheet, amount):
    cell = WriteOnlyCell(sheet, value=amount)
    cell.number_format = "#,##0.00"
    return cell
