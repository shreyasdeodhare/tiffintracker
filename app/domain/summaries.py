"""Deterministic, bounded summary formatting. No transport side effects."""

import re
from decimal import Decimal


def safe_name(value):
    return re.sub(r"[\x00-\x1f\x7f*_~`\u200e\u200f]", " ", value).strip()[:100] or "Participant"


def render_summary(summary_id, kind, snapshot, max_chars=3000):
    status = "PROVISIONAL" if snapshot["provisional"] else "Reconciled snapshot"
    heading = (
        f"TiffinTrack — {kind.title()} Order Summary\n"
        f"Period: {snapshot['start']} to {snapshot['end']}\n"
        f"Snapshot: {snapshot['cutoff']}\nSummary #{summary_id} · {status}\n"
    )
    rows = [
        f"{i}. {safe_name(row['name'])} (#{row['participant_id']}) — "
        f"{row['records']} order records — {row['quantity']} tiffins — INR {Decimal(row['amount']):.2f}"
        for i, row in enumerate(snapshot["participants"], 1)
    ]
    reasons = []
    if snapshot["review"]:
        reasons.append(f"{snapshot['review']} messages awaiting review")
    if snapshot["unresolved"]:
        reasons.append(f"{snapshot['unresolved']} pending/error inbox events")
    if snapshot["coverage_incomplete"]:
        reasons.append("tracking coverage incomplete")
    if snapshot.get("late_capture"):
        reasons.append("snapshot captured after scheduled cutoff")
    if snapshot["provisional"] and not reasons:
        reasons.append("source completeness not established")
    totals = snapshot["totals"]
    footer = (
        f"Participants: {len(rows)}\nOrder records: {totals['records']}\n"
        f"Tiffins: {totals['quantity']}\nAmount: INR {Decimal(totals['amount']):.2f}\n"
        f"Cancelled records: {snapshot['cancelled']}\n"
        f"Review: {snapshot['review']} · Pending/errors: {snapshot['unresolved']}\n"
        f"Coverage: {'incomplete' if snapshot['coverage_incomplete'] else 'tracking enabled for period; upstream completeness not guaranteed'}\n"
        + ("Provisional reasons: " + "; ".join(reasons) + "\n" if reasons else "")
        + "Later changes are not included in this saved snapshot."
    )
    chunks, current = [], []
    for line in rows or ["No confirmed active orders."]:
        if sum(map(len, current)) + len(line) + len(heading) + len(footer) + 100 > max_chars:
            if current:
                chunks.append(current)
            current = []
        current.append(line)
    chunks.append(current)
    return [
        heading + f"Part {i}/{len(chunks)}\n" + "\n".join(lines) + "\n\n" + footer
        for i, lines in enumerate(chunks, 1)
    ]
