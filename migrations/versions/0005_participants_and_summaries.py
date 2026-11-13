"""Canonical guest participants and audited weekly publication snapshots."""

import sqlalchemy as sa
from alembic import op

revision = "0005_participants"
down_revision = "0004_whatsapp"
branch_labels = depends_on = None


def upgrade():
    op.add_column("wa_inbox", sa.Column("processed_at", sa.DateTime(timezone=True)))
    op.alter_column("wa_participants", "member_id", existing_type=sa.Integer(), nullable=True)
    for column in [
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("verified_contact", sa.String(100)),
    ]:
        op.add_column("wa_participants", column)
    op.add_column("orders", sa.Column("participant_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_order_participant",
        "orders",
        "wa_participants",
        ["participant_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.add_column(
        "orders", sa.Column("category", sa.String(20), nullable=False, server_default="unspecified")
    )
    op.add_column(
        "orders",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    # Retain every historical user/mapping/price. Only add canonical references.
    op.execute("""INSERT INTO wa_participants (member_id,display_name,automatically_created)
        SELECT DISTINCT u.id,u.name,false FROM users u JOIN orders o ON o.member_id=u.id
        WHERE NOT EXISTS (SELECT 1 FROM wa_participants p WHERE p.member_id=u.id)""")
    op.execute("""UPDATE orders o SET participant_id=p.id FROM wa_participants p
        WHERE p.member_id=o.member_id AND o.participant_id IS NULL""")
    op.alter_column("orders", "participant_id", existing_type=sa.Integer(), nullable=False)
    op.alter_column("orders", "member_id", existing_type=sa.Integer(), nullable=True)
    op.create_index("ix_orders_participant_id", "orders", ["participant_id"])
    op.create_index("ix_orders_participant_date", "orders", ["participant_id", "order_date"])
    op.create_check_constraint(
        "ck_order_category", "orders", "category IN ('lunch','dinner','unspecified')"
    )
    op.create_table(
        "reconciliation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "summary_publications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("group_jid", sa.String(100), nullable=False),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("source", sa.String(12), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("snapshot_hash", sa.String(64), nullable=False),
        sa.Column(
            "reconciliation_id",
            sa.Integer(),
            sa.ForeignKey("reconciliation_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String(100)),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "group_jid", "kind", "period_start", "period_end", "source", name="uq_summary_period"
        ),
        sa.CheckConstraint(
            "status IN ('pending','sending','sent','failed','unknown')", name="ck_summary_state"
        ),
        sa.CheckConstraint("kind='weekly'", name="ck_summary_weekly_only"),
        sa.CheckConstraint("source IN ('all','manual','whatsapp')", name="ck_summary_source"),
    )
    op.create_table(
        "summary_parts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "publication_id",
            sa.Integer(),
            sa.ForeignKey("summary_publications.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("message_id", sa.String(100), nullable=False, unique=True),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String(100)),
        sa.UniqueConstraint("publication_id", "position", name="uq_summary_part"),
        sa.CheckConstraint(
            "status IN ('pending','sending','sent','failed','unknown')",
            name="ck_summary_part_state",
        ),
    )
    op.create_index("ix_summary_parts_publication_id", "summary_parts", ["publication_id"])
    op.create_table(
        "publication_audits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "publication_id",
            sa.Integer(),
            sa.ForeignKey("summary_publications.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="RESTRICT")),
        sa.Column("action", sa.String(30), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_publication_audits_publication_id", "publication_audits", ["publication_id"]
    )

    op.create_table(
        "participant_audits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "participant_id",
            sa.Integer(),
            sa.ForeignKey("wa_participants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="RESTRICT")),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_participant_audits_participant_id", "participant_audits", ["participant_id"]
    )
    op.execute(
        "INSERT INTO participant_audits(participant_id,action,detail,created_at) SELECT id,'migration_preserved',json_build_object('member_id',member_id),now() FROM wa_participants"
    )

    op.execute("""CREATE FUNCTION tt_guard_summary_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.group_jid,NEW.kind,NEW.period_start,NEW.period_end,NEW.source,NEW.snapshot::text,NEW.snapshot_hash,NEW.reconciliation_id)
         IS DISTINCT FROM ROW(OLD.group_jid,OLD.kind,OLD.period_start,OLD.period_end,OLD.source,OLD.snapshot::text,OLD.snapshot_hash,OLD.reconciliation_id) THEN
        RAISE EXCEPTION 'A published snapshot is immutable' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END; $$""")
    op.execute(
        "CREATE TRIGGER tt_summary_snapshot_immutable BEFORE UPDATE ON summary_publications FOR EACH ROW EXECUTE FUNCTION tt_guard_summary_snapshot()"
    )
    op.execute("""CREATE FUNCTION tt_guard_summary_part() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.publication_id,NEW.position,NEW.body,NEW.message_id)
         IS DISTINCT FROM ROW(OLD.publication_id,OLD.position,OLD.body,OLD.message_id) THEN
        RAISE EXCEPTION 'Summary part contents are immutable' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END; $$""")
    op.execute(
        "CREATE TRIGGER tt_summary_part_immutable BEFORE UPDATE ON summary_parts FOR EACH ROW EXECUTE FUNCTION tt_guard_summary_part()"
    )

    op.create_table(
        "order_audits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "order_id",
            sa.Integer(),
            sa.ForeignKey("orders.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="RESTRICT")),
        sa.Column("action", sa.String(30), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_order_audits_order_id", "order_audits", ["order_id"])
    op.execute(
        "INSERT INTO order_audits(order_id,action,detail,created_at) SELECT id,'migration_preserved',json_build_object('quantity',quantity,'unit_price',unit_price::text,'status',status),now() FROM orders"
    )


def downgrade():
    raise RuntimeError(
        "Guest orders and publication audit cannot be safely downgraded; use a reviewed forward migration"
    )
