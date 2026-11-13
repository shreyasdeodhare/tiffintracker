"""Live WhatsApp tracking: durable inbox, stable identities and encrypted session state."""

import sqlalchemy as sa
from alembic import op

revision = "0004_whatsapp"
down_revision = "0003_submission"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("billing_email_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "orders", sa.Column("source", sa.String(12), nullable=False, server_default="manual")
    )
    op.create_check_constraint("ck_orders_source", "orders", "source IN ('manual','whatsapp')")
    op.execute("""
CREATE TABLE wa_auth (
	key VARCHAR(300) NOT NULL, 
	encrypted_value TEXT NOT NULL, 
	PRIMARY KEY (key)
)
""")
    op.execute("""
CREATE TABLE wa_inbox (
	id SERIAL NOT NULL, 
	event_key VARCHAR(64) NOT NULL, 
	payload JSON NOT NULL, 
	state VARCHAR(12) DEFAULT 'pending' NOT NULL, 
	attempts INTEGER DEFAULT '0' NOT NULL,
	deliveries INTEGER DEFAULT '1' NOT NULL, 
	error VARCHAR(100), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_wa_inbox_state CHECK (state IN ('pending','processed','error')), 
	UNIQUE (event_key)
)
""")
    op.execute("CREATE INDEX ix_wa_inbox_state ON wa_inbox (state)")
    op.execute("""
CREATE TABLE wa_connection (
	id SERIAL NOT NULL, 
	group_jid VARCHAR(100), 
	group_name VARCHAR(200), 
	enabled_since TIMESTAMP WITH TIME ZONE, 
	selected_by INTEGER, 
	status VARCHAR(40) NOT NULL, 
	heartbeat TIMESTAMP WITH TIME ZONE, 
	worker_heartbeat TIMESTAMP WITH TIME ZONE, 
	last_received TIMESTAMP WITH TIME ZONE, 
	error VARCHAR(200), 
	PRIMARY KEY (id), 
	UNIQUE (group_jid), 
	FOREIGN KEY(selected_by) REFERENCES users (id) ON DELETE RESTRICT
)
""")
    op.execute("""
CREATE TABLE wa_participants (
	id SERIAL NOT NULL, 
	member_id INTEGER NOT NULL, 
	display_name VARCHAR(100) NOT NULL, 
	automatically_created BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (member_id), 
	FOREIGN KEY(member_id) REFERENCES users (id) ON DELETE RESTRICT
)
""")
    op.execute("""
CREATE TABLE wa_events (
	id SERIAL NOT NULL, 
	inbox_id INTEGER NOT NULL, 
	group_jid VARCHAR(100) NOT NULL, 
	message_id VARCHAR(150) NOT NULL, 
	kind VARCHAR(12) NOT NULL, 
	sender VARCHAR(100) NOT NULL, 
	participant_id INTEGER, 
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
	body TEXT NOT NULL, 
	category VARCHAR(20) NOT NULL, 
	confidence VARCHAR(10) NOT NULL, 
	state VARCHAR(20) NOT NULL, 
	reason VARCHAR(500) NOT NULL, 
	order_date DATE NOT NULL, 
	quantity INTEGER, 
	order_id INTEGER, 
	target_message_id VARCHAR(150), 
	PRIMARY KEY (id), 
	CONSTRAINT ck_wa_event_state CHECK (state IN ('review','applied','ignored','rejected')), 
	CONSTRAINT ck_wa_event_confidence CHECK (confidence IN ('high','medium','low')), 
	CONSTRAINT ck_wa_event_quantity CHECK (quantity IS NULL OR quantity BETWEEN 1 AND 20), 
	UNIQUE (inbox_id), 
	FOREIGN KEY(inbox_id) REFERENCES wa_inbox (id) ON DELETE RESTRICT, 
	FOREIGN KEY(participant_id) REFERENCES wa_participants (id) ON DELETE RESTRICT, 
	FOREIGN KEY(order_id) REFERENCES orders (id) ON DELETE RESTRICT
)
""")
    op.execute("CREATE INDEX ix_wa_events_order_id ON wa_events (order_id)")
    op.execute("CREATE INDEX ix_wa_events_state ON wa_events (state)")
    op.execute("CREATE INDEX ix_wa_participant_time ON wa_events (participant_id, timestamp)")
    op.execute("CREATE INDEX ix_wa_source ON wa_events (group_jid, message_id, kind)")
    op.execute("""
CREATE TABLE wa_identities (
	jid VARCHAR(100) NOT NULL, 
	participant_id INTEGER NOT NULL, 
	PRIMARY KEY (jid), 
	FOREIGN KEY(participant_id) REFERENCES wa_participants (id) ON DELETE RESTRICT
)
""")
    op.execute("CREATE INDEX ix_wa_identities_participant_id ON wa_identities (participant_id)")
    op.execute("""
CREATE TABLE wa_audits (
	id SERIAL NOT NULL, 
	event_id INTEGER NOT NULL, 
	order_id INTEGER, 
	actor_id INTEGER, 
	action VARCHAR(40) NOT NULL, 
	detail JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_id) REFERENCES wa_events (id) ON DELETE RESTRICT, 
	FOREIGN KEY(order_id) REFERENCES orders (id) ON DELETE RESTRICT, 
	FOREIGN KEY(actor_id) REFERENCES users (id) ON DELETE RESTRICT
)
""")
    op.execute("CREATE INDEX ix_wa_audits_event_id ON wa_audits (event_id)")
    op.execute("INSERT INTO wa_connection (id,status) VALUES (1,'not_started')")


def downgrade():
    raise RuntimeError("WhatsApp history is financial audit data; use a reviewed forward migration")
