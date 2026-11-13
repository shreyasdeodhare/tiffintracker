# Backup, rollback and recovery

The production database is `tiffintrack`, schema `0005_participants`. These instructions do not authorize a live restore or restart.

## Verified backup behavior

Installed `/usr/local/sbin/tiffintrack-backup` matches `scripts/backup_postgres.sh`. Daily cron runs at 02:00 IST with native custom-format pg_dump, restrictive permissions, temporary-file handling, pg_restore archive validation, SHA-256, 14-day retention and nonzero failure exits. Cron logs to `/var/log/tiffintrack-backup.log`; independent alert delivery is not verified.

`/var/backups/tiffintrack` is mode 0700, owned by postgres. Final audit dumps under `final-acceptance-20261112` are mode 0600. A consistent production snapshot restored into an isolated database with matching counts and content hashes for every table. A separate dump made by the actual installed script also restored and passed schema, reconciliation and Excel checks. Restored encrypted WhatsApp credentials decrypted without opening another WhatsApp connection.

No off-server backup destination or successful remote copy was verified. Supply and test destination access, encryption, retention and recovery-key custody before disaster-recovery acceptance. Local backups cannot cover loss of this server. Recovering encrypted sessions also requires the protected session encryption key. Keep protected secrets in a separate recovery store; never include them or database dumps in the source ZIP.

## Isolated restore procedure

1. Verify the dump checksum and `pg_restore --list`.
2. Create an explicitly named disposable database ending `_test`; refuse to overwrite an existing database.
3. Restore with `pg_restore --exit-on-error --no-owner --no-acl` into that disposable database only. Never connect a second receiver to the restored session.
4. Verify schema, participants, aliases, orders, historical prices, messages, inbox, payments, audits, publications and encrypted credentials. Compare against a consistent export snapshot so normal live key updates do not create false mismatches.
5. Use an isolated application context to reconcile daily/weekly/monthly reports and reopen Excel, with transport disabled.
6. Retain evidence, then clean up only the explicitly named test databases/artifacts under the approved retention plan. Do not use broad drop/delete commands.

## Code rollback

Preserve current `tiffintrack:stabilization-20261112` and earlier `tiffintrack:layered-20261112`. They use the same schema; no migration occurred in stabilization. Before an approved rollback, preserve current source/Compose and database, verify target image/source compatibility, then recreate only the app service using the known image with `--no-build --no-deps`. Do not downgrade schema or restore an old database for a code-only rollback. Preserve protected environment, linked session and retired-email cron cleanup.

The pre-stabilization backup is `/var/backups/tiffintrack/stabilization-20261112-220042`. Its source/Compose is historical; do not blindly restore its cron or credentials. After an approved change verify health, correct group, heartbeats, backlog, preserved records, reports, Excel, HTTPS and CSRF.

## Production disaster recovery

Obtain explicit approval before restoring over production, changing schema or restarting. Preserve current state, choose and verify the recovery point, quantify later orders/messages/publications requiring reconciliation and rehearse first. Review outbox state and uncertain deliveries before enabling publication, since older data must not cause resends. Restore matching protected secrets with the database, then verify historical records, attribution, totals, permissions and linked-device status. Do not reset or relink merely to complete a recovery checklist.
