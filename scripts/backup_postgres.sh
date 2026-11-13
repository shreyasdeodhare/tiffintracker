#!/usr/bin/env bash
# Host-side backup; keep permissions restrictive and secrets in .pgpass / PGPASSFILE.
set -Eeuo pipefail
umask 077
: "${BACKUP_DIR:=/var/backups/tiffintrack}"
: "${PGHOST:=127.0.0.1}"
: "${PGPORT:=5432}"
: "${PGDATABASE:=tiffintrack}"
: "${PGUSER:=tiffintrack}"
: "${RETENTION_DAYS:=14}"
export PGHOST PGPORT PGDATABASE PGUSER
mkdir -p "$BACKUP_DIR"
file="$BACKUP_DIR/tiffintrack-$(date -u +%Y%m%dT%H%M%SZ).dump"
trap 'rm -f "$file.tmp"' EXIT
pg_dump --format=custom --no-owner --no-acl --file="$file.tmp"
pg_restore --list "$file.tmp" >/dev/null
mv -- "$file.tmp" "$file"
sha256sum "$file" > "$file.sha256"
# Note: structural validation is NOT a full restore test.
find "$BACKUP_DIR" -maxdepth 1 -type f \( -name 'tiffintrack-*.dump' -o -name 'tiffintrack-*.sha256' \) -mtime +"$RETENTION_DAYS" -delete
printf 'Backup complete: %s\n' "$file"
# Off-server upload/alerts intentionally require site-specific configuration.
