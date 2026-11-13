#!/usr/bin/env bash
# Run as root only after the reviewed migration has been explicitly approved.
set -Eeuo pipefail
if [[ "${1:-}" != --approved ]]; then
  echo "Requires explicit migration/weekly-publication approval and --approved"; exit 2
fi
TIFFIN_BASE=/home/support/tiffintrack
TIFFIN_STAGE=/var/tmp/tiffintrack-layered-20261112
TIFFIN_PY=/var/tmp/tiffintrack-verify-venv/bin/python
TIFFIN_BACKUP=/var/backups/tiffintrack/releases/layered-20261112
exec 9>/run/lock/tiffintrack-release.lock
flock -n 9 || { echo 'Another release is active'; exit 1; }
[[ "$(timedatectl show --property=Timezone --value)" == Asia/Kolkata ]] || { echo 'Verify the IST cron timezone first'; exit 1; }
[[ "$(docker inspect tiffintrack --format '{{.Config.Image}}')" == tiffintrack:audit-fix-20261112 ]] || { echo 'Production image differs from reviewed baseline'; exit 1; }
[[ "$(runuser -u postgres -- psql -d tiffintrack -Atqc 'SELECT version_num FROM alembic_version')" == 0004_whatsapp ]] || exit 1
docker image inspect tiffintrack:layered-20261112 >/dev/null
umask 077
mkdir -p "$TIFFIN_BACKUP"
chmod 700 "$TIFFIN_BACKUP"
cp "$TIFFIN_BASE/.env" "$TIFFIN_BACKUP/environment-before.env"
cp "$TIFFIN_BASE/docker-compose.yml" "$TIFFIN_BACKUP/docker-compose-before.yml"
tar --exclude=.env --exclude=node_modules --exclude=__pycache__ --exclude=.pytest_cache -czf "$TIFFIN_BACKUP/source-before.tgz" -C "$TIFFIN_BASE" .

failure() {
  trap - ERR
  set +e
  revision=$(runuser -u postgres -- psql -d tiffintrack -Atqc 'SELECT version_num FROM alembic_version')
  if [[ "$revision" == 0004_whatsapp ]]; then
    echo 'Migration did not advance; restarting unchanged baseline'
    cd "$TIFFIN_BASE" && docker compose up -d --no-build
  else
    echo 'Schema advanced: retain compatible code and investigate; never start the old writer against this schema'
  fi
  exit 1
}
trap failure ERR
cd "$TIFFIN_BASE"
docker compose stop tiffintrack
runuser -u postgres -- pg_dump -Fc --no-owner --no-acl tiffintrack > "$TIFFIN_BACKUP/production-before-migration.dump"
pg_restore --list "$TIFFIN_BACKUP/production-before-migration.dump" >/dev/null
sha256sum "$TIFFIN_BACKUP/production-before-migration.dump" > "$TIFFIN_BACKUP/production-before-migration.dump.sha256"
sha256sum -c "$TIFFIN_BACKUP/production-before-migration.dump.sha256"
cd "$TIFFIN_STAGE"
TIFFIN_MIGRATION_APPROVED=yes PYTHONPATH="$TIFFIN_STAGE" "$TIFFIN_PY" deploy/apply_layered_migration.py

# Preserve secrets. Copy only reviewed source, then enable the authorized weekly feature.
for directory in app scripts migrations deploy; do
  mkdir -p "$TIFFIN_BASE/$directory"
  cp -a "$TIFFIN_STAGE/$directory/." "$TIFFIN_BASE/$directory/"
done
mkdir -p "$TIFFIN_BASE/bridge"
tar --exclude=node_modules -cf - -C "$TIFFIN_STAGE/bridge" . | tar -xf - -C "$TIFFIN_BASE/bridge"
cp "$TIFFIN_STAGE/manage.py" "$TIFFIN_STAGE/README.md" "$TIFFIN_STAGE/LAYERED-IMPLEMENTATION.md" "$TIFFIN_STAGE/docker-compose.yml" "$TIFFIN_STAGE/Dockerfile" "$TIFFIN_BASE/"
"$TIFFIN_PY" - <<'PY'
import os
from pathlib import Path
p=Path('/home/support/tiffintrack/.env')
lines=[line for line in p.read_text().splitlines() if not line.startswith('SUMMARY_PUBLISHING_ENABLED=')]
temporary=p.with_name('.env.layered.tmp')
temporary.write_text('\n'.join(lines)+'\nSUMMARY_PUBLISHING_ENABLED=true\n')
temporary.chmod(0o600)
os.replace(temporary,p)
PY
chown -R support:support "$TIFFIN_BASE/app" "$TIFFIN_BASE/scripts" "$TIFFIN_BASE/migrations" "$TIFFIN_BASE/bridge" "$TIFFIN_BASE/deploy"
chown support:support "$TIFFIN_BASE/.env" "$TIFFIN_BASE/manage.py" "$TIFFIN_BASE/docker-compose.yml"
cd "$TIFFIN_BASE"
docker compose config --quiet
docker compose up -d --no-build
healthy=false
for attempt in $(seq 1 40); do
  if [[ "$(docker inspect tiffintrack --format '{{.State.Health.Status}}')" == healthy ]]; then healthy=true;break;fi
  sleep 1
done
[[ "$healthy" == true ]]
install -o root -g root -m 644 deploy/tiffintrack-summary.cron.example /etc/cron.d/tiffintrack-summary
systemctl is-active cron
docker compose exec -T tiffintrack flask --app manage.py db current
docker compose exec -T tiffintrack python - <<'PY'
from app import create_app
from app.application.monitoring import tracking_status
from app.application.reporting import period_report
from app.domain.orders import reporting_week
from app.services import local_today
with create_app().app_context() as context:
    report=period_report(*reporting_week(local_today()))
    assert report['reconciled']
    print('canonical_totals',report['totals'],'review',report['review'],'provisional',report['provisional'])
    print('weekly_enabled',context.app.config['SUMMARY_PUBLISHING_ENABLED'])
    print('components',tracking_status())
PY
curl --cacert /etc/tiffintrack/tls/server.crt --fail --silent https://192.168.1.101/health
docker inspect tiffintrack --format 'image={{.Config.Image}} health={{.State.Health.Status}} restarts={{.RestartCount}} memory={{.HostConfig.Memory}} cpu={{.HostConfig.NanoCpus}}'
docker stats --no-stream tiffintrack
echo LAYERED_DEPLOYMENT_COMPLETE_WEEKLY_FRIDAY_1400_IST_ONLY
