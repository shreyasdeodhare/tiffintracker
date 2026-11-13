set -Eeuo pipefail
cd /home/support/tiffintrack
date -Is
sha256sum -c CODE-MANIFEST.sha256 >/dev/null
echo PRODUCTION_CODE_MANIFEST_MATCH
docker compose config --quiet
docker inspect --format 'image={{.Config.Image}} health={{.State.Health.Status}} restarts={{.RestartCount}} oom={{.State.OOMKilled}} memory={{.HostConfig.Memory}} cpus={{.HostConfig.NanoCpus}} pids={{.HostConfig.PidsLimit}} user={{.Config.User}} readonly={{.HostConfig.ReadonlyRootfs}} restart={{.HostConfig.RestartPolicy.Name}}' tiffintrack
docker stats --no-stream --format 'cpu={{.CPUPerc}} memory={{.MemUsage}} pids={{.PIDs}}' tiffintrack
docker top tiffintrack -eo pid,comm
runuser -u postgres -- psql -d tiffintrack -At <<'SQL'
SELECT 'schema',version_num FROM alembic_version;
SELECT 'counts',json_build_object('users',(select count(*) from users),'participants',(select count(*) from wa_participants),'orders',(select count(*) from orders),'events',(select count(*) from wa_events),'inbox',(select count(*) from wa_inbox),'payments',(select count(*) from payments),'publications',(select count(*) from summary_publications),'session_records',(select count(*) from wa_auth));
SELECT 'tracking',status,group_name,group_jid='120363429522418737@g.us' AS expected_group,extract(epoch from now()-heartbeat)::int AS receiver_age,extract(epoch from now()-worker_heartbeat)::int AS worker_age FROM wa_connection;
SELECT 'inbox',state,count(*) FROM wa_inbox GROUP BY state;
SELECT 'events',state,count(*) FROM wa_events GROUP BY state;
SELECT 'db_connections',state,count(*) FROM pg_stat_activity WHERE datname='tiffintrack' GROUP BY state;
SELECT 'constraints',contype,count(*) FROM pg_constraint WHERE connamespace='public'::regnamespace GROUP BY contype;
SELECT 'indexes',count(*) FROM pg_indexes WHERE schemaname='public';
SELECT 'intake_latency_seconds',count(*),round(avg(extract(epoch from processed_at-created_at))::numeric,3),round(max(extract(epoch from processed_at-created_at))::numeric,3) FROM wa_inbox WHERE state='processed';
SELECT 'genuine_order_evidence',o.id,o.participant_id,o.order_date,o.quantity,o.unit_price,o.status,e.id,e.kind,e.group_jid='120363429522418737@g.us' AS expected_group FROM orders o JOIN wa_audits a ON a.order_id=o.id AND a.action='created' JOIN wa_events e ON e.id=a.event_id ORDER BY o.id LIMIT 10;
SQL
cat /etc/cron.d/tiffintrack /etc/cron.d/tiffintrack-summary
timedatectl show --property=Timezone --value
sha256sum /usr/local/sbin/tiffintrack-backup scripts/backup_postgres.sh
stat -c '%a %U %n' /var/backups/tiffintrack /usr/local/sbin/tiffintrack-backup
find /var/backups/tiffintrack -maxdepth 1 -type f -name '*.dump' -printf '%f %s bytes %TY-%Tm-%Td %TH:%TM\n' | sort | tail -3
for sumfile in /var/backups/tiffintrack/tiffintrack-*.dump.sha256; do [ ! -f "$sumfile" ] || sha256sum -c "$sumfile"; done
df -h / /var/backups/tiffintrack
docker system df
openssl x509 -in /etc/tiffintrack/tls/server.crt -noout -enddate
nginx -t
ss -lnt | awk 'NR==1 || /:5432 |:8087 |:443 |:8090 /'
# Only indicate presence, never print an upload destination or credential.
if grep -RqlE '(rclone|restic|borg|aws s3|rsync.*backup)' /etc/cron.d /usr/local/sbin/tiffintrack-backup 2>/dev/null; then echo OFFSERVER_BACKUP_REFERENCE_FOUND_REQUIRES_VERIFICATION;else echo OFFSERVER_BACKUP_NOT_FOUND_IN_APPLICATION_CRON_OR_SCRIPT;fi
for path in /health /auth/login /admin/orders; do curl --cacert /etc/tiffintrack/tls/server.crt -s -o /dev/null -w "$path status=%{http_code} seconds=%{time_total}\n" "https://192.168.1.101$path"; done
echo READ_ONLY_PRODUCTION_BASELINE_COMPLETE
