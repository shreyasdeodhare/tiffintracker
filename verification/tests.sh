set -Eeuo pipefail
STAGE=/var/tmp/tiffintrack-final-acceptance-20261112
mkdir -p "$STAGE"
tar xzf /home/support/tiffintrack-stabilization-candidate.tgz -C "$STAGE"
chmod -R a+rX "$STAGE"
PY=/var/tmp/tiffintrack-verify-venv/bin/python
for name in tiffintrack_final_acceptance_test tiffintrack_final_acceptance_migration_test; do
 if runuser -u postgres -- psql -Atqc "SELECT 1 FROM pg_database WHERE datname='$name'" | grep -q 1; then echo 'Test database exists; refusing overwrite';exit 1;fi
 runuser -u postgres -- createdb "$name"
done
cd "$STAGE"
"$PY" -m compileall -q app scripts tests
runuser -u postgres -- env TEST_POSTGRES_URL=postgresql+psycopg:///tiffintrack_final_acceptance_test TEST_MIGRATION_URL=postgresql+psycopg:///tiffintrack_final_acceptance_migration_test "$PY" -m pytest -q -p no:cacheprovider
/var/tmp/tiffintrack-verify-venv/bin/ruff check .
/var/tmp/tiffintrack-verify-venv/bin/ruff format --check .
(cd bridge && npm test)
# Reuse the installed, locked dependencies without connecting to WhatsApp.
docker cp tiffintrack:/app/bridge/node_modules "$STAGE/bridge/"
chmod -R a+rX "$STAGE/bridge/node_modules"
runuser -u postgres -- env TEST_POSTGRES_DB=tiffintrack_final_acceptance_test node bridge/test/publications-postgres.mjs
echo STABILIZATION_ISOLATED_POSTGRES_AND_OUTBOX_VERIFICATION_COMPLETE

cd /home/support/tiffintrack
docker compose config --quiet
echo FINAL_EXISTING_TESTS_AND_COMPOSE_PASSED
