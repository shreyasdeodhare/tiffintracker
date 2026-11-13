FROM node:24-bookworm-slim@sha256:0998f9a7af5bba96fb32e2aeafde650a625012cb194a57e7f3939bf3e592721f AS node-dependencies
WORKDIR /bridge
COPY bridge/package.json bridge/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund
COPY bridge/patch-baileys.mjs ./
RUN node patch-baileys.mjs

FROM node-dependencies AS node-verified
COPY bridge/*.mjs ./
COPY bridge/test ./test
RUN test -f test/publications.test.mjs && node --test test/*.test.mjs

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libatomic1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app app ./app
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app scripts ./scripts
COPY --from=node-dependencies /usr/local/bin/node /usr/local/bin/node
COPY --from=node-verified --chown=app:app /bridge/node_modules ./bridge/node_modules
COPY --chown=app:app bridge/*.mjs bridge/package*.json bridge/BAILEYS-LICENSE.txt ./bridge/
COPY --chown=app:app wsgi.py manage.py ./
RUN node --version && cd bridge && node --input-type=module -e "await import('baileys'); console.log('WhatsApp runtime import verified')"
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "scripts.run_container"]
