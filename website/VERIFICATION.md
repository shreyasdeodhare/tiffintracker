# Landing page preservation and verification

Status: implemented and reviewed locally on `feature/cloudflare-landing-page`. No push, merge, Cloudflare deployment, DNS change or production-server action has been performed.

## Baseline and scope

Original checkout: clean `publish-current` at `204a8d21effbe9083b8290c39e9feb23da7d5730`. A separate Git worktree and feature branch were created from that commit. The baseline records all 160 pre-existing tracked files and 29 HTTP route declarations. Existing source was inspected before the new website files were created, and the existing test suite completed before implementation.

All additions are confined to `website/`. No existing application file, backend configuration, environment file or credential was edited. All 160 existing tracked files remain byte-for-byte identical to the baseline, including the production-code and complete-source manifests themselves. This verifies unchanged entry points, auth/routes, order/participant/review logic, PostgreSQL models/migrations, WhatsApp/Baileys receiver, worker, Docker image definition/Compose ports/limits, backup and scheduled-job templates.

Runtime Flask URL maps independently loaded from the unchanged baseline checkout and the feature checkout matched: **30 registered rules**. The app factories used an isolated in-memory configuration; this inspection did not connect to PostgreSQL, create tables or run migrations.

## Existing regression results

| Check | Before | After | Comparison |
|---|---|---|---|
| Python suite | 88 passed; 8 skipped; 0 failures | 88 passed; 8 skipped; 0 failures | Identical result for every collected test |
| Bridge suite | 24 passed; 0 failures | 24 passed; 0 failures | Identical passing count |
| Original tracked-file hashes | 160 captured | 160 matched | No changed or missing file |
| HTTP declarations | 29 captured | 29 matched | No route-source change |
| Runtime URL maps | 30 rules from baseline code | 30 matching feature rules | Identical patterns, methods and endpoints |

Python command in each phase:

```text
python -B -m pytest -q -p no:cacheprovider --basetemp=<separate-local-test-directory> --junitxml=<separate-local-report>
```

The raw local JUnit files are retained outside the repository; `review/test-comparison.json` records sanitized per-test outcomes. Before time: 2039.59 seconds; after time: 99.38 seconds. One before-run test (`test_finalized_week_blocks_automatic_changes`) consumed 1916.583 seconds versus 1.269 seconds afterward. No backend code was edited to address this isolated timing outlier, and no application performance improvement is claimed from this comparison.

`npm` is unavailable in this execution environment. The equivalent existing package test script was run directly with Node in `bridge` in both phases:

```text
node --test test/*.test.mjs
```

It passed all 24 tests, including the established envelope, crypto, pairing, group-discovery and publication-state tests. Existing tests were not removed, edited, weakened or selectively filtered.

### Tests not executed

The same eight tests skipped in both runs because local `TEST_POSTGRES_URL`/`TEST_MIGRATION_URL` are not configured:

- `test_forward_migration_preserves_accounts_prices_orders_and_auth`
- `test_concurrent_preparation_claims_one_snapshot`
- `test_connection_lock_refreshes_cached_configuration`
- `test_concurrent_form_submissions`
- `test_real_postgres_migrations_and_constraints`
- `test_postgres_mail_batch_commit_and_failure_isolation`
- `test_export_snapshot_during_concurrent_commit`
- `test_postgres_whatsapp_durable_concurrency_and_rollback`

The separate PostgreSQL outbox helper was not run because there is no configured disposable local PostgreSQL test database/dependency installation. No production database or support server was used as a substitute. Earlier production/audit test reports remain historical evidence and are not counted as newly executed tests here.

Cloudflare deployment, custom-domain DNS/TLS, remote Wrangler validation, live backend login, live WhatsApp messages and production runtime health were not retested or changed in this website-only task. JSON configuration and asset isolation were checked locally; a real Cloudflare deployment remains subject to approval. `_headers` enforcement and Cloudflare missing-route behavior require checking on the eventual approved preview: the local Python static server does not enforce Cloudflare response-header/fallback rules.

## Landing-page verification

- Home page, CSS, JavaScript, SVG, robots file and explicit 404 page returned local HTTP 200.
- Every HTML asset reference exists and every fragment link has a matching target. Public output is restricted to the seven files in `public/`.
- JavaScript passed `node --check`. The new preservation helper passed Ruff checks/formatting and its own read-only comparison.
- Browser inspected at desktop 1280 pixels and responsive widths 320, 360 and 768. Measured document widths stayed within the viewport; no horizontal overflow was observed.
- Day example: 3 order records, 4 tiffins, ₹320. Week: 5 records, 9 tiffins, ₹720. Month: 21 records, 35 tiffins, ₹2,800. Values are explicitly illustrative and no application orders are read or written.
- Native FAQ disclosure expanded correctly. Dashboard-access navigation leads to the access explanation, not a guessed public/private backend address.
- The public page contains no form or external link, and its script contains no fetch, authentication, storage or backend integration. Sample rendering uses `textContent`.
- Captured browser warning/error log was empty during the inspected interaction sequence. Header policy declares no outgoing connections, forms, embedded frames or objects; no cookies, analytics or third-party assets are added.
- Desktop and mobile screenshots were saved as review artifacts outside the repository. A temporary local static preview was used only for QA; it is not a deployment.

## Every changed file

All paths below are additions. There are no modifications/deletions outside `website/`.

| File within website | Purpose |
|---|---|
| `public/index.html` | Standalone accessible landing page and clearly labelled illustrative report |
| `public/assets/styles.css` | Independent responsive visual styling and focus/reduced-motion support |
| `public/assets/site.js` | Browser-only sample period controls; no backend requests or persistent state |
| `public/assets/logo.svg` | Local original tiffin-stack brand icon |
| `public/404.html` | Separate public-site missing-page document and access explanation |
| `public/_headers` | Static-site security headers; unrelated to backend headers |
| `public/robots.txt` | Crawler rules for this public introduction only |
| `wrangler.pages.jsonc` | Explicit optional Pages output configuration |
| `wrangler.workers.jsonc` | Explicit alternative Workers static-assets configuration for the screenshot's error |
| `README.md` | Local review, exact Pages/Workers settings, DNS/HTTPS approval and isolated rollback instructions |
| `preservation-baseline.json` | Original commit, complete existing-file hashes and route declarations |
| `check_preservation.py` | Read-only reproduction of original-file and route-source invariants |
| `review/test-comparison.json` | Sanitized baseline/after test outcomes and runtime-route comparison |
| `VERIFICATION.md` | This scope, changed-file register, evidence and verification limits |

## Approval and rollback boundary

Only a local feature-branch change has been prepared. No GitHub write, automatic Cloudflare build trigger or production merge has occurred. Review the branch against the baseline commit before separately approving a push. Check connected Cloudflare branch deployment settings before any approved push, since a push could otherwise trigger public preview hosting. Public deployment and DNS changes require their own explicit approval.

Rollback of a future approved landing-page deployment is limited to reverting the website-only commit or selecting the static project's previous deployment. There is no backend image change, server restart, database migration or data restore to reverse.
