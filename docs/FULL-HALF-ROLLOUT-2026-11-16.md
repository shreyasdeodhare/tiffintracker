# Full/half pricing and multi-line order rollout - 16 November 2026

Status: deployed and healthy as tiffintrack:portions-20261116. Source changes were prepared on feature/full-half-multiline-20261116. Production deployment and GitHub source publication are separate operations; publication follows explicit user authorization.

## Confirmed business rules

| Message | Tiffin quantity | Amount |
|---|---:|---:|
| 1 full tiffin | 1 | INR 80 |
| 1 half tiffin | 1 | INR 60 |
| 1 full tiffin option A + 1 full tiffin option B | 2 | INR 160 |
| 2 full tiffin / 1 rajma / 1 methi aloo (separate lines) | 2 | INR 160 |
| 1 full tiffin + 1 half tiffin | 2 | INR 140 |
| 2 full tiffins and 3 half tiffins | 5 | INR 340 |

Dish descriptions explain the declared meals; they do not add extra billable tiffins. Supported option labels are A/B or 1/2. Recognized dish-allocation descriptions in this rollout are rajma and methi aloo, with declared quantities that must reconcile to the explicit tiffin total. Other, conditional, conflicting, oversized or multi-person wording remains in administrator review; the parser does not guess.

The original plain single-line order behavior and configurable standard price remain intact. Explicit full and half prices default to 80.00 and 60.00. The existing protected production environment file was not edited. Historical stored prices are retained, never recomputed.

## Representation and lifecycle

No schema migration or model-column change was needed. Quantities at the same portion/meal price are grouped into existing Order rows. Mixed full/half requests use separate price rows, with per-line options, portion and descriptions in existing JSON audit records. Reports distinguish order records from tiffin quantity; the requested count is the tiffin quantity. All amount calculations use the existing stored Decimal unit prices and independent reconciliation.

Exact source-message edits replace only the corresponding price groups. Removed groups are cancelled with audit history; newly added groups are linked to the original source for deduplication and reconciliation. Same-portion changes preserve their stored historical price. A deletion or unique-source cancellation cancels all groups from that source. Generic quantity changes across mixed portions remain in review because their allocation is unknown. Ownership checks, finalized-period checks and out-of-order protections remain in force.

The admin order form has full/half selection and a legacy standard option for existing/historical manual entry. The review form can explicitly confirm a portion for unclear wording. Typed automatic/itemized pricing starts with order dates on or after 16 November 2026; older confirmed orders and saved publications were not repriced or rewritten.

## Today-only re-check

The operator command defaults to dry-run:

    python -m scripts.reprocess_today_meals
    python -m scripts.reprocess_today_meals --apply

It locks the selected connection and reuses existing unresolved events dated today in the application timezone, in source order. It never inserts duplicate events or resubmits already-applied orders. Unsupported wording stays under review. The approved live run on 16 November considered 0 unresolved events dated today, so it applied 0 changes. Older reviews remain untouched.

## Verification and production preservation

- 165 Python tests passed on disposable PostgreSQL databases, including 19 new lifecycle tests on PostgreSQL as well as SQLite, parser/pricing tests and existing migration, identity, reconciliation, authentication and reporting regressions.
- 24 unchanged Node/WhatsApp bridge tests passed.
- Ruff lint and format checks passed.
- Production installed-code checks match the tested file manifest.
- New full/half selector, admin pages and Excel exports were checked read-only after deployment; no synthetic production orders or WhatsApp messages were created for verification.
- The live image verified both user-confirmed examples and the configured 80/60 prices.
- Existing schema 0005_participants, historical orders, saved publication hashes, protected .env, Nginx sites and cron hashes were preserved.
- The existing linked WhatsApp session reconnected with fresh receiver and worker heartbeats. The Friday-only 14:00 IST Saturday-Friday summary schedule is unchanged.
- One existing application container remains: 512 MiB, 1.5 CPUs, 128 PIDs, read-only root and host binding 127.0.0.1:8087 -> 8000. No additional application container or service was introduced.

A derived image was built offline from the prior production image, copying only the tested Python/template changes. No dependency upgrade was performed. The base Dockerfile and docker-compose.yml remain unchanged; docker-compose.override.yml selects the new release image only and retains all existing settings.

## Deployment and rollback record

The first activation passed container, endpoint and reconnect checks but an incorrectly escaped verification command failed. The controlled deployment restored and verified the previous image and code. The verification command was corrected and syntax-checked; the second activation passed all gates, followed by the approved today-only review re-check. No production schema migration or database restore was performed.

The previous stabilization image and protected source/database backups remain available on the host. Rollback uses the retained prior image and source, removes only the new image override after verifying ownership, and rechecks application/WhatsApp health. Existing data is preserved; no automatic database rollback or publication replay is performed.

A genuine new group order still needs to be observed in normal use for live business acceptance. The isolated tests and live code/proxy checks are not represented as a synthetic live WhatsApp send.

## Files and delivery

New modules: domain/meal_items.py (strict item grammar and prices), application/meal_orders.py (price-group lifecycle), infrastructure/meal_orders.py (persistence queries), application/reprocess_meals.py (today-only re-check), scripts/reprocess_today_meals.py (operator CLI), tests/test_meal_items.py (regressions). Existing parser, order use cases, HTTP adapters and admin templates were extended without changing the database schema or existing route names.

PORTION-RELEASE-MANIFEST.json lists the tested runtime/test file hashes. The dated older PDFs and source manifests remain historical references and do not cover this new feature. Use this guide and the current release manifest for this rollout.
