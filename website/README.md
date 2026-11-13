# TiffinTrack public website

This is an independent, buildless landing page. Only `public/` is served. It introduces TiffinTrack and contains a labelled browser-only example; it is not a replacement for the Flask application or its dashboard.

No backend imports, API calls, credentials, environment changes, database access, migrations, container changes, cron changes, dependency upgrades or production-server operations are needed. Existing application access stays separate. The public page deliberately does not publish a private IP, assume an application hostname, or pretend to offer public sign-in.

## Local review

From the repository root:

```text
python -m http.server 4173 --bind 127.0.0.1 --directory website/public
```

Visit `http://127.0.0.1:4173/`. Day/Week/Month changes illustrative figures only; FAQ items use native disclosure controls. Stop the local server after review. No package installation or build step is required.

## Recommended Cloudflare Pages settings

The screenshot shows **Workers** running `npx wrangler deploy`, not a successful Pages static deployment. Do not run backend Python or Docker builds on Pages. After approval to push and deploy, create/select a separate **Pages** project and use:

| Setting | Value |
|---|---|
| Repository | `shreyasdeodhare/tiffintracker` |
| Initial review branch | `feature/cloudflare-landing-page` after approved push |
| Root directory | `website` |
| Framework preset | None |
| Build command | `exit 0` |
| Build output directory | `public` |
| Build environment variables | None |
| Served directory | `website/public` only |

Do not choose the repository root as the output directory: source, historical evidence and backend deployment files must not become public assets. Set the production branch only after explicit merge/deployment approval. Preview deployments are public exposure too and require approval. No project or DNS setting has been changed by this work.

`wrangler.pages.jsonc` is an explicitly named, optional Pages configuration for operator use. Dashboard Git builds can use the table above without it. After approval, an operator using a compatible installed Wrangler can deploy from `website` with:

```text
wrangler pages deploy public --project-name tiffintracker-website --config wrangler.pages.jsonc
```

Cloudflare may prompt to select/create that separate Pages project. Confirm the intended account/project before proceeding. Do not run the command until publication is approved. There is no npm/package lock or new dependency added to the existing repository runtime.

## Alternative for the existing Workers screen

To retain the Workers project visible in the screenshot instead of using Pages, use the separate `wrangler.workers.jsonc`. It declares `assets.directory = "./public"`, addressing the missing-static-directory error. It has no script proxy, backend bindings, secrets, database, routes or cron triggers.

After separate deployment approval, set the Workers project root to `website` and the deploy command to:

```text
npx wrangler@4.61.0 deploy --config wrangler.workers.jsonc
```

The pinned example matches the Wrangler version shown in the supplied screenshot. It is an operator command, not an installed/upgraded backend dependency. The config name `tiffintracker` matches that screenshot's Worker; verify the target before authorizing deployment. Use either the Pages route or the Workers route intentionally, not both against the same hostname. Neither moves the Flask application into Cloudflare Workers.

The two configurations use non-default filenames so an unrelated root/backend build cannot discover them automatically and so Pages does not accidentally load the Workers assets configuration. Commands select their intended configuration explicitly.

## DNS and HTTPS after approval

1. Confirm ownership and the desired public website hostname. No owned hostname has been supplied for this change; none is invented or changed.
2. Verify the approved `pages.dev` deployment first. In the Pages project's **Custom domains** area, associate the hostname before adding a manual DNS record.
3. For a subdomain, follow Cloudflare's required CNAME from the chosen hostname to the approved `<project>.pages.dev` target. A zone-apex setup has additional Cloudflare nameserver requirements; review the existing DNS zone and preserve mail and other services.
4. Wait for certificate issuance and verify HTTPS, redirects, assets, security headers, 404 handling, mobile layout and the unchanged private app access. Do not bypass a certificate error.
5. Do not create DNS records to the private support-server IP, proxy the public website to PostgreSQL, publish port 8087/8090, alter the existing SSH route, or expose the dashboard as part of this static-site change.

The public homepage and any future public application hostname are separate decisions. Making the dashboard public would require its own reviewed access and TLS plan, beyond this branch.

## Isolation, review and rollback

All additions are inside `website/`. All 160 pre-existing tracked files are verified byte-for-byte against `preservation-baseline.json`; this includes app entry points, routes, Docker configuration, migrations, connector, workers, cron templates, backup scripts, secrets example and existing manifests. Existing checksum files are not edited to hide differences. The 29 baseline HTTP route declarations are recorded as well.

Run `python website/check_preservation.py` from this branch to recheck the existing files and source routes. Run the existing Python and bridge tests separately; `VERIFICATION.md` records the executed baseline and after results and the exact local limitations.

The feature branch can be reviewed against `204a8d21effbe9083b8290c39e9feb23da7d5730`. Review should show only `website/` additions. No push or merge is authorized by this document. After an approved future release, rollback is limited to the static project's previous deployment or the website-only commit; it needs no database restore, migration, application-image rollback or server restart. Do not delete pre-existing backend files or disable backend scheduled jobs.

## Official configuration references

- [Pages static HTML](https://developers.cloudflare.com/pages/framework-guides/deploy-anything/)
- [Pages build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [Pages Wrangler commands](https://developers.cloudflare.com/workers/wrangler/commands/pages/)
- [Workers static assets configuration](https://developers.cloudflare.com/workers/static-assets/binding/)
- [Pages custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/)

These are preparation instructions, not evidence of an executed Cloudflare deployment.
