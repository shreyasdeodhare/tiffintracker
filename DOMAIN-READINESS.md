# Domain and HTTPS readiness — no changes executed

Public hostname publication remains deferred until private acceptance and owner approval. Confirm hostname ownership, DNS account access, the existing Cloudflare route/service and origin reachability. The SSH hostname does not establish permission to edit its DNS zone. Preserve private access and do not add another application container.

| Choice | DNS record | Purpose |
|---|---|---|
| Direct IPv4 | A to the approved public IPv4 | Resolves the hostname to a reachable HTTPS ingress |
| Direct IPv6 | AAAA to the approved public IPv6 | Requires working IPv6; omit an unusable record |
| Cloudflare Tunnel | CNAME to `<tunnel-uuid>.cfargotunnel.com` | Routes the approved hostname through the authorized tunnel |

Do not publish the private `192.168.1.101` address as a public-reachability solution or install conflicting records. A/AAAA map hostnames to IP addresses; tunnel CNAME records point to the tunnel subdomain. [DNS record reference](https://developers.cloudflare.com/dns/manage-dns-records/reference/dns-record-types/), [tunnel routing](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/routing-to-tunnel/dns/).

A host-managed tunnel can connect outbound to Cloudflare without additional inbound ports. This is an option for a separately approved phase, not a newly deployed service. [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/).

Prepare a separate Nginx hostname server block, trusted TLS and renewal. Retain upstream `127.0.0.1:8087`, actual Host including any forwarded port, overwritten forwarding headers, secure cookies and CSRF. Keep HTTP-to-HTTPS redirection on the intended ingress without loops. An HTTPS tunnel origin needs a trusted certificate and matching origin server name; do not disable certificate verification. [HTTPS origin guidance](https://developers.cloudflare.com/tunnel/troubleshooting/https-origins/).

After hostname/DNS approval, verify resolution, trusted TLS, redirects, actual browser login, cookies, CSRF, authorization, reports, Excel and continued private access. Do not expose PostgreSQL or bridge port 8090 through a tunnel.

Current host finding: PostgreSQL listens on all interfaces and UFW allows port 5432 globally. TiffinTrack HBA rules reject off-network database access, but the shared database service remains network-reachable. Inventory other applications and approved source networks before changing global firewall rules. No DNS, firewall, certificate or public route was changed during this audit.
