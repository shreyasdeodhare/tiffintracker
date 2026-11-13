# Access TiffinTrack

Use the current application password. No permanent localhost forward is guaranteed. Preserve the linked WhatsApp session. See FINAL-ACCEPTANCE.md for checks and pending browser authentication acceptance. The private LAN link is https://192.168.1.101/admin/orders.

On the server LAN: https://192.168.1.101
On the server VPN: https://100.72.251.13

These are private addresses; the deployment workstation could not reach them directly. From another network, use the same authorized Cloudflare SSH route with local forwarding:

```powershell
ssh -L 8443:127.0.0.1:443 -o 'ProxyCommand=cloudflared access ssh --hostname ssh.thesourcecode.in' support@ssh.thesourcecode.in
```

Keep SSH open and visit https://127.0.0.1:8443 . Accept the temporary self-signed certificate warning only for this server (or install its certificate after verifying it through your trusted SSH connection). The temporary certificate is valid for 90 days; replace it with trusted TLS when the domain is configured.

Administrator: shreyas.deodhare@redberyltech.com

The protected `/home/support/tiffintrack-admin-access.txt` is an initial provisioning record and may be stale after a password change. Use current application credentials; do not reset them for an acceptance checklist. SSH and application passwords are separate. No credentials are included in this archive.


## Automatic WhatsApp tracking

After login open `/admin/whatsapp` to inspect the existing connected account and selected group. Ordinary group membership is sufficient; group ownership is unnecessary. Clear messages update tracking automatically; ambiguous requests require review. No guest login/email account is created. Do not relink a healthy session. Historical setup instructions remain in WHATSAPP.md.

Use the SSH command above when a temporary forward is required. Nginx preserves the tunnel port in the Host header. The temporary certificate warning remains expected after verification of its identity; the certificate expires on 9 February 2027 at 23:36:57 IST. Current browser password authentication was not reverified during this audit.
