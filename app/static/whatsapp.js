// Same-origin admin-only endpoints; QR credentials are never logged or cached.
async function refreshConnection() {
  try {
    const response = await fetch('/admin/whatsapp/status.json', {cache: 'no-store'});
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) return;
    const status = await response.json();
    document.getElementById('wa-status').textContent = status.connection + ' · processor: ' + status.worker + ' · waiting: ' + status.pending + ' · errors: ' + status.errors;
    if (status.last_disconnect) document.getElementById('wa-status').textContent += ' · WhatsApp code: ' + status.last_disconnect;
    const relink = document.getElementById('wa-relink');
    if (relink) relink.hidden = status.connection !== 'logged_out';
    const image = document.getElementById('wa-qr');
    if (image) {
      image.hidden = !status.qr;
      if (status.qr && image.dataset.generation !== String(status.qr_generation)) {
        image.src = '/admin/whatsapp/qr.png?t=' + Date.now();
        image.dataset.generation = String(status.qr_generation);
      }
    }
  } catch (_) { document.getElementById('wa-status').textContent = 'Connection status unavailable'; }
}
setInterval(refreshConnection, 10000);
