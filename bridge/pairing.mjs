// Protocol handling follows the reviewed scope of upstream Baileys PR #2765.
// https://github.com/WhiskeySockets/Baileys/pull/2765 (MIT; see BAILEYS-LICENSE.txt).
import {randomBytes} from 'node:crypto';

export function currentSecretQR(qr, secret) {
  const fields=qr.split(',');
  if (fields.length < 4 || !secret) throw new Error('Unexpected pairing QR format');
  fields[3]=secret;
  return fields.join(',');
}

export function refreshRegistration(node, {creds, save, render}) {
  const valid=node?.tag==='notification' && node.attrs?.type==='companion_reg_refresh' &&
    Array.isArray(node.content) && node.content.some(child=>['companion_reg_refresh','pair-device-rotate-qr'].includes(child.tag));
  if (!valid) return 'ignored_malformed';
  if (creds.me) return 'ignored_registered';
  creds.advSecretKey=randomBytes(32).toString('base64');
  save({advSecretKey:creds.advSecretKey});
  render(); // Re-render the current ref, without consuming another ref or resetting its timer.
  return 'rotated';
}

export function newerVersion(known, candidate) {
  const valid=value=>Array.isArray(value) && value.length===3 && value.every(x=>Number.isSafeInteger(x) && x>=0);
  if (!valid(candidate)) return known;
  if (!valid(known)) return candidate;
  for(let i=0;i<3;i++) {
    if(candidate[i]>known[i]) return candidate;
    if(candidate[i]<known[i]) return known;
  }
  return known;
}
