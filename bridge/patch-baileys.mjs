// Small, version-checked upstream compatibility fix; no auth bypass or secret logging.
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const needle='const stanza = buildAckStanza(node, errorCode, authState.creds.me.id);';
const replacement=`// tiffintrack-prelogin-notification-ack: upstream PR #2765
        const meId = authState.creds.me?.id;
        if (node.tag === 'message' && !meId) {
            throw new Boom('cannot ack a message before creds.me is set');
        }
        const stanza = buildAckStanza(node, errorCode, meId);`;

export function patchAcknowledgement(source) {
  if(source.includes('tiffintrack-prelogin-notification-ack')) return source;
  if(source.split(needle).length!==2) throw new Error('Baileys source changed; review the pairing patch');
  return source.replace(needle,replacement);
}

if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const directory=path.dirname(fileURLToPath(import.meta.url));
  const pkg=JSON.parse(readFileSync(path.join(directory,'node_modules/baileys/package.json'),'utf8'));
  if(pkg.version!=='7.0.0-rc14') throw new Error('Pairing patch is scoped to Baileys 7.0.0-rc14');
  const target=path.join(directory,'node_modules/baileys/lib/Socket/messages-recv.js');
  writeFileSync(target,patchAcknowledgement(readFileSync(target,'utf8')));
  process.stdout.write('Version-checked WhatsApp pairing acknowledgement fix applied\n');
}
