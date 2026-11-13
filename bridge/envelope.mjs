import { createHash } from 'node:crypto';

export function eventKind(message, forced = null) {
  return forced || (message?.editedMessage ? 'edit' : 'original');
}

export function envelope({group, id, sender, aliases = [], timestamp, body, kind = 'original', target = null, name = ''}) {
  if (!group?.endsWith('@g.us') || !id || id.length > 150) throw new Error('Invalid message key');
  const payload = {group_jid: group, message_id: id, sender, aliases: [...new Set(aliases.filter(Boolean))],
    timestamp, body: String(body || '').slice(0, 10000), truncated: String(body || '').length > 10000, kind, target_message_id: target, push_name: String(name).slice(0,100)};
  // Stable transport ID for originals and deletes; edit revisions retain body/time identity.
  const revision = kind === 'edit' ? `:${timestamp}:${payload.body}` : '';
  const eventKey = createHash('sha256').update(`${group}:${id}:${kind}${revision}`).digest('hex');
  return {eventKey, payload};
}
