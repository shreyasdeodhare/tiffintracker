import test from 'node:test';
import assert from 'node:assert/strict';
import {envelope,eventKind} from '../envelope.mjs';
const base={group:'123@g.us',id:'stable-id',sender:'919876543210@s.whatsapp.net',timestamp:1800000000,body:'2 tiffins'};
test('replayed originals retain the same event key',()=>assert.equal(envelope(base).eventKey,envelope({...base,name:'Different display name'}).eventKey));
test('edits and deletes are distinct durable events',()=>{
  assert.notEqual(envelope(base).eventKey,envelope({...base,kind:'edit'}).eventKey);
  assert.notEqual(envelope({...base,kind:'edit'}).eventKey,envelope({...base,kind:'edit',body:'3 tiffins'}).eventKey);
  assert.notEqual(envelope(base).eventKey,envelope({...base,kind:'delete'}).eventKey);
});
test('non-group messages are rejected',()=>assert.throws(()=>envelope({...base,group:'123@s.whatsapp.net'})));
test('Baileys editedMessage updates remain edits after content unwrap',()=>assert.equal(eventKind({editedMessage:{message:{conversation:'3 tiffins'}}}),'edit'));
