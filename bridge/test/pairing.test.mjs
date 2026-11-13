import test from 'node:test';
import assert from 'node:assert/strict';
import {currentSecretQR,refreshRegistration,newerVersion} from '../pairing.mjs';
import {patchAcknowledgement} from '../patch-baileys.mjs';

const notification=tag=>({tag:'notification',attrs:{type:'companion_reg_refresh'},content:[{tag,attrs:{}}]});
for (const tag of ['companion_reg_refresh','pair-device-rotate-qr']) {
  test(`unpaired registration refresh handles ${tag} without spending a reference`,()=>{
    const creds={advSecretKey:'old-secret'};
    const qr='https://wa.me/settings/linked_devices#current-ref,noise,identity,old-secret,1';
    let rendered,saved;
    assert.equal(refreshRegistration(notification(tag),{creds,save:update=>saved=update,render:()=>rendered=currentSecretQR(qr,creds.advSecretKey)}),'rotated');
    assert.equal(Buffer.from(creds.advSecretKey,'base64').length,32);
    assert.equal(saved.advSecretKey,creds.advSecretKey);
    assert.equal(rendered.split(',')[0],qr.split(',')[0]);
    assert.equal(rendered.split(',')[3],creds.advSecretKey);
    assert.equal(currentSecretQR(qr,creds.advSecretKey),rendered);
  });
}
test('refresh never rotates credentials after an identity has been paired',()=>{
  const creds={me:{id:'synthetic-user'},advSecretKey:'existing-secret'};
  const fail=()=>assert.fail('registered credentials were modified');
  assert.equal(refreshRegistration(notification('companion_reg_refresh'),{creds,save:fail,render:fail}),'ignored_registered');
  assert.equal(creds.advSecretKey,'existing-secret');
});
test('malformed refresh notification does not change pairing material',()=>{
  const fail=()=>assert.fail('malformed notification was applied');
  assert.equal(refreshRegistration(notification('unexpected'),{creds:{},save:fail,render:fail}),'ignored_malformed');
});
test('WhatsApp version discovery never falls back below the known good revision',()=>{
  assert.deepEqual(newerVersion([2,3000,9999],[2,3000,999]),[2,3000,9999]);
  assert.deepEqual(newerVersion([2,3000,999],[2,3000,9999]),[2,3000,9999]);
  assert.deepEqual(newerVersion([2,3000,9999],undefined),[2,3000,9999]);
});
test('dependency patch is idempotent and refuses an unexpected upstream source',()=>{
  const source='const stanza = buildAckStanza(node, errorCode, authState.creds.me.id);';
  const changed=patchAcknowledgement(source);
  assert.equal(patchAcknowledgement(changed),changed);
  assert.throws(()=>patchAcknowledgement('unrecognized source'));
});
test('notification ack works before pairing, while message ack still requires identity',async()=>{
  const fragment=patchAcknowledgement('const stanza = buildAckStanza(node, errorCode, authState.creds.me.id);');
  const make=new Function('buildAckStanza','authState','Boom',`return async (node,errorCode)=>{${fragment};return stanza}`);
  const build=(node,code,id)=>({tag:node.tag,id});
  const unpaired=make(build,{creds:{}},Error);
  assert.deepEqual(await unpaired({tag:'notification'}),{tag:'notification',id:undefined});
  await assert.rejects(unpaired({tag:'message'}));
  const paired=make(build,{creds:{me:{id:'synthetic'}}},Error);
  assert.equal((await paired({tag:'message'})).id,'synthetic');
});
