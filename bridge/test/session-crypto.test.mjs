import test from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes} from 'node:crypto';
import {seal,unseal} from '../session-crypto.mjs';

test('session state survives authenticated encryption round trip',()=>{
  const key=randomBytes(32), value={creds:{test:'synthetic-secret'},keys:[1,2,3]};
  const cipher=seal(value,key);
  assert.deepEqual(unseal(cipher,key),value);
  assert.ok(!cipher.includes('synthetic-secret'));
  assert.notEqual(cipher,seal(value,key));
});
test('tampered session ciphertext is rejected',()=>{
  const key=randomBytes(32), bytes=Buffer.from(seal({test:true},key),'base64');
  bytes[bytes.length-1]^=1;
  assert.throws(()=>unseal(bytes.toString('base64'),key));
});
test('the wrong session key cannot decrypt credentials',()=>{
  assert.throws(()=>unseal(seal({test:true},randomBytes(32)),randomBytes(32)));
});
