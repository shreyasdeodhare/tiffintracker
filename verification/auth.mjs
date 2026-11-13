import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Pool} from '/var/tmp/tiffintrack-final-acceptance-20261112/bridge/node_modules/pg/esm/index.mjs';
import {unseal} from '/var/tmp/tiffintrack-final-acceptance-20261112/bridge/session-crypto.mjs';
const secret=process.env.WA_SESSION_KEY;
assert.ok(secret && secret.length>=32);
const pool=new Pool({host:'/var/run/postgresql',user:'postgres',database:'tiffintrack_final_restore_test'});
try {
 const rows=await pool.query('SELECT encrypted_value FROM wa_auth WHERE key=$1',['creds']);
 assert.equal(rows.rows.length,1);
 const key=createHash('sha256').update('tiffintrack-session-v1:'+secret).digest();
 const value=unseal(rows.rows[0].encrypted_value,key);
 assert.ok(value && typeof value==='object');
 console.log('RESTORED_SESSION_AUTHENTICATED_DECRYPTION_VERIFIED_NO_WHATSAPP_CONNECTION');
} finally {await pool.end();}
