import {createCipheriv,createDecipheriv,randomBytes} from 'node:crypto';

export function seal(value,key,replacer) {
  const nonce=randomBytes(12);
  const cipher=createCipheriv('aes-256-gcm',key,nonce);
  const bytes=Buffer.concat([cipher.update(JSON.stringify(value,replacer)),cipher.final()]);
  return Buffer.concat([nonce,cipher.getAuthTag(),bytes]).toString('base64');
}

export function unseal(value,key,reviver) {
  const bytes=Buffer.from(value,'base64');
  const decipher=createDecipheriv('aes-256-gcm',key,bytes.subarray(0,12));
  decipher.setAuthTag(bytes.subarray(12,28));
  return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)),decipher.final()]).toString(),reviver);
}
