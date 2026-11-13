import { createServer } from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import { Pool } from 'pg';
import pino from 'pino';
import QRCode from 'qrcode';
import { envelope, eventKind } from './envelope.mjs';
import {seal,unseal} from './session-crypto.mjs';
import {currentSecretQR,refreshRegistration,newerVersion} from './pairing.mjs';
import {createGroupDiscovery} from './group-discovery.mjs';
import {createSummaryPublisher} from './publications.mjs';

// Third-party protocol code must never print Signal sessions, messages or pairing codes.
for (const method of ['log','debug','info','warn','error','trace']) console[method] = () => {};
const {default: makeWASocket, BufferJSON, initAuthCreds, proto, DisconnectReason,
  makeCacheableSignalKeyStore, normalizeMessageContent, jidNormalizedUser,
  fetchLatestWaWebVersion, DEFAULT_CONNECTION_CONFIG, Browsers} = await import('baileys');
const token = process.env.WA_BRIDGE_TOKEN;
const sessionSecret = process.env.WA_SESSION_KEY;
if (!token || token.length < 32 || !sessionSecret || sessionSecret.length < 32) {
  process.stderr.write('WhatsApp connector secrets are missing\n'); process.exit(1);
}
const encryptionKey = createHash('sha256').update('tiffintrack-session-v1:' + sessionSecret).digest();
const pool = new Pool({connectionString: process.env.DATABASE_URL.replace('postgresql+psycopg:', 'postgresql:'), max: 3, connectionTimeoutMillis: 10000});
const logger = pino({level:'silent'});
let socket, qrValue = null, qrTime = 0, connection = 'starting', groups = [], targetGroup = null;
let reconnectTimer, closing = false, failures = 0, serial = Promise.resolve();
let webVersion=DEFAULT_CONNECTION_CONFIG.version, lastDisconnect=null, qrGeneration=0;
let targetName=null, groupDiscoveryError=null;
const discoverGroups=createGroupDiscovery({
  getSelected:()=>targetGroup?{id:targetGroup,name:targetName || 'Selected order group'}:null,
  fetchGroups:async()=>{
    const result=await socket.groupFetchAllParticipating();
    return Object.values(result).map(item=>({id:item.id,name:item.subject}));
  },
  onGroups:value=>{groups=value;groupDiscoveryError=null;},
  onError:error=>{
    const code=Number.isInteger(error?.output?.statusCode)?error.output.statusCode:0;
    groupDiscoveryError=`Group list temporarily unavailable (code ${code})`;
    process.stderr.write(`WhatsApp optional group discovery failed status=${code}; receiver remains active\n`);
  }
});

function encrypt(value) {
  return seal(value,encryptionKey,BufferJSON.replacer);
}
function decrypt(value) {
  return unseal(value,encryptionKey,BufferJSON.reviver);
}
async function readAuth(key) {
  const result=await pool.query('SELECT encrypted_value FROM wa_auth WHERE key=$1',[key]);
  return result.rows.length ? decrypt(result.rows[0].encrypted_value) : null;
}
async function writeAuth(key,value,client=pool) {
  if (value == null) return client.query('DELETE FROM wa_auth WHERE key=$1',[key]);
  return client.query('INSERT INTO wa_auth (key,encrypted_value) VALUES ($1,$2) ON CONFLICT (key) DO UPDATE SET encrypted_value=EXCLUDED.encrypted_value',[key,encrypt(value)]);
}
async function authState() {
  const creds=await readAuth('creds') || initAuthCreds();
  return {creds, keys: {
    get: async (type,ids) => {
      const result={};
      for (const id of ids) {
        let value=await readAuth(`${type}:${id}`);
        if (type==='app-state-sync-key' && value) value=proto.Message.AppStateSyncKeyData.fromObject(value);
        result[id]=value;
      }
      return result;
    },
    set: async data => {
      const client=await pool.connect();
      try {
        await client.query('BEGIN');
        for (const [type,values] of Object.entries(data)) for (const [id,value] of Object.entries(values)) await writeAuth(`${type}:${id}`,value,client);
        await client.query('COMMIT');
      } catch (error) { await client.query('ROLLBACK'); throw error; }
      finally {client.release();}
    }
  }, save: () => writeAuth('creds',creds)};
}
function enqueue(work, operation='event') {
  serial=serial.then(work).catch(error => {
    connection='storage_error';
    const name=String(error?.name || 'Error').replace(/[^A-Za-z_]/g,'').slice(0,60);
    const code=/^[A-Z0-9_]{1,32}$/.test(String(error?.code))?error.code:'none';
    const status=Number.isInteger(error?.output?.statusCode)?error.output.statusCode:0;
    const frames=String(error?.stack || '').split('\n').slice(1,4).join('\n');
    process.stderr.write(`WhatsApp task failed operation=${operation} type=${name} code=${code} status=${status}\n${frames}\n`);
    process.exit(1);
  });
}
function seconds(value) {
  if (value?.toNumber) return value.toNumber();
  return Number(value || Math.floor(Date.now()/1000));
}
function contentText(content) {
  return content?.conversation || content?.extendedTextMessage?.text || content?.imageMessage?.caption || content?.videoMessage?.caption || content?.documentMessage?.caption || '[media: non-text content]';
}
async function persist(message, forceKind=null) {
  const key=message.key || {};
  if (!targetGroup || key.remoteJid !== targetGroup || !key.id) return;
  // Published summaries are not incoming business orders or review messages.
  if (key.fromMe) {
    const own=await pool.query('SELECT id FROM summary_parts WHERE message_id=$1',[key.id]);
    if (own.rows.length) return;
  }
  const content=normalizeMessageContent(message.message) || {};
  const protocol=content.protocolMessage;
  let kind=eventKind(message.message, forceKind), id=key.id, body=contentText(content), target=content.extendedTextMessage?.contextInfo?.stanzaId || null;
  let stamp=seconds(message.messageTimestamp);
  if (protocol) {
    if (protocol.type===proto.Message.ProtocolMessage.Type.REVOKE) {kind='delete'; id=protocol.key?.id; body='';}
    else if (protocol.type===proto.Message.ProtocolMessage.Type.MESSAGE_EDIT) {kind='edit'; id=protocol.key?.id; body=contentText(normalizeMessageContent(protocol.editedMessage));}
    else return;
    if (protocol.timestampMs) stamp=Math.floor(seconds(protocol.timestampMs)/1000);
  }
  const rawSender=key.participant || (key.fromMe ? socket.user?.id : 'unknown');
  const sender=jidNormalizedUser(rawSender || 'unknown');
  const aliases=[];
  if (key.participantAlt) aliases.push(jidNormalizedUser(key.participantAlt));
  if (sender.endsWith('@lid')) {
    const phone=await socket.signalRepository.lidMapping.getPNForLID(sender);
    if (phone) aliases.push(jidNormalizedUser(phone));
  }
  const value=envelope({group:targetGroup,id,sender,aliases,timestamp:stamp,body,kind,target,name:message.pushName || ''});
  const client=await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query('INSERT INTO wa_inbox (event_key,payload) VALUES ($1,$2) ON CONFLICT (event_key) DO UPDATE SET deliveries=wa_inbox.deliveries+1',[value.eventKey,JSON.stringify(value.payload)]);
    await client.query('UPDATE wa_connection SET last_received=now() WHERE id=1');
    await client.query('COMMIT');
  } catch (error) {await client.query('ROLLBACK'); throw error;}
  finally {client.release();}
}
async function refresh() {
  const result=await pool.query('SELECT group_jid,group_name FROM wa_connection WHERE id=1');
  targetGroup=result.rows[0]?.group_jid || null;
  targetName=result.rows[0]?.group_name || null;
  await pool.query('UPDATE wa_connection SET status=$1,heartbeat=now(),error=$2 WHERE id=1',[connection,lastDisconnect?`WhatsApp disconnect code ${lastDisconnect}`:null]);
}
async function loadGroups() {
  if (connection!=='connected') return;
  await discoverGroups();
}
async function start() {
  const auth=await authState();
  // Keep a successfully discovered revision across reconnects; never regress to an older fallback.
  try {
    const latest=await fetchLatestWaWebVersion({signal:AbortSignal.timeout(10000)});
    if(latest.isLatest) webVersion=newerVersion(webVersion,latest.version);
  } catch (_) { /* Preserve the last known revision if discovery is temporarily unavailable. */ }
  const current=makeWASocket({auth:{creds:auth.creds,keys:makeCacheableSignalKeyStore(auth.keys,logger)},logger,
    version:webVersion,browser:Browsers.ubuntu('Chrome'),
    markOnlineOnConnect:false, syncFullHistory:false, shouldSyncHistoryMessage:()=>false,
    // Keep own-device protocol traffic available for Signal/app-state key synchronization.
    // Personal chats are never enqueued, stored, classified or displayed by this app.
    shouldIgnoreJid:jid=>jid.endsWith('@g.us') && (!targetGroup || jid!==targetGroup), generateHighQualityLinkPreview:false});
  socket=current;
  current.ws.on('CB:notification,type:companion_reg_refresh',node=>{
    if(current!==socket) return;
    refreshRegistration(node,{creds:auth.creds,
      save:update=>current.ev.emit('creds.update',update),
      render:()=>enqueue(async()=>{
        if(qrValue && current===socket) current.ev.emit('connection.update',{qr:currentSecretQR(qrValue,auth.creds.advSecretKey)});
      })});
  });
  current.ev.on('creds.update',()=>enqueue(auth.save,'save_auth'));
  current.ev.on('connection.update',update=>{
    if(current!==socket) return;
    if (update.qr) {qrValue=currentSecretQR(update.qr,auth.creds.advSecretKey);qrGeneration++; qrTime=Date.now(); connection='awaiting_phone_link';}
    if (update.connection==='open') {
      connection='connected'; qrValue=null; failures=0;lastDisconnect=null;
      enqueue(refresh,'connection_ready');
      void loadGroups();
    }
    if (update.connection==='close' && !closing) {
      qrValue=null;
      const code=update.lastDisconnect?.error?.output?.statusCode;
      lastDisconnect=Number.isInteger(code)?code:null;
      process.stderr.write(`WhatsApp connection closed code=${lastDisconnect || 0} paired_identity=${!!auth.creds.me}\n`);
      if (code===DisconnectReason.loggedOut) {connection='logged_out'; groups=[];}
      else {
        connection='reconnecting';
        clearTimeout(reconnectTimer);
        // The restart following pairing must wait for queued encrypted credential writes.
        enqueue(async()=>{
          await refresh();
          reconnectTimer=setTimeout(()=>start().catch(()=>process.exit(1)),code===DisconnectReason.restartRequired?250:Math.min(60000,1000*2**Math.min(++failures,6)));
        });
      }
      enqueue(refresh);
    }
  });
  socket.ev.on('messages.upsert',({messages})=>{
    // Includes offline append deliveries; old timestamps are excluded by the Python start boundary.
    for (const message of messages) if (message.key?.remoteJid===targetGroup) enqueue(()=>persist(message));
  });
  socket.ev.on('messages.update',updates=>{
    for (const {key,update} of updates) {
      if (update.message) enqueue(()=>persist({key,...update}));
      else if (update.messageStubType===proto.WebMessageInfo.StubType.REVOKE) enqueue(()=>persist({key,...update},'delete'));
    }
  });
  socket.ev.on('messages.delete',updates=>{
    if (updates.keys) for (const key of updates.keys) enqueue(()=>persist({key,message:{},messageTimestamp:Math.floor(Date.now()/1000)},'delete'));
  });
}

const server=createServer(async(req,res)=>{
  res.setHeader('Cache-Control','no-store');
  const actual=Buffer.from(req.headers.authorization || ''), expected=Buffer.from('Bearer '+token);
  if (actual.length!==expected.length || !timingSafeEqual(actual,expected)) {res.writeHead(403); return res.end();}
  if (req.method==='POST' && req.url==='/relink') {
    enqueue(async()=>{
      closing=true;clearTimeout(reconnectTimer);
      socket?.ev.removeAllListeners();socket?.end(undefined);
      await pool.query('DELETE FROM wa_auth');
      qrValue=null;groups=[];connection='starting';lastDisconnect=null;closing=false;
      await start();
    });
    res.writeHead(202);return res.end();
  }
  try {
    if (req.method==='POST' && req.url==='/refresh') {await refresh();res.writeHead(204);return res.end();}
    if (req.method!=='GET') {res.writeHead(405); return res.end();}
    if (req.url==='/status') {res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({connection,groups,group_discovery_error:groupDiscoveryError,qr:!!qrValue && Date.now()-qrTime<60000,qr_generation:qrGeneration,last_disconnect:lastDisconnect,web_version:webVersion}));}
    if (req.url==='/qr' && qrValue && Date.now()-qrTime<60000) {res.setHeader('Content-Type','image/png'); return res.end(await QRCode.toBuffer(qrValue,{width:280,margin:2}));}
    res.writeHead(404);res.end();
  } catch (_) {res.writeHead(503);res.end();}
});
await refresh();
const publisher=createSummaryPublisher({pool,getSocket:()=>socket,getSelected:()=>targetGroup,
  getConnection:()=>connection,enabled:process.env.SUMMARY_PUBLISHING_ENABLED==='true'});
await publisher.recoverInterrupted();
setInterval(()=>void publisher.tick(),3000).unref();
server.listen(8090,'127.0.0.1');
setInterval(()=>enqueue(refresh,'heartbeat'),5000).unref();
setInterval(()=>void loadGroups(),60000).unref();
await start();
for (const signal of ['SIGINT','SIGTERM']) process.on(signal,async()=>{
  closing=true;clearTimeout(reconnectTimer);socket?.end(undefined);server.close();
  await serial;await pool.end();process.exit(0);
});
