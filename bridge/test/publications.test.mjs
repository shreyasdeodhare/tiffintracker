import test from 'node:test';
import assert from 'node:assert/strict';
import {createSummaryPublisher} from '../publications.mjs';

function fixture({connected=true,enabled=true,error=false,confirmed=true,stall=false,wrongId=false}={}) {
  const calls=[];
  let state='pending',sends=0;
  const part={id:1,publication_id:1,position:1,group_jid:'123@g.us',body:'Fixture summary',message_id:'fixture-id'};
  const client={release(){},async query(sql,args=[]){
    calls.push([sql,args]);
    if(sql.startsWith('SELECT part.')) return {rows:state==='pending'?[part]:[]};
    if(sql.startsWith("UPDATE summary_parts SET status='sending'")) state='sending';
    if(sql.startsWith('UPDATE summary_parts SET status=$2')) state=args[1];
    return {rows:[]};
  }};
  const pool={async connect(){return client;},query:client.query.bind(client)};
  const publisher=createSummaryPublisher({pool,getSelected:()=>part.group_jid,getConnection:()=>connected?'connected':'reconnecting',enabled,
    timeoutMs:stall?5:30000,getSocket:()=>({async sendMessage(group,body,options){
      assert.equal(state,'sending');
      assert.ok(calls.some(([sql])=>sql==='COMMIT'));
      assert.equal(group,part.group_jid);assert.equal(body.text,part.body);assert.equal(options.messageId,part.message_id);
      sends++;
      if(stall) return new Promise(()=>{});
      if(error) throw new Error('ambiguous timeout');
      return confirmed?{key:{id:wrongId?"different-id":part.message_id}}:{};
    }})});
  return {publisher,calls,state:()=>state,sends:()=>sends};
}

test('weekly sender commits claim before sending and never replays sent part',async()=>{
  const f=fixture();assert.equal(await f.publisher.tick(),true);
  assert.equal(f.state(),'sent');assert.equal(await f.publisher.tick(),false);assert.equal(f.sends(),1);
});
test('uncertain transport remains unknown and is not automatically retried',async()=>{
  const f=fixture({error:true});assert.equal(await f.publisher.tick(),false);
  assert.equal(f.state(),'unknown');await f.publisher.tick();assert.equal(f.sends(),1);
});
test('missing confirmation is unknown rather than sent',async()=>{
  const f=fixture({confirmed:false});await f.publisher.tick();assert.equal(f.state(),'unknown');
});
test('disabled or disconnected publisher cannot send',async()=>{
  for(const options of [{enabled:false},{connected:false}]) {
    const f=fixture(options);assert.equal(await f.publisher.tick(),false);assert.equal(f.sends(),0);
  }
});
test('interrupted claims are made unknown on receiver restart',async()=>{
  const f=fixture();await f.publisher.recoverInterrupted();
  assert.equal(f.calls.filter(([sql])=>sql.includes("SET status='unknown'")).length,2);
});

test('send deadline preserves unknown outcome and prevents a retry',async()=>{
 const f=fixture({stall:true});await f.publisher.tick();assert.equal(f.state(),'unknown');
 await f.publisher.tick();assert.equal(f.sends(),1);
});
test('unexpected transport identity is not marked successful',async()=>{
 const f=fixture({wrongId:true});await f.publisher.tick();assert.equal(f.state(),'unknown');
});
