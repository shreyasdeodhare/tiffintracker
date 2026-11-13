// Explicit isolated integration test; never imports Baileys or opens WhatsApp.
import assert from 'node:assert/strict';
import {Pool} from 'pg';
import {createSummaryPublisher} from '../publications.mjs';

const database=process.env.TEST_POSTGRES_DB || '';
assert.ok(database.startsWith('tiffintrack_') && database.endsWith('_test'),'Disposable database required');
const pool=new Pool({database,host:'/var/run/postgresql',user:'postgres',max:3});
try {
  const group='120363222222222@g.us';
  // Reset only this disposable, mock-only fixture so an interrupted test can
  // be rerun. Production reconciliation never executes this reset SQL.
  await pool.query("UPDATE summary_parts part SET status='pending',error=NULL,sent_at=NULL FROM summary_publications p WHERE part.publication_id=p.id AND p.group_jid=$1",[group]);
  await pool.query("UPDATE summary_publications SET status='pending',error=NULL,sent_at=NULL WHERE group_jid=$1",[group]);
  const before=await pool.query("SELECT part.* FROM summary_parts part JOIN summary_publications p ON p.id=part.publication_id WHERE p.group_jid=$1 ORDER BY part.position",[group]);
  assert.ok(before.rows.length>0,'Python concurrency fixture must prepare a publication first');
  const sent=[];
  const options={pool,getSelected:()=>group,getConnection:()=>'connected',enabled:true,
    getSocket:()=>({async sendMessage(destination,content,metadata){
      assert.equal(destination,group);
      const claim=await pool.query('SELECT status FROM summary_parts WHERE message_id=$1',[metadata.messageId]);
      assert.equal(claim.rows[0].status,'sending');
      assert.ok(content.text.includes('Weekly Order Summary'));
      await new Promise(resolve=>setTimeout(resolve,30));
      sent.push(metadata.messageId);
      return {key:{id:metadata.messageId}};
    }})};
  const one=createSummaryPublisher(options),two=createSummaryPublisher(options);
  for(let index=0;index<before.rows.length+1;index++) await Promise.all([one.tick(),two.tick()]);
  assert.equal(sent.length,before.rows.length);
  assert.equal(new Set(sent).size,sent.length);
  const states=await pool.query('SELECT status FROM summary_publications WHERE group_jid=$1',[group]);
  assert.ok(states.rows.every(row=>row.status==='sent'));
  console.log('POSTGRES_CONCURRENT_OUTBOX_CLAIM_AND_MOCK_TRANSPORT_PASSED; no WhatsApp connection used');
} finally {await pool.end();}
