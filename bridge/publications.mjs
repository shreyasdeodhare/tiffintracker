// Durable weekly-summary transport. No schedule, parsing or financial arithmetic.
export function createSummaryPublisher({pool, getSocket, getSelected, getConnection, enabled,timeoutMs=30000}) {
  let busy = false;
  async function recoverInterrupted() {
    await pool.query("UPDATE summary_parts SET status='unknown',error='receiver_interrupted' WHERE status='sending'");
    await pool.query("UPDATE summary_publications SET status='unknown',error='receiver_interrupted' WHERE status='sending'");
  }
  async function tick() {
    if (!enabled || busy || getConnection() !== 'connected' || !getSelected()) return false;
    busy = true;
    let client, part;
    try {
      client = await pool.connect();
      await client.query('BEGIN');
      const result = await client.query(`SELECT part.*,p.group_jid FROM summary_parts part
        JOIN summary_publications p ON p.id=part.publication_id
        WHERE p.group_jid=$1 AND p.kind='weekly' AND p.status IN ('pending','sending')
        AND part.status='pending'
        AND NOT EXISTS (SELECT 1 FROM summary_parts earlier WHERE earlier.publication_id=p.id
          AND earlier.position<part.position AND earlier.status<>'sent')
        ORDER BY p.id,part.position LIMIT 1 FOR UPDATE OF p,part SKIP LOCKED`,[getSelected()]);
      part = result.rows[0];
      if (!part) {await client.query('ROLLBACK'); return false;}
      await client.query("UPDATE summary_parts SET status='sending',error=NULL WHERE id=$1",[part.id]);
      await client.query("UPDATE summary_publications SET status='sending',attempts=attempts+1,error=NULL WHERE id=$1",[part.publication_id]);
      await client.query("INSERT INTO publication_audits(publication_id,action,detail,created_at) VALUES($1,'sending',$2,now())",[part.publication_id,JSON.stringify({part:part.position,message_id:part.message_id})]);
      await client.query('COMMIT');
      client.release(); client = null;
      // Anything after invoking sendMessage may already have reached WhatsApp.
      // Never automatically turn such an exception back into pending.
      try {
        if (getConnection()!=='connected' || part.group_jid!==getSelected()) {
          await mark(part,'failed','not_connected_before_send');
          return false;
        }
        const socket=getSocket();
        if (!socket) {await mark(part,'failed','socket_unavailable_before_send');return false;}
        let timer,response;
        try {
          response=await Promise.race([
            socket.sendMessage(part.group_jid,{text:part.body},{messageId:part.message_id}),
            new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('send_timeout')),timeoutMs);})
          ]);
        } finally {clearTimeout(timer);}
        if (!response?.key?.id) {await mark(part,'unknown','missing_transport_confirmation');return false;}
        if (response.key.id!==part.message_id) {await mark(part,'unknown','unexpected_transport_identity');return false;}
        await mark(part,'sent',null);
        return true;
      } catch (_) {
        await mark(part,'unknown','transport_outcome_unknown');
        return false;
      }
    } catch (error) {
      if (client) {try {await client.query('ROLLBACK');} catch { /* connection failed */ }}
      // If claim committed, it stays sending and requires operator reconciliation.
      const code=/^[A-Z0-9]{5}$/.test(String(error?.code))?error.code:'none';
      process.stderr.write(`weekly_publication_storage_error code=${code}; inspect durable publication state\n`);
      return false;
    } finally {
      client?.release();busy=false;
    }
  }
  async function mark(part,status,error) {
    const client=await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query("SELECT id FROM summary_publications WHERE id=$1 FOR UPDATE",[part.publication_id]);
      await client.query("UPDATE summary_parts SET status=$2::text,error=$3,sent_at=CASE WHEN $2::text='sent' THEN now() ELSE NULL END WHERE id=$1",[part.id,status,error]);
      if(status!=='sent') {
        await client.query('UPDATE summary_publications SET status=$2,error=$3 WHERE id=$1',[part.publication_id,status,error]);
      } else {
        await client.query(`UPDATE summary_publications p SET status=CASE WHEN EXISTS
          (SELECT 1 FROM summary_parts part WHERE part.publication_id=p.id AND part.status<>'sent')
          THEN 'pending' ELSE 'sent' END,
          sent_at=CASE WHEN NOT EXISTS (SELECT 1 FROM summary_parts part WHERE part.publication_id=p.id AND part.status<>'sent')
          THEN now() ELSE NULL END WHERE p.id=$1`,[part.publication_id]);
      }
      await client.query("INSERT INTO publication_audits(publication_id,action,detail,created_at) VALUES($1,$2,$3,now())",[part.publication_id,status,JSON.stringify({part:part.position,message_id:part.message_id,error})]);
      await client.query('COMMIT');
    } catch(error) {await client.query('ROLLBACK');throw error;}
    finally {client.release();}
  }
  return {tick,recoverInterrupted};
}
