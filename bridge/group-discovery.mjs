// Optional UI metadata must never block or restart the durable message receiver.
export function createGroupDiscovery({getSelected, fetchGroups, onGroups, onError, now=Date.now}) {
  let busy=false, failures=0, retryAt=0;
  return async function discover() {
    const selected=getSelected();
    if(selected?.id) {onGroups([selected]);return;}
    if(busy || now()<retryAt) return;
    busy=true;
    try {
      const groups=await fetchGroups();
      onGroups(groups);
      failures=0;retryAt=0;
    } catch(error) {
      retryAt=now()+Math.min(900000,60000*2**Math.min(failures++,4));
      onError(error);
    } finally {busy=false;}
  };
}
