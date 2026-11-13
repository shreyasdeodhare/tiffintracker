import test from 'node:test';
import assert from 'node:assert/strict';
import {createGroupDiscovery} from '../group-discovery.mjs';

test('selected group does not trigger optional all-group API requests',async()=>{
  let output;
  const selected={id:'123@g.us',name:'Authorized order group'};
  const run=createGroupDiscovery({getSelected:()=>selected,fetchGroups:()=>assert.fail('Should not query all groups'),onGroups:value=>output=value,onError:()=>assert.fail()});
  await run();assert.deepEqual(output,[selected]);
});
test('WhatsApp group-list 500 is contained and backs off without killing receiver',async()=>{
  let time=0,requests=0,errors=0;
  const run=createGroupDiscovery({getSelected:()=>null,fetchGroups:async()=>{requests++;throw Object.assign(new Error('synthetic failure'),{output:{statusCode:500}});},onGroups:()=>assert.fail(),onError:()=>errors++,now:()=>time});
  await run();await run();assert.equal(requests,1);assert.equal(errors,1);
  time=60000;await run();assert.equal(requests,2);
  time=120000;await run();assert.equal(requests,2);
});
test('concurrent metadata refreshes do not duplicate requests',async()=>{
  let resolve,requests=0;
  const run=createGroupDiscovery({getSelected:()=>null,fetchGroups:()=>{requests++;return new Promise(r=>resolve=r);},onGroups:()=>{},onError:()=>assert.fail()});
  const pending=run();await run();assert.equal(requests,1);resolve([]);await pending;
});
