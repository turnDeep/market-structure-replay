import assert from 'node:assert/strict';
import {currentReplayTitle} from '../ui/shared/current-model-title.mjs';
let checks=0;
const eq=(a,b)=>{assert.deepEqual(a,b);checks++;};
for(const symbol of ['NQ','ES']){
  const snapshot={id:'same-artifact',title:`${symbol}=F · v0.13 追体験`,replay:{strategyVersion:'v0.13'},queries:{executions:{rows:[{price:1}]}}};
  const frozen=JSON.stringify(snapshot);
  const personal={title:`${symbol}=F · v0.12 追体験`,theme:'classic',appearance:'light',filters:{saved:'preserve'},chartOverrides:{c:{type:'line'}},textEdits:{note:'preserve'}};
  const otherFields=JSON.stringify({...personal,title:null});
  const target=currentReplayTitle(snapshot,personal.title);
  eq(target,snapshot.title);
  // Same public setAppTitle semantics: update that property, not the record.
  personal.title=target;
  eq(JSON.stringify({...personal,title:null}),otherFields);
  eq(currentReplayTitle(snapshot,personal.title),null); // Idempotent.
  eq(JSON.stringify(snapshot),frozen);
  for(const custom of ['',undefined,`${symbol} study`,`custom v0.12 title`,`${symbol}=F · v0.12 追体験 (notes)`])eq(currentReplayTitle(snapshot,custom),null);
  eq(currentReplayTitle({...snapshot,replay:{strategyVersion:'v0.12'}},`${symbol}=F · v0.12 追体験`),null);
  eq(currentReplayTitle({...snapshot,title:'Custom v0.13'},`${symbol}=F · v0.12 追体験`),null);
}
eq(currentReplayTitle({title:'NKD=F · v0.13 追体験',replay:{strategyVersion:'v0.13'}},'NKD=F · v0.12 追体験'),null);
eq(currentReplayTitle(null,'anything'),null);
console.log(JSON.stringify({status:'PASS_SAVED_TITLE_GUARD',checks,queryRowsChanged:false,appIdentityChanged:false,storageCleared:false}));
