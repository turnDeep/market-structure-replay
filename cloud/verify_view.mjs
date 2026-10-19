import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const [market,work]=process.argv.slice(2);
assert.ok(['nq','nikkei'].includes(market));
const current=JSON.parse(await fs.readFile(path.join(work,'result/snapshot.json'),'utf8'));
const before=JSON.parse(await fs.readFile(path.join(work,'before.json'),'utf8'));
const {buildView}=await import(`./views/${market}/replay-model.mjs`);
const {selectCase,focusedView}=await import(`./views/${market}/decision-focus.mjs`);
const q=current.queries,data={bars:q.market_prices.rows,primary:q.primary_zones.rows,tactical:q.tactical_zones.rows,
  fills:q.executions.rows,decisions:q.decisions.rows,states:q.daily_states.rows};
let checks=0,prefixComparisons=0;
const ok=(v,m)=>{assert.ok(v,m);checks++;},eq=(a,b,m)=>{assert.deepEqual(a,b,m);checks++;};
const n=before.queries.market_prices.rows.length;
eq(current.id,before.id,'stable app identity');eq(current.buildStatus,'complete');
ok(data.bars.length>=n,'not truncated');eq(data.states.length,data.bars.length);
eq(current.replay.defaultDate,data.bars.at(-1).date);
for(const [name,query] of Object.entries(before.queries))eq(q[name].rows.filter(r=>r.index<n),query.rows,'frozen reviewed rows: '+name);
for(let i=0;i<data.bars.length;i++){
  const b=data.bars[i];eq(b.index,i);eq(data.states[i].date,b.date);
  ok(i===0||data.bars[i-1].date<b.date,'ordered dates');
  if(i>=199){const sma=data.bars.slice(i-199,i+1).reduce((s,b)=>s+b.close,0)/200;ok(Math.abs(sma-b.sma200)<1e-8,'independent SMA200');}
}
// An explicit strategy migration has no immutable prior trade rows (n=0).
// Test the first real observation, never a fictitious cutoff of -1.
const cutoffs=[...new Set([Math.max(0,n-1),data.bars.length-1,...data.decisions.filter(d=>d.index>=n).map(d=>d.index)])];
for(const i of cutoffs){
  const options={cutoff:i,window:120,pan:0,sma:true,ob:true,fvg:true,ifvg:false,tactical:false,waiting:false,history:false,frames:['1D','1W','1M']};
  const truncated={bars:data.bars.slice(0,i+1),states:data.states.slice(0,i+1),primary:data.primary.filter(r=>r.index<=i),
    tactical:data.tactical.filter(r=>r.index<=i),fills:data.fills.filter(r=>r.index<=i),decisions:data.decisions.filter(r=>r.index<=i)};
  for(const mode of ['decision','all','none']){
    const a=buildView(data,options),b=buildView(truncated,options);
    const full=focusedView(data,a,selectCase(a,null),mode,true,true),cut=focusedView(truncated,b,selectCase(b,null),mode,true,true);
    eq(full,cut,'as-of prefix '+i+'/'+mode);prefixComparisons++;
    for(const key of ['bars','knownFills','knownDecisions','sourceBars'])ok(full[key].every(r=>r.index<=i),'no future '+key);
  }
}
let entry=null;
for(const f of data.fills){
  const b=data.bars[f.index],v=JSON.parse(f.visual);eq(v.known_index,f.index);
  if(f.kind==='BUY'){ok(entry===null,'single position');entry=f;eq(f.index,f.decision_index+1);}
  else{
    ok(entry?.position_id===f.position_id,'own position');const points=f.price-entry.price;
    if(market==='nikkei')eq(f.net_pnl_points,points);else ok(Math.abs(f.net_pnl_usd-(points*20-5))<1e-7,'USD ledger');
    entry=null;
  }
  if(f.kind!=='STOP')eq(f.price,b.open+(f.kind==='BUY'?1:-1)*(market==='nq'?.25:5));
  else{const expected=(b.open<=f.initial_stop?b.open:f.initial_stop)-(market==='nq'?.25:5);eq(f.price,expected);ok(b.low<=f.initial_stop);}
}
for(const d of data.decisions){
  const v=JSON.parse(d.visual);eq(v.known_index,d.index);
  ok(!v.fill,'decision has no future fill');
  ok(v.marks.every(m=>m.index<=m.known_index&&m.known_index<=d.index),'causal anchors');
  ok(v.levels.every(l=>l.from_index<=d.index),'causal levels');
}
for(const s of data.states)if(s.pending_order_id)ok(data.decisions.some(d=>d.id===s.pending_order_id&&d.index<=s.index),'pending decision is visible');
const audit={status:'PASS_DATA_VIEW_AND_LEDGER',checks,prefixComparisons,market,bars:data.bars.length,end:data.bars.at(-1).date,
  upstream_detector_full_prefix_proof:false,predictive_edge_validated:false,browser_rendering:'unchanged compiled view; separately checked'};
await fs.writeFile(path.join(work,'result/view_audit.json'),JSON.stringify(audit,null,2));
console.log(JSON.stringify(audit));
