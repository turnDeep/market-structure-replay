import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {summarizeModel} from '../ui/es/model-summary.mjs';
import {buildView} from '../ui/es/replay-model.mjs';
import {selectCase,focusedView} from '../ui/es/decision-focus.mjs';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args=process.argv.slice(2),opt=k=>args.includes(k)?args[args.indexOf(k)+1]:null;
const html=gunzipSync(await fs.readFile(path.join(root,'assets/es.html.gz'))).toString();
const snapshot=JSON.parse(html.match(/<script type="application\/json" id="data-app-reviewed-snapshot">([\s\S]*?)<\/script>/)[1]);
const q=snapshot.queries,bars=q.market_prices.rows,fills=q.executions.rows,states=q.daily_states.rows;
assert.equal(snapshot.replay.pointValue,50);assert.equal(snapshot.replay.commissionSide,2.5);
let checks=0;const eq=(a,b)=>{assert.deepEqual(a,b);checks++;},near=(a,b)=>{assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);checks++;};
const saved=opt('--saved-mtm')?(await fs.readFile(opt('--saved-mtm'),'utf8')).trim().split(/\r?\n/).map(JSON.parse):null;
if(saved)eq(saved.length,bars.length);
let position=null,realized=0,closed=0,wins=0,losses=0,largest=0;
for(const b of bars){
  for(const f of fills.filter(f=>f.index===b.index)){
    if(f.kind==='BUY'){eq(position,null);position=f;}
    else{eq(position.position_id,f.position_id);const p=(f.price-position.price)*50-5;
      near(f.net_pnl_usd,p);realized+=p;closed++;wins+=p>0;losses+=p<0;largest=Math.max(largest,p);position=null;}
  }
  const got=summarizeModel(fills,b,states[b.index]);
  near(got.realized_net_usd,realized);eq([got.closed_trades,got.wins,got.losses],[closed,wins,losses]);
  eq(got.win_rate_pct,closed?100*wins/closed:null);eq(got.open_position_id,position?.position_id||null);
  const open=position?(b.close-position.price)*50-2.5:null;
  if(open===null)eq(got.open_pnl_after_entry_fee_usd,null);else near(got.open_pnl_after_entry_fee_usd,open);
  near(got.realized_excluding_largest_gain_usd,realized-largest);
  eq(got,summarizeModel(fills.filter(f=>f.index<=b.index),b,states[b.index]));
  if(saved){eq(saved[b.index].datetime.slice(0,10),b.date);near(realized,saved[b.index].realized_net_usd);near(open??0,saved[b.index].open_pnl_after_entry_fee_usd);}
}
const data={bars,fills,states,primary:q.primary_zones.rows,tactical:q.tactical_zones.rows,decisions:q.decisions.rows};
const fvg=fills.find(f=>f.kind==='BUY'&&JSON.parse(f.visual).retest_zone);assert.ok(fvg);
eq(JSON.parse(fvg.visual).retest_zone.kind,'IFVG');eq(JSON.parse(fvg.visual).fvg_used,false);
const v=buildView(data,{cutoff:fvg.index,window:120,pan:0,sma:true,frames:['1D','1W','1M']});
const focused=focusedView(data,v,selectCase(v,fvg.id),'decision',true,true);
eq(focused.retestZone.id,JSON.parse(fvg.visual).retest_zone.id);
assert.ok(focused.retestSources.length&&focused.retestSources.every(r=>r.index<=fvg.index));checks++;
const exit=fills.find(f=>f.route==='構造保護安値割れSELL');assert.ok(exit);checks++;
const ev=JSON.parse(exit.visual);eq(ev.exit_reason,'major_protected_low_close_failure');
eq(ev.levels.find(l=>l.id==='protected-low').price,2552);
eq(bars[exit.decision_index].date,'2018-12-19');
const report={status:'PASS_ES_ASOF_SUMMARY',checks,cutoffs:bars.length,saved_ledger_reconciled:!!saved,
  last:summarizeModel(fills,bars.at(-1),states.at(-1)),fvgBuy:fvg.date,protectiveSell:exit.date,
  strategy_status:'NOT_VALIDATED',scope:'Saved-data accounting and view tests, not predictive validation'};
if(opt('--report'))await fs.writeFile(opt('--report'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
