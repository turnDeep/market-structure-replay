import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {summarizeModel} from '../ui/nq/model-summary.mjs';

const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args=process.argv.slice(2),option=name=>{const i=args.indexOf(name);return i<0?null:args[i+1];};
const packed=await fs.readFile(path.join(root,'assets/nq.html.gz'));
const html=gunzipSync(packed).toString('utf8');
const snapshot=JSON.parse([...html.matchAll(/<script([^>]*)>(.*?)<\/script>/gs)].filter(m=>m[1].includes('type="application/json"')).map(m=>m[2]).join(''));
const q=snapshot.queries,fills=q.executions.rows,bars=q.market_prices.rows,states=q.daily_states.rows;
const config={pointValue:snapshot.replay.pointValue,commissionSide:snapshot.replay.commissionSide};
let checks=0;const eq=(a,b,m)=>{assert.deepEqual(a,b,m);checks++;};const near=(a,b,m)=>{assert.ok(Math.abs(a-b)<1e-7,`${m}: ${a} != ${b}`);checks++;};

// Independent running ledger uses actual fill prices, not saved closed PNL.
let position=null,realized=0,closed=0,wins=0,losses=0,largest=0;
const mtmPath=option('--saved-mtm'),savedMtm=mtmPath?(await fs.readFile(mtmPath,'utf8')).trim().split(/\r?\n/).map(JSON.parse):null;
if(savedMtm)eq(savedMtm.length,bars.length,'saved MTM coverage');
for(let i=0;i<bars.length;i++){
  for(const f of fills.filter(f=>f.index===i)){
    if(f.kind==='BUY'){assert.equal(position,null);position=f;}
    else{
      const pnl=(f.price-position.price)*20-5;
      realized+=pnl;closed++;if(pnl>0)wins++;if(pnl<0)losses++;largest=Math.max(largest,pnl);position=null;
    }
  }
  const result=summarizeModel(fills,bars[i],states[i],config);
  near(result.realized_net_usd,realized,'independent realized');
  eq([result.closed_trades,result.wins,result.losses],[closed,wins,losses],'counts');
  eq(result.win_rate_pct,closed?100*wins/closed:null,'denominator');
  eq(result.open_position_id,position?.position_id||null,'position');
  const open=position?(bars[i].close-position.price)*20-2.5:null;
  if(open===null)eq(result.open_pnl_after_entry_fee_usd,null,'flat is not an open trade');else near(result.open_pnl_after_entry_fee_usd,open,'entry fee included');
  near(result.largest_closed_gain_usd,largest,'largest known profit');
  near(result.realized_excluding_largest_gain_usd,realized-largest,'excluding only one winner');
  eq(result,summarizeModel(fills.filter(f=>f.index<=i),bars[i],states[i],config),'whole-input versus as-of input');
  if(savedMtm){
    eq(savedMtm[i].datetime.slice(0,10),bars[i].date,'saved ledger date');
    near(result.realized_net_usd,savedMtm[i].realized_net_usd,'saved realized');
    near(result.open_pnl_after_entry_fee_usd??0,savedMtm[i].open_pnl_after_entry_fee_usd,'saved open after entry fee');
  }
}

// Small software fixture: pending orders aren't fills; future errors stay future.
const b=(index,close)=>({index,date:`fixture-${index}`,close}),flat={position_id:null},long={position_id:'p1'};
const fake=[{index:1,kind:'BUY',position_id:'p1',price:100},{index:3,kind:'SELL',position_id:'p1',price:102,net_pnl_usd:35}];
eq(summarizeModel(fake,b(0,99),{...flat,pending_order_id:'pending'}).closed_trades,0);
eq(summarizeModel(fake,b(0,99),flat).win_rate_pct,null);
near(summarizeModel(fake,b(1,100),long).open_pnl_after_entry_fee_usd,-2.5,'entry-day fee');
eq(summarizeModel([...fake,{index:100,kind:'broken'}],b(1,100),long),summarizeModel(fake,b(1,100),long),'bad future row ignored');
eq(summarizeModel(fake,b(3,102),flat).wins,1);
eq(summarizeModel(fake,b(3,102),flat).open_pnl_after_entry_fee_usd,null);
assert.throws(()=>summarizeModel(fake,b(1,100),flat),/一致しません/);checks++;
assert.throws(()=>summarizeModel([{...fake[0]},{...fake[1],net_pnl_usd:40}],b(3,102),flat),/一致しません/);checks++;
const zero=[fake[0],{...fake[1],price:100.25,net_pnl_usd:0}];
eq(summarizeModel(zero,b(3,100.25),flat).breakeven,1);
eq(summarizeModel(zero,b(3,100.25),flat).win_rate_pct,0);
const last=summarizeModel(fills,bars.at(-1),states.at(-1),config);
const report={status:'PASS_NQ_ASOF_SUMMARY',checks,cutoffs:bars.length,saved_ledger_reconciled:!!savedMtm,last,
  original_trading_rows_changed:false,strategy_parameters_changed:false,strategy_status:'NOT_VALIDATED'};
if(option('--report'))await fs.writeFile(option('--report'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
