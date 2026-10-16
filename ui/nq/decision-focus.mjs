import {priceBounds,STATE_JA} from './replay-model.mjs';

export function selectCase(view,selectedId){
  const explicit=view.knownFills.find(e=>e.id===selectedId);
  const pending=view.knownDecisions.find(d=>d.id===view.state.pending_order_id);
  const event=explicit||(!pending&&(view.chartFills.at(-1)||view.knownFills.at(-1)))||null;
  const decision=event?.order_id?view.knownDecisions.find(d=>d.id===event.order_id):(!event?pending:null);
  const detail=event||decision||null;
  let evidence=null;
  if(detail?.visual){const parsed=JSON.parse(detail.visual);if(parsed.known_index<=view.cutoff)evidence=parsed;}
  return {event,decision,detail,evidence,pending:!event&&!!pending};
}

export function focusedView(data,view,choice,mode,showSma,showStop=true){
  const ev=choice.evidence;
  const sourceZone=ev?.adopted_zone;
  const zoneChanges=sourceZone?data.primary.filter(z=>z.id===sourceZone.id&&z.index<=view.cutoff&&z.available_index<=view.cutoff):[];
  const atDecision=zoneChanges.filter(z=>z.index<=ev?.decision_index).at(-1);
  const latest=zoneChanges.at(-1);
  const adopted=sourceZone?{...sourceZone,kind:'OB',direction:'bullish',source:'primary_v09',
    index:ev.decision_index,date:ev.date,origin_date:sourceZone.origin_date,available_date:sourceZone.available_date,
    state:atDecision?.state||'tradeable_saved_gate',current_state:latest?.state||'unknown',
    current_state_text:STATE_JA[latest?.state]||'状態ログ不明',body_low:sourceZone.body_low,body_high:sourceZone.body_high,
    from:Math.max(view.start,sourceZone.available_index),to:Math.min(view.end,ev.decision_index),adopted:true}:null;
  const visibleAdopted=adopted&&adopted.from<=adopted.to?adopted:null;
  const adoptedSegments=[];
  const changesAtDecision=zoneChanges.filter(z=>z.index<=ev?.decision_index);
  if(visibleAdopted){for(let i=0;i<changesAtDecision.length;i++){
    const z=changesAtDecision[i],from=Math.max(view.start,z.index),to=Math.min(view.end,ev.decision_index,changesAtDecision[i+1]?changesAtDecision[i+1].index-1:ev.decision_index);
    if(from<=to&&['active','recovered','reclaim_wait'].includes(z.state))adoptedSegments.push({...z,from,to});
  }}
  const bands=mode==='all'?view.bands.filter(z=>z.id!==adopted?.id):[];
  const zoneRows=mode==='all'?view.zones:(visibleAdopted?[visibleAdopted]:[]);
  const within=(m)=>m.index>=view.start&&m.index<=view.end&&m.known_index<=view.cutoff;
  const visibleMarks=ev?ev.marks.filter(within):[];
  const levels=ev?ev.levels.filter(l=>l.from_index<=view.end&&(showStop||l.id!=='sl')):[];
  const geometry=[...view.bars.flatMap(b=>[b.low,b.high,...(showSma&&Number.isFinite(b.sma200)?[b.sma200]:[])]),
    ...visibleMarks.map(m=>m.price),...levels.map(l=>l.price),...(visibleAdopted?[visibleAdopted.low,visibleAdopted.high]:[])];
  const low=Math.min(...geometry),high=Math.max(...geometry),pad=Math.max((high-low)*.11,10);
  const bounds=ev?{low:low-pad,high:high+pad}:priceBounds(view.bars,showSma);
  return {...view,bounds,bands,zones:zoneRows,adoptedZone:mode==='none'?null:visibleAdopted,adoptedSegments:mode==='none'?[]:adoptedSegments,evidence:ev,
    evidenceMarks:visibleMarks,evidenceLevels:levels,focusSources:zoneChanges,caseOutside:!!ev&&!visibleMarks.length,
    selectedCase:choice,referenceMode:mode==='all'};
}

export function fittedWindow(evidence,event,cutoff){
  const end=event?.index??cutoff;
  const needed=Math.max(60,end-evidence.focus_start+9);
  const window=[60,120,180,252,504,756].find(w=>w>=needed)||756;
  return {window,pan:Math.max(0,cutoff-end),truncated:needed>756};
}
