export const VALID = new Set(['active','recovered']);
export const STATE_JA={active:'有効',recovered:'回復済み・有効',reclaim_wait:'回復待ち',filled:'充填済み',invalidated:'失効',expired:'期限切れ',distal_close_break:'遠端の終値突破'};
export function floorIndex(bars,date){let lo=0,hi=bars.length-1,result=0;while(lo<=hi){const mid=(lo+hi)>>1;if(bars[mid].date<=date){result=mid;lo=mid+1;}else hi=mid-1;}return result;}
export function priceBounds(rows,sma){const v=rows.flatMap(r=>[r.low,r.high,...(sma&&Number.isFinite(r.sma200)?[r.sma200]:[])]),low=Math.min(...v),high=Math.max(...v),pad=Math.max((high-low)*.1,10);return {low:low-pad,high:high+pad};}
export function zoneView(events,cutoff,start,end,bounds,opt){
  const groups=new Map();for(const r of events){if(r.index>cutoff||r.available_index>cutoff)continue;if(!groups.has(r.id))groups.set(r.id,[]);groups.get(r.id).push(r);}
  const rows=[],bands=[],sources=[];
  for(const changes of groups.values()){
    const latest=changes.at(-1);if(!opt.frames.includes(latest.timeframe)||!opt.kinds.includes(latest.kind)||latest.high<bounds.low||latest.low>bounds.high)continue;
    const allowed=s=>VALID.has(s)||(opt.waiting&&s==='reclaim_wait');if(!opt.history&&!allowed(latest.state))continue;
    const segments=[];for(let i=0;i<changes.length;i++){const r=changes[i],until=Math.min(changes[i+1]?changes[i+1].index-1:cutoff,end);if(!allowed(r.state)||r.index>end||until<start)continue;segments.push({...r,from:Math.max(start,r.index),to:until,current:allowed(latest.state)});}
    if(!segments.length)continue;rows.push({...latest,asof_index:cutoff,asof_state:latest.state,display_start:segments[0].from,display_end:segments.at(-1).to});bands.push(...segments);sources.push(...changes);
  }
  rows.sort((a,b)=>b.available_index-a.available_index||a.id.localeCompare(b.id));return {rows,bands,sources};
}
export function buildView(data,opt){
  const cutoff=Math.max(0,Math.min(data.bars.length-1,opt.cutoff)),end=Math.max(0,cutoff-opt.pan),start=Math.max(0,end-opt.window+1),bars=data.bars.slice(start,end+1),bounds=priceBounds(bars,opt.sma);
  const zopt={...opt,kinds:[...(opt.ob?['OB']:[]),...(opt.fvg?['FVG']:[]),...(opt.ifvg?['IFVG']:[])]},primary=zoneView(data.primary,cutoff,start,end,bounds,zopt),tactical=opt.tactical?zoneView(data.tactical,cutoff,start,end,bounds,zopt):{rows:[],bands:[],sources:[]};
  const knownFills=data.fills.filter(e=>e.index<=cutoff),knownDecisions=data.decisions.filter(e=>e.index<=cutoff),chartFills=knownFills.filter(e=>e.index>=start&&e.index<=end);
  return {cutoff,start,end,bars,bounds,primary,tactical,zones:[...primary.rows,...tactical.rows],bands:[...primary.bands,...tactical.bands],knownFills,knownDecisions,chartFills,state:data.states[cutoff],latest:data.bars[cutoff],sourceBars:data.bars.slice(Math.max(0,start-199),end+1)};
}
