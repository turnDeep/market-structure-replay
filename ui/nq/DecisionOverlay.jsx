import React from 'react';

const num=v=>v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const shortDate=d=>d.slice(5);
const ink=family=>family==='sell'?'#af5429':family==='stop'?'#80439a':'#166d98';

export function DecisionOverlay({view,x,y,left,right,top,height,width,layer}){
  const evidence=view.evidence;
  if(!evidence)return null;
  const col=ink(evidence.family),mobile=width<650;
  const selected=view.adoptedZone;
  const candleStep=(right-left)/view.bars.length;
  const guideLabels=view.evidenceLevels.map(l=>({...l,targetY:y(l.price)})).sort((a,b)=>a.targetY-b.targetY);
  let previous=top-25;
  for(const l of guideLabels){l.labelY=Math.max(top+10,l.targetY,previous+21);previous=l.labelY;}
  if(guideLabels.length&&guideLabels.at(-1).labelY>top+height-10){
    const shift=guideLabels.at(-1).labelY-(top+height-10);for(const l of guideLabels)l.labelY-=shift;
  }
  const selectedMarks=view.evidenceMarks;
  const paths=evidence.family==='sell'?['1','N','2']:evidence.family==='recovery'?['H','1','2']:['2','L','3'];
  const points=paths.map(code=>selectedMarks.find(m=>m.code===code)).filter(Boolean);
  return <g data-testid={`decision-overlay-${layer}`} data-family={evidence.family}>
    {selected&&layer==='background'&&<g data-testid="adopted-ob-band" data-zone-id={selected.id}>
      {view.adoptedSegments.map((s,i)=><g key={`${s.index}-${i}`} data-adopted-state={s.state} data-from-index={s.from} data-to-index={s.to}>
        <rect x={x(s.from)-candleStep/2} y={y(selected.high)} width={Math.max(2,candleStep*(s.to-s.from+1))} height={Math.max(1,y(selected.low)-y(selected.high))} fill={s.state==='reclaim_wait'?'#97a6ad':'#2c85ad'} fillOpacity={s.state==='reclaim_wait'?.015:.11} stroke={s.state==='reclaim_wait'?'#7b898e':'#166d98'} strokeWidth="1.1" strokeDasharray={s.state==='reclaim_wait'?'4 3':undefined}/>
        {s.state!=='reclaim_wait'&&<rect x={x(s.from)-candleStep/2} y={y(selected.body_high)} width={Math.max(2,candleStep*(s.to-s.from+1))} height={Math.max(1,y(selected.body_low)-y(selected.body_high))} fill="#2c85ad" fillOpacity=".16"/>}
      </g>)}
      <rect x={left+5} y={Math.max(top+3,y(selected.high)-24)} width={mobile?118:225} height="21" rx="3" fill="#edf7fb" stroke="#166d98" strokeWidth=".7"/>
      <text x={left+10} y={Math.max(top+17,y(selected.high)-9)} fontSize={mobile?10:12} fill="#125876">① 採用OB{mobile?'（判断時）':` · ${selected.timeframe} · 判断時に有効`}</text>
    </g>}
    {points.length>1&&layer==='background'&&<path d={points.map((m,i)=>`${i?'L':'M'}${x(m.index)},${y(m.price)}`).join(' ')} fill="none" stroke={col} strokeDasharray="5 4" strokeWidth="1.25" strokeOpacity=".65" data-testid="structure-path"/>}
    {layer==='background'&&view.evidenceLevels.map(l=>{const sl=l.id==='sl',weak=['local-support','neckline','recovery-ref'].includes(l.id);return <line key={l.id} data-evidence-level={l.id} x1={Math.max(left,x(Math.max(view.start,l.from_index)))} x2={right} y1={y(l.price)} y2={y(l.price)} stroke={sl?'#80439a':weak?'#87968f':col} strokeWidth={weak?.85:1.25} strokeDasharray={sl?'6 4':weak?'3 4':undefined}/>;})}
    {layer==='foreground'&&guideLabels.map(l=><g key={l.id} data-testid={`decision-level-${l.id}`}>
      {!mobile&&<path d={`M${right-5},${y(l.price)}L${right+7},${l.labelY}`} stroke={l.id==='sl'?'#80439a':col} strokeWidth=".8" fill="none"/>}
      <rect x={mobile?left+3:right+8} y={l.labelY-11} width={mobile?122:177} height="19" rx="3" fill="#fafbf9" fillOpacity=".93"/>
      <text x={mobile?left+7:right+12} y={l.labelY+3} fontSize={mobile?9.5:11.5} fill={l.id==='sl'?'#80439a':col}>{mobile?l.label:l.label+' '+num(l.price)}</text>
    </g>)}
    {layer==='foreground'&&selectedMarks.map((m,i)=>{const yy=y(m.price),xx=x(m.index),above=m.code==='1'||m.code==='2'||m.code==='H',offset=m.code==='3a'?-34:m.code==='3b'?29:above?-20:23;
      const brief=m.code==='1'?(evidence.family==='sell'?'H1':'安値'):m.code==='2'?(evidence.family==='sell'?'H2':evidence.family==='reversal'?'接触':evidence.family==='stop'?'SL到達':'突破'):m.code==='N'?'首線':m.code==='H'?'参照高値':m.code==='L'?'安値':m.code==='3a'?'終値①':m.code==='3b'?'終値②':'判断';
      return <g key={m.code} data-evidence-mark={m.code} data-evidence-date={m.date}>
        <title>{m.code} {m.label} / {m.date} / {num(m.price)} · この根拠が既知になる日 {view.sourceBars.find(b=>b.index===m.known_index)?.date||evidence.date}</title>
        <circle cx={xx} cy={yy} r="5" fill="#fafbf9" stroke={col} strokeWidth="1.5"/>
        <line x1={xx} x2={xx} y1={yy} y2={yy+offset} stroke={col} strokeWidth=".7"/>
        <rect x={Math.max(left+2,Math.min(right-(mobile?25:75),xx-(mobile?10:35)))} y={yy+offset-11} width={mobile?22:72} height="20" rx="3" fill={col}/>
        <text x={Math.max(left+2,Math.min(right-(mobile?25:75),xx-(mobile?10:35)))+(mobile?11:36)} y={yy+offset+3} textAnchor="middle" fontSize={mobile?9:11} fill="white">{mobile?m.code:`${m.code} ${brief}`}</text>
      </g>;})}
  </g>;
}

export function DecisionSteps({choice}){
  if(!choice.evidence)return <p className="empty-state">この時点では注文判断・約定がありません。日付を進めると、当時の根拠が表示されます。</p>;
  const evidence=choice.evidence,steps=[...evidence.steps,evidence.execution_step];
  const names={sell_priority:'SELLとの競合なし',exit_reason_reentry:'決済後の再エントリー規則',early_conditions:'早期反転の条件',reentry_context:'新しい価格反応',current_support_reaction:'現在の支持反応',market_path:'市場構造・経路',support_provenance:'接触前の支持役割',source_zone:'採用OBの有効性',close_risk_geometry:'終値と初期SLの位置',execution_budget:'保有・注文・再使用の制約',reaction_trigger:'反応の扱い',recovery_structure:'新しい回復構造'};
  return <><div className="decision-steps" data-testid="decision-steps" data-reviewed-rows>{steps.map(s=><div key={s.code} className={s.passed===null?'decision-step waiting-step':'decision-step'}>
    <div><span className="step-code">{s.code}</span><strong>{s.title}</strong><small>{s.passed===null?'未観測':'成立'}</small></div><p>{s.detail}</p>
  </div>)}</div>{evidence.gates.length>0&&<details className="visual-gates" data-reviewed-rows><summary>図の価格条件に加えた原本のゲート判定</summary><div>{evidence.gates.filter(g=>g.status!=='not_applicable').map(g=><span key={g.name} title={g.reason}>✓ {names[g.name]||g.name}<small>{g.reason.includes('observed_only')?'観察のみ':g.passed?'通過':'不成立'}</small></span>)}</div></details>}</>;
}
