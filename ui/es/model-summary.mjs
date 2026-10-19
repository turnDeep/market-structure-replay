// Presentation-only accounting over fills known at the selected close.
// Never reads the eventual exit of a position that is still open as of then.
export function summarizeModel(fills,latest,state,{pointValue=50,commissionSide=2.5,contracts=1}={}){
  if(!Number.isInteger(latest?.index)||!Number.isFinite(latest.close))throw new Error('集計の選択日・終値が不正です');
  if(!Number.isFinite(pointValue)||pointValue<=0||!Number.isFinite(commissionSide)||commissionSide<0||!Number.isInteger(contracts)||contracts<1)throw new Error('集計の契約倍率・手数料が不正です');
  const known=fills.filter(f=>f.index<=latest.index).sort((a,b)=>a.index-b.index);
  let position=null,realized=0,closed=0,wins=0,losses=0,breakeven=0,largest=0;
  for(const f of known){
    if(!Number.isInteger(f.index)||!Number.isFinite(f.price))throw new Error('保存約定の価格・時点が不正です');
    if(f.kind==='BUY'){
      if(position)throw new Error('単一ポジションの保存台帳と一致しません');
      position=f;
    }else if(f.kind==='SELL'||f.kind==='STOP'){
      if(!position||position.position_id!==f.position_id||!Number.isFinite(f.net_pnl_usd))throw new Error('保存決済の対応・損益が不正です');
      const expected=(f.price-position.price)*pointValue*contracts-2*commissionSide*contracts;
      if(Math.abs(f.net_pnl_usd-expected)>1e-7)throw new Error('保存損益と契約倍率・手数料の計算が一致しません');
      realized+=f.net_pnl_usd;closed++;
      if(f.net_pnl_usd>0)wins++;else if(f.net_pnl_usd<0)losses++;else breakeven++;
      largest=Math.max(largest,f.net_pnl_usd);position=null;
    }else throw new Error('未対応の保存約定種別です');
  }
  if((position?.position_id||null)!==(state?.position_id||null))throw new Error('選択日の保有状態と保存約定が一致しません');
  const open=position?(latest.close-position.price)*pointValue*contracts-commissionSide*contracts:null;
  return {asof_date:latest.date,asof_index:latest.index,realized_net_usd:realized,closed_trades:closed,
    wins,losses,breakeven,win_rate_pct:closed?100*wins/closed:null,
    open_position_id:position?.position_id||null,open_pnl_after_entry_fee_usd:open,
    largest_closed_gain_usd:largest,realized_excluding_largest_gain_usd:realized-largest,
    point_value_usd:pointValue,commission_per_side_usd:commissionSide,contracts};
}
