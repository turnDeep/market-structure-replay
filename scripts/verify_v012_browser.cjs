/* Optional local browser QA. Requires Playwright and an installed Edge.
 * Tests only the loopback app and repository-owned saved artifacts. No orders,
 * remote app mutations, data acquisition or strategy execution occur here.
 */
const {chromium}=require('playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
const zlib=require('node:zlib');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const output=path.resolve(process.argv[2]||path.join(root,'.refresh-work/v012-20261019/browser'));
const url=process.argv[3]||'http://127.0.0.1:4190';
assert.equal(new URL(url).hostname,'127.0.0.1','Only the local preview is in scope');
const snapshot=html=>JSON.parse(html.match(/<script type="application\/json" id="data-app-reviewed-snapshot">([\s\S]*?)<\/script>/)[1]);
const fixture=async(market,archive=false)=>snapshot(zlib.gunzipSync(await fs.readFile(path.join(root,'assets',...(archive?['archive','v011']:[]),market+'.html.gz'))).toString());

(async()=>{
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const context=await browser.newContext({viewport:{width:1500,height:1100},acceptDownloads:true});
  const page=await context.newPage();
  page.setDefaultTimeout(15000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const report={scope:'Local saved-artifact/UI verification, not live strategy or Cloud deployment validation',markets:{}};
  try{
    for(const market of ['nq','nikkei']){
      console.log('Checking',market);
      const data=await fixture(market),old=await fixture(market,true);
      const q=data.queries,oldBuys=new Set(old.queries.executions.rows.filter(r=>r.kind==='BUY').map(r=>r.date));
      const changed=q.executions.rows.filter(r=>r.kind==='BUY'&&!oldBuys.has(r.date));
      assert.ok(changed.length,'A real changed-alpha BUY must be available for inspection');
      const selected=changed.at(-1),asof=selected.date;
      await page.goto(url+'/?market='+market);
      await page.getByText('NQ・日経先物｜v0.12 売買判断リプレイ',{exact:true}).waitFor({timeout:45000});
      const frame=page.frameLocator('iframe');
      await frame.locator('.replay-page').waitFor({timeout:45000});
      const heading=await frame.locator('.replay-heading').innerText();
      assert.ok(heading.includes('v0.12')&&heading.includes('NOT_VALIDATED'));
      await frame.getByTestId('asof-date').fill(asof);
      await frame.getByTestId('asof-date').blur();
      await frame.getByTestId('asof-label').filter({hasText:asof+' の終値後'}).waitFor();
      await frame.getByLabel('根拠を見る取引',{exact:true}).selectOption(selected.id);
      await frame.getByRole('button',{name:'根拠の範囲へ',exact:true}).click();
      await frame.locator('[data-event-id]').filter({has:frame.locator('title')}).first().waitFor();
      assert.ok((await frame.getByTestId('selected-case').innerText()).includes(asof));
      assert.ok((await frame.locator('.replay-page').innerText()).includes('選択日までのモデル集計'));
      assert.ok(!(await frame.locator('.replay-page').innerText()).includes('undefined日足'));
      const save=page.waitForEvent('download');
      await frame.getByRole('button',{name:'表示時点のJSON保存',exact:true}).click();
      const download=await save;
      assert.ok(download.suggestedFilename().includes('v012'),download.suggestedFilename());
      const jsonPath=path.join(output,download.suggestedFilename());await download.saveAs(jsonPath);
      const exported=JSON.parse(await fs.readFile(jsonPath,'utf8'));
      assert.equal(exported.asof,asof);
      assert.ok(exported.bars.every(b=>b.date<=asof));
      assert.ok(exported.fills.every(f=>f.date<=asof));
      assert.equal(exported.status,'NOT_VALIDATED');
      const fills=q.executions.rows.filter(r=>r.date<=asof),closed=fills.filter(r=>r.kind!=='BUY');
      const realized=closed.reduce((sum,r)=>sum+(market==='nq'?r.net_pnl_usd:r.net_pnl_points),0);
      const cards=frame.locator('.asof-results strong');
      const visibleRealized=Number((await cards.nth(0).innerText()).replace(/[^0-9.\-]/g,''));
      const visibleClosed=Number((await cards.nth(1).innerText()).split('/')[0].trim());
      assert.ok(Math.abs(visibleRealized-realized)<1e-7,'Visible realized P&L must match the independent as-of ledger');
      assert.equal(visibleClosed,closed.length,'Visible closed-trade count');
      if(exported.model_summary){
        assert.equal(exported.model_summary.closed_trades,closed.length);
        const shown=exported.model_summary.realized_net_usd??exported.model_summary.realized_net_points;
        assert.ok(Math.abs(shown-realized)<1e-7,'Exported as-of realized ledger');
      }
      await page.screenshot({path:path.join(output,market+'-asof.png'),fullPage:true});
      await frame.getByTestId('candlestick-chart').screenshot({path:path.join(output,market+'-chart.png')});
      await frame.getByRole('button',{name:'最新日へ',exact:true}).click();
      await frame.getByTestId('asof-label').filter({hasText:data.replay.end+' の終値後'}).waitFor();
      const screenText=await frame.locator('body').innerText();
      assert.ok(!screenText.includes('v0.11'),'No stale active-view version labels');
      report.markets[market]={status:'PASS',end:data.replay.end,bars:q.market_prices.rows.length,
        changedAlphaBuyInspected:asof,download:download.suggestedFilename(),closedThroughAsOf:closed.length,
        decisionEvidence:true,asOfFutureGuard:true,modelSummaryPresent:true,visibleSummaryLedgerChecked:true};
      console.log('PASS market',market);
    }
    // Streamlit's visible label wraps a visually clipped native radio input.
    // Click the same visible text a user clicks, not the clipped input itself.
    await page.getByText('NQ · NASDAQ 100先物',{exact:true}).click();
    await page.frameLocator('iframe').locator('.replay-heading').filter({hasText:'NQ=F'}).waitFor();
    report.marketSwitching=true;
    const mobileContext=await browser.newContext({viewport:{width:390,height:844}});
    const mobile=await mobileContext.newPage();
    await mobile.goto(url+'/?market=nq');
    await mobile.frameLocator('iframe').getByTestId('candlestick-chart').waitFor({timeout:45000});
    await mobile.screenshot({path:path.join(output,'nq-mobile.png'),fullPage:true});
    assert.equal(await mobile.evaluate(()=>innerWidth),390);
    report.mobileRendering=true;
    assert.deepEqual(errors,[],'No browser JavaScript exceptions');
    report.status='PASS_LOCAL_BROWSER';report.javascriptErrors=errors;
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
    await mobileContext.close();
    await context.close();
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
