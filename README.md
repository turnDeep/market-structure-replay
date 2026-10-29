# Market Structure Replay

NQ・ES先物のv0.12売買判断を、保存済みデータで追体験するStreamlitアプリです。
NQ / ES切替、日付リプレイ、ローソク足、200SMA、採用OB/FVG、売買矢印と判断根拠を表示します。

## NQ・ESへの切替（2026-10-19）

- 通常メニューと日次更新は **NQ=F / ES=F**。旧NKDは通常表示・更新から外し、`assets/archive/nkd-v012/`へ保存しています。旧`cloud/state/nikkei`・原本ソースも削除していません。
- NQのHTML・価格・売買・表示状態のIDは今回変更していません。ESは以前のv0.12αと同じ設定で全履歴を再計算し、2026-10-14までの44決済と6,573日分の損益・保有状態に一致しています。
- ESは2000-09-18～2026-10-16の6,575有効日足。1枚 **50USD/pt**、片道手数料2.5USD・スリッページ0.25pt。NQは20USD/ptです。
- ESの構造後FVG/IFVG再テストBUYは紫のR帯を表示します。収録履歴の実例（2003-10-06 BUY）はIFVGです。通常の天井SELLと、構造保護安値を2終値連続で割るSELLを区別します。これは既存ルールの可視化であり、新ルールの追加ではありません。
- 取得・再計算・表示検査の記録は`assets/es_addition_20261019.json`。旧NQ・日経のリリース監査は歴史記録としてそのまま残します。GitHubの更新時刻・共有設定は変更しません。

## v0.12（2026-10-19・初回NQ/日経リリースの記録）

検証時の **v0.12α**（依頼文のv0.11α）を正式な **v0.12** とします。
両市場に同じ条件を適用し、市場別の最適化は行いません。

- 変更は `legacy_rules.buy_reference_radius: 2 → 1` のみ。既存半径2・5に加え、左右1本の厳密な高値・安値を、右側の足を観測した時点から供給します。
- β・γは不採用。`emergency_buffer_atr=0.10`、`major_decline_bars=15`を維持します。
- 価格・原本OB入力は保存済みの2026-10-16までを固定し、過去の全売買・判断根拠・日次状態を再計算します。価格の新規取得ではありません。
- 通常の継続BUYオフ、回復BUYオン、売りルール、各市場の従来の約定・コスト・契約倍率を維持します。
- 旧v0.11の表示HTMLとmanifestは `assets/archive/v011/` に保存。移行前コミットは `assets/v012_release.json` に記録します。旧版は凍結比較用で、日次更新の対象はv0.12だけです。
- `assets/v012_release.json`は全履歴再計算、研究時αの2026-10-14までの取引一致、表示データの検査結果と新しい履歴ハッシュを記録します。日次更新ではv0.12内の過去履歴を固定します。
- **正式版という名称は、戦略の予測優位性や実運用の検証完了を意味しません。`NOT_VALIDATED`を維持します。** 日経αの減益・DD拡大を含む先行研究結果も、昇格によって取り消しません。

確認記録は [v0.12リリース検査](qa/release-v012-20261019.md) を参照してください。
現行画面・出典定義の表記はv0.12に統一し、[表記の追補検査](qa/version-labels-20261019.md)も実施しています。
旧`file://`ローカルHTMLや旧版アーカイブは実際のv0.11結果なので、その名前をv0.12へ書き換えません。

既存のStreamlit Community Cloudアプリがこのリポジトリの`main` /
`streamlit_app.py`を参照していれば、push後は同じURLでv0.12へ反映されます。
通常、新規アプリの作成やデータの手動アップロードは不要です。表示中のタブは
再読込してください。価格データは今回2026-10-16までを固定しており、日次更新は
従来どおりGitHub Actionsがv0.12の設定で行います。Cloud側のデプロイ完了確認は
GitHubへのpush・ローカル表示確認とは別です。

## 非公開で使うための設定

**このリポジトリはPrivateで運用してください。** アプリ・データをPublicリポジトリにpushすると、後からPrivateにしても第三者が取得したコピーは回収できません。
コード側は認証を独自実装しません。Streamlit Community Cloudのアクセス制御を利用します。

1. GitHubの **Settings → General → Danger Zone → Change repository visibility → Private** を設定。
2. Streamlit Community CloudにGitHubでログインし、対象のPrivateリポジトリへのアクセスを許可。
3. **Create app / Deploy an app** で次を指定。
   - Repository: `turnDeep/market-structure-replay`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Advanced settings: Python **3.11**
4. デプロイ後、アプリの **Settings → Sharing → Only specific people can view this app** を確認。
5. 必要な閲覧者だけを登録。ログアウトした別ブラウザでアクセス制限を確認。

GitHubがPrivateでも、Streamlitの共有設定をPublicに変更すれば画面は公開されます。両方の設定を確認してください。
無料Community Cloudはアクセスがないと休止するため、常時稼働・即時表示の保証はありません。

## ローカル起動

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Mac/Linuxは `python3 -m venv .venv` と `.venv/bin/python` に読み替えます。

## データの範囲・更新

以下はNQ/ES切替時点の範囲です。以降の最新範囲はアプリおよび`assets/manifest.json`を参照してください。

- NQ: `NQ=F`。2000-09-18～2026-10-16、有効日足6,561本。
- ES: `ES=F`、E-mini S&P 500先物。2000-09-18～2026-10-16、有効日足6,575本。
- `assets/manifest.json` に各HTMLと各クエリの行数・SHA-256・保存結果生成日時を記録。
- 閲覧時にYahoo APIを呼びません。取得・再計算は **GitHub Actionsで米国先物営業日の翌日09:00頃（日本時間）** に実行します。PCの電源やStreamlit画面の起動は不要です。実注文は行いません。
- 2026-10-16に検出した旧末尾10月15日の暫定状態は、その後の確定候補再取得で解消済みです。現在は両市場とも10月16日までを保存しています。確定候補の採用後も、後日の配信修正がないことを保証するものではありません。
- 最新の更新記録は [2026-10-16 更新監査](assets/refresh_20261016.json) です。10月14日までの全クエリ行のハッシュを固定して継続検査しています。
- 定期更新が全検査に成功した場合だけ、NQ・ESと継続計算入力を同じGitコミットで反映します。取得失敗、空応答、不正OHLCV、過去の売買変更、表示データの不整合、テスト失敗、push競合時は未検証結果を公開しません。新しい確定足がなければコミットしません。

### クラウド定期更新

実行状態と手動実行は [Cloud daily market refresh](https://github.com/turnDeep/market-structure-replay/actions/workflows/refresh-data.yml) で確認できます。
`Run workflow` の `refresh` は取得・検査後の更新、`verify` は最後の実データ1日を前日までの状態から全履歴再計算するオフライン回帰検査です。後者は公開データを変更しません。

- 時刻：米国CME株価指数先物の営業日の翌日09:00 JST。火曜〜土曜09:00にCME営業日判定を行い、完全休場日の翌日は更新ジョブをスキップします。短縮取引日の翌日は対象です。金曜分は土曜朝に処理し、日曜・月曜朝は定期実行しません。日本の祝日では判定しません。GitHubの混雑で開始が遅れることがあり、定刻保証はありません。
- 営業日判定：`pandas_market_calendars==5.5.0` の `CME_Equity`。将来の臨時休場や取引所による変更を完全に保証するものではありません。手動更新は営業日判定にかかわらず利用できます。
- 取得：yfinance + curl-cffiのChromeセッション。NQ→ESの順に取得し、最大3回まで再試行します。NKDは取得対象ではありません。
- 確定判定：直近17:00 America/Chicago境界と、**配信元タイムゾーンの当日を含めない境界**の厳しい方を採用します。YahooのNY日付がまだ前日である日本の09:00時点では、日付ラベルが1日以上遅れる場合があります。CME取引所の正式セッション暦・清算値を保証するものではありません。
- 新規データ：実際に返った有効な行だけを使用。休日の推定行・価格補間なし。古いYahoo修正は記録しますが凍結履歴に遡及適用しません。
- 売買：固定v0.12・継続BUYオフ・回復BUYオン。原本v0.11計算基盤に半径1供給とBUY参照半径1を適用し、契約倍率・スリッページ等は保存設定を維持。最適化はしません。
- 検査：入力・コードハッシュ、過去の約定事実、過去の全表示行、独立した200SMA/損益計算、判断・約定クロック、将来行遮断、未約定判断の表示を確認します。
- 一時的な失敗：GitHub Actionsの失敗状態と14日保持の検査レポートに残し、直前の公開世代を維持します。勝手にローカル実行へ切り替えません。
- 権限：更新ジョブのみ組み込みの`GITHUB_TOKEN`に`contents: write`を付けます。個人PAT・Yahooログイン・ブローカー鍵は不要です。通常pushが拒否されたら停止し、force pushしません。
- 費用・停止：PrivateリポジトリのActions利用枠・組織ポリシーの影響を受けます。利用枠不足、GitHub/Yahoo障害、上流仕様変更では更新できません。
- Streamlit：このrepoの`main`をデプロイしている既存アプリにコミットが反映されます。休止・再起動や反映待ちがあり、開いたままのブラウザは再読込が必要です。GitHubのPrivateとアプリの閲覧制限は別です。

処理コードは`cloud/refresh.py`、原本ソースとそのハッシュは`cloud/vendor/`・`cloud/package_manifest.json`、継続計算入力は`cloud/state/`です。いずれもこのPrivateリポジトリだけで再実行でき、Windowsの元研究フォルダには依存しません。
新しいNQのFVG再テスト経路など、既存コンパイル済みビューが正確に表現できない経路が出た場合も、誤表示せず更新を停止します。これは戦略変更ではなく表示互換性の安全装置です。

以下は管理者による従来の手動インポート手順です。クラウド入力世代と不一致になるため、定期更新運用中に単独で実行せず、cloud/stateも整合させてください。

```powershell
python scripts/import_replays.py --nq <NQの完成済みHTML> --es <ESの完成済みHTML>
python -m pytest -q
```

取り込み時、ローカルユーザーディレクトリを含む出典メタデータをファイル名だけにします。全クエリの価格・売買・ゾーン・日次状態の行が元HTMLと一致しない場合は取り込みを中断します。元HTMLのハッシュも保持します。

## 分析上の境界

### NQ・ESの「選択日までのモデル集計」

両市場とも根拠を選ぶ操作の下に集計を表示します。チャートの表示幅ではなく、収録開始から選択日終値までに約定した保存履歴が対象です。

- 決済済み損益：既知のSELL/STOPの手数料控除後損益をUSDで合計。
- 決済数・勝ち・負け・勝率：未決済と翌始値待ち注文を除外。同値も勝率の分母に含め、決済0件の勝率は欠測。
- 未決済の含み損益：1枚・NQ 20USD/ptまたはES 50USD/pt、選択日終値で評価し、新規手数料2.50USDを控除。将来の決済手数料は未控除で、実現損益とは別表示。
- 最大1取引の決済益と、その1取引だけを除いた実現損益：大きな利益への依存を確認する補助情報。

この集計UIの初回追加時には売買アルゴリズム・価格データ・保存約定を変更していません。v0.12への移行時には、集計もv0.12で再計算した約定に更新します。表示時点のJSON保存にも`model_summary`を含めます。レビュー用の画面ソースは`ui/nq/`、全選択日の独立照合は`scripts/verify_nq_summary.mjs`です。初回確認結果は [NQ集計の追加検査](qa/nq-summary-20261016.md) を参照してください。

### 共通の注意点

- **NOT_VALIDATED**：表示・配布の検査と、戦略の予測優位性・実運用可否は別です。
- 選択日当時の表示、判断日終値と翌有効観測日の始値約定、通常SELLと初期STOP、反転BUYと回復BUYを区別します。
- 200SMAは追加の売買条件ではありません。FVGの扱いは保存済み銘柄別結果の説明を維持しています。
- ESの元取得時の不正日足53行は除外済み・補間なし。NQ旧区間の凍結・追加区間の扱いも維持しています。除外日は隣接足・ATR等に影響しえます。これは取引所認証済み限月データではありません。
- データをブラウザに送るため、許可された閲覧者は収録期間全体を取得できます。「選択日まで表示」は機密データのアクセス制御ではありません。
- 保存済み結果に将来の利益保証はなく、投資助言や発注システムではありません。

## 構成

`streamlit_app.py` が銘柄切替と固定HTMLの表示を担当し、`replay_assets.py` が圧縮HTMLの整合性を確認します。
`assets/*.html.gz` は既存UI・既存データをまとめた配布物です。クラウドは固定原本の売買計算を行い、同じコンパイル済みUIへ検査済みJSONだけを渡します。チャット履歴・認証情報・元の研究フォルダ全体は同梱しません。必要な研究ソースと入力だけをPrivate repoに保持します。凍結CSVの出典列と原本検出器の未使用の文書CLI既定値には元PCのパス文字列が残りますが、クラウド処理はそのパスを参照せず、すべて明示したrepo内の入力を使用します。閲覧用HTMLの出典は従来どおりパスを除去します。

## 検査

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

データ整合性、元のクエリ行ハッシュ、銘柄許可リスト、不正パス・破損データ拒否、秘密情報の基本検査を実施します。
GitHub ActionsのチェックはLinux / Python 3.11で実行します。ブラウザの実表示確認やCloud認証確認とは別です。

今回のローカル・実ブラウザ確認結果と未完了事項は [確認記録](qa/verification.md) を参照してください。
スマートフォン縦向きではチャート注釈が密になるため、詳細確認は横向きの利用を推奨します。

## 参照

- [Streamlit Community Cloud 配置手順](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit 公開・非公開の設定](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [st.iframe](https://docs.streamlit.io/develop/api-reference/text/st.iframe)
