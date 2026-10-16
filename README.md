# Market Structure Replay

NQ・日経先物のv0.11売買判断を、保存済みデータで追体験するStreamlitアプリです。
NQ / 日経切替、日付リプレイ、ローソク足、200SMA、採用OB/FVG、売買矢印と判断根拠を、既存ブラウザ版のまま表示します。

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

以下の本数・収録末尾は2026-10-16導入時点の記録です。以降の最新範囲はアプリおよび`assets/manifest.json`を参照してください。

- NQ: `NQ=F`。2000-09-18～2026-10-15、有効日足6,560本。
- 日経: `NKD=F`、**CMEドル建て日経先物**。2004-02-17～2026-10-15、有効日足5,531本。大阪・円建て契約ではありません。
- `assets/manifest.json` に各HTMLと各クエリの行数・SHA-256・保存結果生成日時を記録。
- 閲覧時にYahoo APIを呼びません。取得・再計算は **GitHub Actionsで毎日09:00頃（日本時間）** に実行します。PCの電源やStreamlit画面の起動は不要です。実注文は行いません。
- **2026-10-16の追加確認で、保存末尾10月15日のYahoo値がまだ変動していると判明しました。この末尾は暫定扱いに訂正しています。** 10月14日以前の凍結履歴は変更せず、10月15日は配信元の日付が進んだ後に限って再取得・再計算します。現在の品質状態はアプリの警告とmanifestに表示します。
- 最新の更新記録は [2026-10-16 更新監査](assets/refresh_20261016.json) です。10月14日までの全クエリ行のハッシュを固定して継続検査しています。
- 定期更新が全検査に成功した場合だけ、NQ・日経と継続計算入力を同じGitコミットで反映します。取得失敗、空応答、不正OHLCV、過去の売買変更、表示データの不整合、テスト失敗、push競合時は未検証結果を公開しません。新しい確定足がなければコミットしません。

### クラウド定期更新

実行状態と手動実行は [Cloud daily market refresh](https://github.com/turnDeep/market-structure-replay/actions/workflows/refresh-data.yml) で確認できます。
`Run workflow` の `refresh` は取得・検査後の更新、`verify` は最後の実データ1日を前日までの状態から全履歴再計算するオフライン回帰検査です。後者は公開データを変更しません。

- 時刻：毎日09:00 JST（00:00 UTC）。GitHubの混雑で開始が遅れることがあり、定刻保証はありません。
- 取得：yfinance + curl-cffiのChromeセッション。NQ→日経の順に取得し、最大3回まで再試行します。
- 確定判定：直近17:00 America/Chicago境界と、**配信元タイムゾーンの当日を含めない境界**の厳しい方を採用します。YahooのNY日付がまだ前日である日本の09:00時点では、日付ラベルが1日以上遅れる場合があります。CME取引所の正式セッション暦・清算値を保証するものではありません。
- 新規データ：実際に返った有効な行だけを使用。休日の推定行・価格補間なし。古いYahoo修正は記録しますが凍結履歴に遡及適用しません。
- 売買：固定v0.11・継続BUYオフ・回復BUYオンのまま。NQとNKDのシンボル以外は共通の原本コード、契約倍率・スリッページ等は保存設定を維持。最適化はしません。
- 検査：入力・コードハッシュ、過去の約定事実、過去の全表示行、独立した200SMA/損益計算、判断・約定クロック、将来行遮断、未約定判断の表示を確認します。
- 一時的な失敗：GitHub Actionsの失敗状態と14日保持の検査レポートに残し、直前の公開世代を維持します。勝手にローカル実行へ切り替えません。
- 権限：更新ジョブのみ組み込みの`GITHUB_TOKEN`に`contents: write`を付けます。個人PAT・Yahooログイン・ブローカー鍵は不要です。通常pushが拒否されたら停止し、force pushしません。
- 費用・停止：PrivateリポジトリのActions利用枠・組織ポリシーの影響を受けます。利用枠不足、GitHub/Yahoo障害、上流仕様変更では更新できません。
- Streamlit：このrepoの`main`をデプロイしている既存アプリにコミットが反映されます。休止・再起動や反映待ちがあり、開いたままのブラウザは再読込が必要です。GitHubのPrivateとアプリの閲覧制限は別です。

処理コードは`cloud/refresh.py`、原本ソースとそのハッシュは`cloud/vendor/`・`cloud/package_manifest.json`、継続計算入力は`cloud/state/`です。いずれもこのPrivateリポジトリだけで再実行でき、Windowsの元研究フォルダには依存しません。
新しいNQのFVG再テスト経路など、既存コンパイル済みビューが正確に表現できない経路が出た場合も、誤表示せず更新を停止します。これは戦略変更ではなく表示互換性の安全装置です。

以下は管理者による従来の手動インポート手順です。クラウド入力世代と不一致になるため、定期更新運用中に単独で実行せず、cloud/stateも整合させてください。

```powershell
python scripts/import_replays.py --nq <NQの完成済みHTML> --nikkei <日経の完成済みHTML>
python -m pytest -q
```

取り込み時、ローカルユーザーディレクトリを含む出典メタデータをファイル名だけにします。全クエリの価格・売買・ゾーン・日次状態の行が元HTMLと一致しない場合は取り込みを中断します。元HTMLのハッシュも保持します。

## 分析上の境界

- **NOT_VALIDATED**：表示・配布の検査と、戦略の予測優位性・実運用可否は別です。
- 選択日当時の表示、判断日終値と翌有効観測日の始値約定、通常SELLと初期STOP、反転BUYと回復BUYを区別します。
- 200SMAは追加の売買条件ではありません。FVGの扱いは保存済み銘柄別結果の説明を維持しています。
- 日経の不正OHLC等205行除外、NQ旧区間の凍結・追加区間の扱い、契約倍率・コスト差などの既存前提を変更していません。
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
