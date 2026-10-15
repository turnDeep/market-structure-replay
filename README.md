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

- NQ: `NQ=F`。2000-09-18～2026-10-14、有効日足6,559本。
- 日経: `NKD=F`、**CMEドル建て日経先物**。2004-02-17～2026-10-14、有効日足5,530本。大阪・円建て契約ではありません。
- `assets/manifest.json` に各HTMLと各クエリの行数・SHA-256・保存結果生成日時を記録。
- 保存済み結果の閲覧専用です。**価格取得・再計算・自動更新・実注文は実装していません。** 閲覧時にYahoo APIを呼びません。
- 元データは既存研究でyfinance / curl-cffi等から取得した保存結果です。このパッケージ作成時に再取得や売買ルールの変更はしていません。
- 更新は、管理者が別途取得・品質確認・v0.11再計算を完了したブラウザ版を取り込み、テスト後にcommit/pushします。失敗した未検証結果で既存データを置き換えないでください。

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
`assets/*.html.gz` は既存UI・既存データをまとめた配布物です。Python側でローソク足や売買判定を作り直していません。
元の研究フォルダ、チャット履歴、認証情報、絶対ローカルパスは同梱しません。

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
