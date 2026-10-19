"""Streamlit entrypoint: NQ / ES saved-decision replay, not live trading."""

from __future__ import annotations

import streamlit as st

from replay_assets import AssetError, load_manifest, load_replay

st.set_page_config(
    page_title="Market Structure Replay",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LABELS = {"nq": "NQ · NASDAQ 100先物", "es": "ES · S&P 500先物"}


@st.cache_data(show_spinner=False, max_entries=2)
def cached_html(market: str, item: dict) -> str:
    return load_replay(market, item)


def main() -> None:
    st.title("Market Structure Replay", anchor=False)
    try:
        manifest = load_manifest()
    except AssetError as exc:
        st.error(str(exc))
        st.info("管理者が保存済みブラウザ版を取り込んでから再度開いてください。配置手順はREADMEを参照してください。")
        st.stop()

    version = manifest.get("strategy_version", "v0.11")
    st.caption(f"NQ・ES先物｜{version} 売買判断リプレイ")

    markets = list(LABELS)
    requested = st.query_params.get("market", "nq")
    if requested not in markets:
        requested = "nq"
    selected = st.radio(
        "分析対象", markets, index=markets.index(requested),
        format_func=LABELS.get, horizontal=True, key="market_selector",
    )
    if st.query_params.get("market") != selected:
        st.query_params["market"] = selected
    item = manifest["markets"][selected]
    refresh_text = "クラウド定期更新：毎日9時頃（日本時間）" if manifest.get("automatic_refresh") else "自動更新なし"
    st.caption(
        f"保存データ最終日 {item['end']} ｜ {item['symbol']} ｜ "
        f"閲覧専用 ｜ {refresh_text}"
    )
    quality = item.get("data_quality", {})
    if quality.get("provisional_from_date"):
        st.warning(f"{quality['provisional_from_date']}以降の末尾データは暫定です。Yahoo配信値の変動を確認したため、確定扱いにしません。確定確認済みの保存範囲は{quality['verified_through']}までです。次回、配信元の日付が進んだ後にこの暫定末尾だけを再取得・再計算します。")
    with st.sidebar:
        st.header("表示設定", anchor=False)
        frame_height = st.slider("リプレイ画面の高さ", 800, 2400, 1400, 100)
        st.markdown("**データと計算**")
        st.write(f"収録期間：{item['start']} ～ {item['end']}")
        st.write(f"有効日足：{item['bars']:,}本")
        st.caption("収録期間はファイル全体の範囲です。選択日当時の状態はリプレイ画面内で確認してください。")
        st.write(f"保存結果の生成日時：{item['snapshot_generated_at']}")
        st.caption("価格の取得・再計算はGitHubのクラウドで実行します。画面は検査済みの保存結果を読み込みます。新しい確定足がない日や更新失敗時は直前の結果を維持します。更新時刻には遅延があり、表示中の画面は再読込が必要です。")
        st.link_button("更新状況・実行履歴", "https://github.com/turnDeep/market-structure-replay/actions/workflows/refresh-data.yml")
        st.markdown("**アクセス制限**")
        st.caption("GitHubのPrivate設定に加え、Streamlit CloudのSharingを「Only specific people can view this app」に設定してください。アプリ内表示は認証の代わりにはなりません。")

    with st.expander("使い方・分析上の注意", expanded=False):
        if version == "v0.12":
            st.markdown("**v0.12**：検証時のαを正式採用。v0.11からBUY参照半径を2→1とし、確定済み半径1ピボットを追加しています。β・γは採用していません。価格を固定して過去の売買履歴を再計算した別バージョンです。旧v0.11はリポジトリの比較用アーカイブに保存しています。")
        st.markdown(
            "日付は下のリプレイ画面内で指定します。採用された帯・売買矢印・履歴行から、"
            "当時の判断根拠を確認できます。銘柄ごとの表示状態は独立しています。\n\n"
            "200SMAは参照指標です。BUY・通常SELL・初期STOPを区別し、判断日と翌観測日の"
            "約定日を混同しないでください。FVGの扱いは各保存結果の説明を維持しています。\n\n"
            "**NQ=FはNASDAQ 100、ES=FはS&P 500のE-mini先物**です。"
            "1枚固定でNQは20USD/pt、ESは50USD/pt。いずれも片道手数料2.5USD、"
            "片道スリッページ0.25ptをモデルに反映しています。"
            "既存モデルは **NOT_VALIDATED** のままで、将来の成績や実運用の優位性を保証しません。"
        )
    try:
        html = cached_html(selected, item)
    except AssetError as exc:
        st.error(str(exc))
        st.stop()
    # Only repository-owned, manifest-checked HTML is allowed here. No URL,
    # upload, user HTML, query parameter, or remote response reaches this iframe.
    st.iframe(html, height=frame_height, alt=f"{LABELS[selected]} {version} 過去時点リプレイ")


if __name__ == "__main__":
    main()
