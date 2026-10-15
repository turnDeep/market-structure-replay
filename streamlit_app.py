"""Streamlit entrypoint: NQ / Nikkei saved-decision replay, not live trading."""

from __future__ import annotations

import streamlit as st

from replay_assets import AssetError, load_manifest, load_replay

st.set_page_config(
    page_title="Market Structure Replay",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LABELS = {"nq": "NQ · NASDAQ 100先物", "nikkei": "日経 · CMEドル建て先物"}


@st.cache_data(show_spinner=False, max_entries=2)
def cached_html(market: str, item: dict) -> str:
    return load_replay(market, item)


def main() -> None:
    st.title("Market Structure Replay", anchor=False)
    st.caption("NQ・日経先物｜v0.11 売買判断リプレイ")
    try:
        manifest = load_manifest()
    except AssetError as exc:
        st.error(str(exc))
        st.info("管理者が保存済みブラウザ版を取り込んでから再度開いてください。配置手順はREADMEを参照してください。")
        st.stop()

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
    st.caption(
        f"保存データ最終日 {item['end']} ｜ {item['symbol']} ｜ "
        "保存済み結果の閲覧専用・自動更新なし"
    )
    with st.sidebar:
        st.header("表示設定", anchor=False)
        frame_height = st.slider("リプレイ画面の高さ", 800, 2400, 1400, 100)
        st.markdown("**データと計算**")
        st.write(f"収録期間：{item['start']} ～ {item['end']}")
        st.write(f"有効日足：{item['bars']:,}本")
        st.caption("収録期間はファイル全体の範囲です。選択日当時の状態はリプレイ画面内で確認してください。")
        st.write(f"保存結果の生成日時：{item['snapshot_generated_at']}")
        st.caption("このアプリは価格APIを呼びません。画面の再読込だけでは価格や売買判定は更新されません。")
        st.markdown("**アクセス制限**")
        st.caption("GitHubのPrivate設定に加え、Streamlit CloudのSharingを「Only specific people can view this app」に設定してください。アプリ内表示は認証の代わりにはなりません。")

    with st.expander("使い方・分析上の注意", expanded=False):
        st.markdown(
            "日付は下のリプレイ画面内で指定します。採用された帯・売買矢印・履歴行から、"
            "当時の判断根拠を確認できます。銘柄ごとの表示状態は独立しています。\n\n"
            "200SMAは参照指標です。BUY・通常SELL・初期STOPを区別し、判断日と翌観測日の"
            "約定日を混同しないでください。FVGの扱いは各保存結果の説明を維持しています。\n\n"
            "日経は大阪・円建て契約ではなく **NKD=F（CMEドル建て日経）** です。"
            "コスト・契約倍率等の前提は銘柄ごとに異なります。"
            "既存モデルは **NOT_VALIDATED** のままで、将来の成績や実運用の優位性を保証しません。"
        )
    try:
        html = cached_html(selected, item)
    except AssetError as exc:
        st.error(str(exc))
        st.stop()
    # Only repository-owned, manifest-checked HTML is allowed here. No URL,
    # upload, user HTML, query parameter, or remote response reaches this iframe.
    st.iframe(html, height=frame_height, alt=f"{LABELS[selected]} v0.11 過去時点リプレイ")


if __name__ == "__main__":
    main()
