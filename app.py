import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time

# ページ設定
st.set_page_config(
    page_title="日本株マルチチャート",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.stock-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    margin: 0.2rem 0;
    background: #f8f9fa;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'current_watchlist' not in st.session_state:
    st.session_state.current_watchlist = 'default'

@st.cache_data
def load_stock_data():
    """株式データを読み込む"""
    try:
        df = pd.read_csv('data_j.csv')
        df = df[['コード', '銘柄名', '市場・商品区分', '33業種区分']].copy()
        df = df.rename(columns={
            'コード': 'code',
            '銘柄名': 'name',
            '市場・商品区分': 'market',
            '33業種区分': 'sector'
        })
        df = df[df['market'].isin(['プライム（内国株式）', 'スタンダード（内国株式）', 'グロース（内国株式）'])]
        df['code'] = df['code'].astype(str).str.zfill(4)
        df['ticker'] = df['code'] + '.T'
        df['display_name'] = df['code'] + ' - ' + df['name']
        return df
    except Exception as e:
        st.error(f"データファイルの読み込みエラー: {e}")
        return pd.DataFrame()

def calculate_vwap_bands(df, period=20):
    """VWAP バンドを計算"""
    if len(df) < period:
        return df
    
    # VWAP計算
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # VWAPからの偏差
    df['vwap_dev'] = df['Close'] - df['vwap']
    
    # 標準偏差計算
    df['vwap_std'] = df['vwap_dev'].rolling(window=period).std()
    
    # バンド計算
    df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
    df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']
    
    return df

@st.cache_data(ttl=300)
def get_stock_data(ticker, period='3mo', interval='1d'):
    """株価データを取得（90日分）"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        
        # 休日・取引時間外のデータを除外
        df = df.dropna()
        
        # VWAPバンド計算
        df = calculate_vwap_bands(df)
        return df
    except Exception as e:
        st.error(f"株価データの取得エラー ({ticker}): {e}")
        return None

def create_multi_chart(tickers_data, timeframe='1d', display_days=20):
    """12銘柄の4列×3行チャートを作成（90日データ、20日表示）"""
    if not tickers_data:
        return None

    # 4列×3行のサブプロット作成
    fig = make_subplots(
        rows=3, cols=4,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[f"{data['name'][:8]} ({data['code']})" for data in tickers_data[:12]]
    )

    for i, stock_data in enumerate(tickers_data[:12]):
        if stock_data['data'] is None or stock_data['data'].empty:
            continue
        
        df = stock_data['data']
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        # 最新20日分を表示用に取得
        display_df = df.iloc[-display_days:] if len(df) >= display_days else df
        
        # 休日を詰めるために日付を文字列に変換
        x_values = display_df.index.strftime('%m/%d').tolist()
        
        # ローソク足チャート
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=display_df['Open'],
                high=display_df['High'],
                low=display_df['Low'],
                close=display_df['Close'],
                name=stock_data['name'],
                increasing={'line': {'color': '#00D4AA'}, 'fillcolor': '#00D4AA'},
                decreasing={'line': {'color': '#FF6B6B'}, 'fillcolor': '#FF6B6B'},
                showlegend=False
            ),
            row=row, col=col
        )

        # VWAP
        if 'vwap' in display_df.columns and not display_df['vwap'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap'],
                    mode='lines',
                    name=f'VWAP_{i}',
                    line=dict(color='#FFD93D', width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（2σ）
        if 'vwap_upper_2' in display_df.columns and not display_df['vwap_upper_2'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap_upper_2'],
                    mode='lines',
                    line=dict(color='rgba(255, 107, 107, 0.4)', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap_lower_2'],
                    mode='lines',
                    line=dict(color='rgba(255, 107, 107, 0.4)', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 107, 0.03)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（1σ）
        if 'vwap_upper_1' in display_df.columns and not display_df['vwap_upper_1'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap_upper_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.6)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap_lower_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.6)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(106, 90, 205, 0.08)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>📈 日本株マルチチャート - {timeframe} (90日データ/20日表示)</b>",
            font=dict(size=18, color='#2C3E50'),
            x=0.5
        ),
        height=900,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=9, family="Arial, sans-serif"),
        margin=dict(l=15, r=15, t=50, b=15),
        dragmode='pan',
        showlegend=False
    )

    # すべてのX軸を category 型に設定（休日を詰める）
    fig.update_xaxes(
        type='category',
        showgrid=True,
        gridwidth=0.3,
        gridcolor='rgba(128,128,128,0.2)',
        tickangle=45,
        tickfont=dict(size=8)
    )
    
    # Y軸の設定
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor='rgba(128,128,128,0.2)',
        tickfont=dict(size=8)
    )

    # レンジスライダーを無効化
    for i in range(1, 13):
        row = ((i-1) // 4) + 1
        col = ((i-1) % 4) + 1
        fig.update_xaxes(rangeslider_visible=False, row=row, col=col)

    return fig

def save_watchlist(name, tickers):
    """ウォッチリストを保存"""
    if not os.path.exists('watchlists'):
        os.makedirs('watchlists')
    with open(f'watchlists/{name}.json', 'w', encoding='utf-8') as f:
        json.dump(tickers, f, ensure_ascii=False, indent=2)

def load_watchlist(name):
    """ウォッチリストを読み込み"""
    try:
        with open(f'watchlists/{name}.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def get_watchlist_names():
    """保存されたウォッチリスト名を取得"""
    if not os.path.exists('watchlists'):
        return []
    files = [f[:-5] for f in os.listdir('watchlists') if f.endswith('.json')]
    return files

def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>📈 日本株マルチチャート</h1>
        <p>最大12銘柄同時表示 - 90日データ/20日表示 - VWAP 1σ & 2σバンド対応</p>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    stock_df = load_stock_data()
    
    if stock_df.empty:
        st.error("株式データの読み込みに失敗しました。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 時間足設定
        st.subheader("⏰ 時間足設定")
        timeframe_options = {
            '5分足': ('5d', '5m'),
            '日足': ('3mo', '1d'),
            '週足': ('1y', '1wk'),
            '月足': ('2y', '1mo')
        }
        
        selected_timeframe = st.selectbox(
            "時間足",
            options=list(timeframe_options.keys()),
            index=1
        )
        
        period, interval = timeframe_options[selected_timeframe]
        
        # 銘柄検索と選択
        st.subheader("🔍 銘柄選択")
        
        # 検索フィルター
        search_term = st.text_input("銘柄名またはコードで検索", placeholder="例: トヨタ、7203")
        
        # 市場・業種フィルター
        col1, col2 = st.columns(2)
        with col1:
            market_filter = st.selectbox(
                "市場",
                ["全て"] + sorted(stock_df['market'].unique().tolist())
            )
        
        with col2:
            sector_options = ["全て"] + sorted([s for s in stock_df['sector'].unique() if pd.notna(s)])
            sector_filter = st.selectbox("業種", sector_options)
        
        # フィルタリング
        filtered_df = stock_df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(search_term, na=False, case=False) |
                filtered_df['code'].str.contains(search_term, na=False, case=False)
            ]
        
        if market_filter != "全て":
            filtered_df = filtered_df[filtered_df['market'] == market_filter]
            
        if sector_filter != "全て":
            filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
        
        # 銘柄選択（最大12個）
        available_options = filtered_df['ticker'].tolist()
        selected_tickers = st.multiselect(
            f"銘柄を選択 (最大12個) - {len(filtered_df)}件中",
            options=available_options,
            default=[t for t in st.session_state.selected_stocks if t in available_options],
            format_func=lambda x: stock_df[stock_df['ticker'] == x]['display_name'].iloc[0] if not stock_df[stock_df['ticker'] == x].empty else x,
            max_selections=12,
            key="stock_multiselect"
        )
        
        # 選択状態を保存
        st.session_state.selected_stocks = selected_tickers
        
        # 選択中の銘柄数表示
        st.info(f"選択中: {len(selected_tickers)}/12 銘柄")
        
        # ウォッチリスト管理
        st.subheader("⭐ ウォッチリスト管理")
        
        # 既存ウォッチリスト
        watchlist_names = get_watchlist_names()
        
        if watchlist_names:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_watchlist = st.selectbox(
                    "ウォッチリスト",
                    watchlist_names,
                    key="watchlist_select"
                )
            with col2:
                if st.button("📥 読込"):
                    watchlist_tickers = load_watchlist(selected_watchlist)
                    # 既存の選択に追加（重複除去）
                    new_selection = list(set(st.session_state.selected_stocks + watchlist_tickers))[:12]
                    st.session_state.selected_stocks = new_selection
                    st.success(f"'{selected_watchlist}'を読み込みました")
                    st.rerun()
            
            # ウォッチリスト削除
            if st.button(f"🗑️ '{selected_watchlist}'を削除", type="secondary"):
                os.remove(f'watchlists/{selected_watchlist}.json')
                st.success(f"'{selected_watchlist}'を削除しました")
                st.rerun()
        
        # 新規ウォッチリスト作成
        with st.expander("➕ 新しいウォッチリスト作成"):
            new_watchlist_name = st.text_input("ウォッチリスト名", key="new_watchlist_name")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 現在の選択を保存"):
                    if new_watchlist_name and selected_tickers:
                        save_watchlist(new_watchlist_name, selected_tickers)
                        st.success(f"'{new_watchlist_name}'に{len(selected_tickers)}銘柄を保存しました")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("名前と銘柄を入力してください")
            
            with col2:
                if st.button("📋 空のリスト作成"):
                    if new_watchlist_name:
                        save_watchlist(new_watchlist_name, [])
                        st.success(f"'{new_watchlist_name}'を作成しました")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("名前を入力してください")
        
        # クイック追加ボタン
        if watchlist_names and selected_tickers:
            st.subheader("⚡ クイック操作")
            target_list = st.selectbox("追加先", watchlist_names, key="target_list")
            if st.button(f"➕ '{target_list}'に追加"):
                existing = load_watchlist(target_list)
                new_list = list(set(existing + selected_tickers))
                save_watchlist(target_list, new_list)
                st.success(f"{len(selected_tickers)}銘柄を追加しました")
                time.sleep(1)
                st.rerun()
        
        # 人気銘柄ボタン
        st.subheader("🔥 人気銘柄")
        if st.button("人気銘柄12選を選択"):
            popular_stocks = [
                '7203.T', '6758.T', '8306.T', '6861.T',  # トヨタ、ソニー、三菱UFJ、キーエンス
                '9984.T', '8035.T', '4519.T', '6367.T',  # ソフトバンクG、東京エレクトロン、中外製薬、ダイキン
                '7974.T', '4063.T', '8001.T', '9020.T'   # 任天堂、信越化学、伊藤忠、JR東日本
            ]
            st.session_state.selected_stocks = popular_stocks
            st.success("人気銘柄12選を選択しました")
            st.rerun()
    
    # メインエリア
    if selected_tickers:
        st.subheader(f"📊 マルチチャート表示 - {selected_timeframe}")
        
        with st.spinner("チャートを読み込み中..."):
            # 各銘柄のデータを取得
            tickers_data = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(selected_tickers):
                stock_info = stock_df[stock_df['ticker'] == ticker]
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                    code = stock_info.iloc[0]['code']
                else:
                    name = ticker
                    code = ticker.replace('.T', '')
                
                stock_data = get_stock_data(ticker, period, interval)
                
                tickers_data.append({
                    'ticker': ticker,
                    'name': name,
                    'code': code,
                    'data': stock_data
                })
                
                progress_bar.progress((i + 1) / len(selected_tickers))
            
            progress_bar.empty()
            
            # マルチチャート作成
            multi_chart = create_multi_chart(tickers_data, selected_timeframe)
            
            if multi_chart:
                st.plotly_chart(multi_chart, use_container_width=True)
                
                # 銘柄一覧と最新価格
                st.subheader("💰 銘柄別最新価格・VWAP情報")
                
                cols = st.columns(4)
                for i, stock_data in enumerate(tickers_data[:12]):
                    with cols[i % 4]:
                        if stock_data['data'] is not None and not stock_data['data'].empty:
                            latest = stock_data['data'].iloc[-1]
                            prev_close = stock_data['data'].iloc[-2]['Close'] if len(stock_data['data']) > 1 else latest['Close']
                            change = latest['Close'] - prev_close
                            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:8]}",
                                value=f"¥{latest['Close']:,.0f}",
                                delta=f"{change_pct:+.2f}%"
                            )
                            
                            # VWAP情報
                            if 'vwap' in stock_data['data'].columns and not pd.isna(latest['vwap']):
                                vwap_diff = ((latest['Close'] - latest['vwap']) / latest['vwap']) * 100
                                st.caption(f"VWAP: ¥{latest['vwap']:,.0f} ({vwap_diff:+.1f}%)")
                        else:
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:8]}",
                                value="データなし",
                                delta=None
                            )
            else:
                st.error("チャートの作成に失敗しました")
    else:
        st.info("👈 左側から銘柄を選択してください（最大12銘柄）")
        
        # 使い方ガイド
        with st.expander("📖 使い方ガイド"):
            st.markdown("""
            ### 🔍 銘柄選択方法
            1. **検索**: 銘柄名やコードで検索
            2. **フィルター**: 市場・業種で絞り込み
            3. **複数選択**: 最大12銘柄まで選択可能
            4. **人気銘柄**: ワンクリックで主要12銘柄を選択
            
            ### ⭐ ウォッチリスト活用
            - **保存**: 現在の選択をウォッチリストに保存
            - **読込**: 保存済みリストを読み込み
            - **追加**: 既存リストに新しい銘柄を追加
            - **削除**: 不要なリストを削除
            
            ### 📊 チャート機能
            - **90日データ**: 過去90日分のデータを取得
            - **20日表示**: 最新20日分を表示（ドラッグで移動可能）
            - **VWAPバンド**: 1σ（紫）と2σ（赤）を表示
            - **休日除去**: 土日祝日は自動で詰めて表示
            """)
    
    # フッター
    st.markdown("---")
    st.markdown("💡 チャートはドラッグで移動・ズーム、90日分のデータから20日分を表示中")

if __name__ == "__main__":
    main()
