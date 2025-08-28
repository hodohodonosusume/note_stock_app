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
    page_title="日本株リアルタイムチャート",
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
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.stSelectbox > div > div {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

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
        # 選択用の表示名を作成
        df['display_name'] = df['code'] + ' ' + df['name']
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
def get_stock_data(ticker, period='1mo', interval='1d'):
    """株価データを取得"""
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

def create_multi_chart(tickers_data, timeframe='1d'):
    """12銘柄の4列×3行チャートを作成"""
    if not tickers_data:
        return None

    # 4列×3行のサブプロット作成
    fig = make_subplots(
        rows=3, cols=4,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[f"{data['name'][:10]} ({data['code']})" for data in tickers_data[:12]]
    )

    colors = ['#00D4AA', '#FF6B6B', '#FFD93D', '#6A5ACD', '#FF69B4', '#32CD32',
              '#FF4500', '#1E90FF', '#DC143C', '#00CED1', '#9370DB', '#FFA500']

    for i, stock_data in enumerate(tickers_data[:12]):
        if stock_data['data'] is None or stock_data['data'].empty:
            continue
        
        df = stock_data['data']
        row = (i // 4) + 1
        col = (i % 4) + 1
        color = colors[i % len(colors)]
        
        # 休日を詰めるために日付を文字列に変換
        x_values = df.index.strftime('%m/%d').tolist()
        
        # ローソク足チャート
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=stock_data['name'],
                increasing={'line': {'color': '#00D4AA'}, 'fillcolor': '#00D4AA'},
                decreasing={'line': {'color': '#FF6B6B'}, 'fillcolor': '#FF6B6B'},
                showlegend=False
            ),
            row=row, col=col
        )

        # VWAP
        if 'vwap' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df['vwap'],
                    mode='lines',
                    name=f'VWAP_{i}',
                    line=dict(color='#FFD93D', width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（1σ）
        if 'vwap_upper_1' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df['vwap_upper_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.5)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df['vwap_lower_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.5)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(106, 90, 205, 0.1)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>📈 日本株マルチチャート - {timeframe}</b>",
            font=dict(size=20, color='#2C3E50'),
            x=0.5
        ),
        height=900,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=10, family="Arial, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
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

    # 各サブプロットのX軸レンジスライダーを無効化
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
        <p>12銘柄同時表示 - 3802銘柄から自由選択可能</p>
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
            '日足': ('1mo', '1d'),
            '週足': ('6mo', '1wk'),
            '月足': ('2y', '1mo')
        }
        
        selected_timeframe = st.selectbox(
            "時間足",
            options=list(timeframe_options.keys()),
            index=1
        )
        
        period, interval = timeframe_options[selected_timeframe]
        
        # 表示銘柄選択方法
        st.subheader("📊 銘柄選択")
        display_mode = st.radio(
            "選択方法",
            ["手動選択", "ウォッチリスト", "人気銘柄", "業種別"]
        )
        
        selected_tickers = []
        
        if display_mode == "手動選択":
            st.markdown("**🔍 銘柄を検索・選択してください（最大12銘柄）**")
            
            # 検索機能
            search_term = st.text_input(
                "銘柄検索",
                placeholder="銘柄名またはコードで検索（例：トヨタ、7203）",
                help="銘柄名やコードの一部を入力すると候補が絞り込まれます"
            )
            
            # 検索結果をフィルタリング
            filtered_stocks = stock_df.copy()
            if search_term:
                filtered_stocks = stock_df[
                    stock_df['name'].str.contains(search_term, na=False, case=False) |
                    stock_df['code'].str.contains(search_term, na=False, case=False)
                ]
            
            # 表示する選択肢を制限（パフォーマンス向上）
            display_limit = 100 if search_term else 50
            options_list = filtered_stocks['display_name'].head(display_limit).tolist()
            
            # マルチセレクト
            selected_display_names = st.multiselect(
                f"銘柄選択 ({len(filtered_stocks)}銘柄中)",
                options=options_list,
                default=[],
                max_selections=12,
                help="最大12銘柄まで選択できます"
            )
            
            # 選択された銘柄のティッカーを取得
            if selected_display_names:
                selected_stocks = stock_df[stock_df['display_name'].isin(selected_display_names)]
                selected_tickers = selected_stocks['ticker'].tolist()
                
                st.success(f"✅ {len(selected_tickers)}銘柄を選択中")
                
                # 選択された銘柄の一覧表示
                with st.expander("選択中の銘柄"):
                    for _, row in selected_stocks.iterrows():
                        st.write(f"• {row['code']} {row['name']}")
            else:
                st.info("👆 上のボックスから銘柄を選択してください")
                if not search_term:
                    st.write("💡 検索ボックスに銘柄名やコードを入力すると見つけやすくなります")
        
        elif display_mode == "ウォッチリスト":
            watchlist_names = get_watchlist_names()
            if watchlist_names:
                selected_watchlist = st.selectbox(
                    "ウォッチリスト選択",
                    watchlist_names
                )
                if selected_watchlist:
                    selected_tickers = load_watchlist(selected_watchlist)[:12]
                    st.success(f"✅ {selected_watchlist}から{len(selected_tickers)}銘柄を読み込み")
            else:
                st.info("ウォッチリストがありません")
        
        elif display_mode == "人気銘柄":
            # 人気銘柄（例）
            popular_stocks = [
                '7203.T', '6758.T', '8306.T', '6861.T',  # トヨタ、ソニー、三菱UFJ、キーエンス
                '9984.T', '8035.T', '4519.T', '6367.T',  # ソフトバンクG、東京エレクトロン、中外製薬、ダイキン
                '7974.T', '4063.T', '8001.T', '9020.T'   # 任天堂、信越化学、伊藤忠、JR東日本
            ]
            selected_tickers = popular_stocks
            st.success("✅ 人気銘柄12社を選択")
        
        else:  # 業種別
            sectors = stock_df['sector'].unique()
            selected_sector = st.selectbox(
                "業種選択",
                sorted([s for s in sectors if pd.notna(s)])
            )
            if selected_sector:
                sector_stocks = stock_df[stock_df['sector'] == selected_sector]['ticker'].tolist()[:12]
                selected_tickers = sector_stocks
                st.success(f"✅ {selected_sector}から{len(selected_tickers)}銘柄を選択")
        
        # ウォッチリスト管理
        st.subheader("⭐ ウォッチリスト管理")
        with st.expander("新しいリスト作成"):
            new_watchlist_name = st.text_input("リスト名")
            if st.button("作成"):
                if new_watchlist_name:
                    save_watchlist(new_watchlist_name, [])
                    st.success(f"'{new_watchlist_name}'を作成しました")
                    st.rerun()
        
        # 現在の銘柄をウォッチリストに追加
        if selected_tickers:
            watchlist_names = get_watchlist_names()
            if watchlist_names:
                target_watchlist = st.selectbox(
                    "追加先ウォッチリスト",
                    watchlist_names,
                    key="target_watchlist"
                )
                if st.button("現在の銘柄を追加"):
                    save_watchlist(target_watchlist, selected_tickers)
                    st.success(f"'{target_watchlist}'に追加しました")
    
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
                st.subheader("💰 銘柄別最新価格")
                
                cols = st.columns(4)
                for i, stock_data in enumerate(tickers_data[:12]):
                    with cols[i % 4]:
                        if stock_data['data'] is not None and not stock_data['data'].empty:
                            latest = stock_data['data'].iloc[-1]
                            prev_close = stock_data['data'].iloc[-2]['Close'] if len(stock_data['data']) > 1 else latest['Close']
                            change = latest['Close'] - prev_close
                            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:10]}",
                                value=f"¥{latest['Close']:,.0f}",
                                delta=f"{change_pct:+.2f}%"
                            )
                        else:
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:10]}",
                                value="データなし",
                                delta=None
                            )
            else:
                st.error("チャートの作成に失敗しました")
    else:
        st.info("👈 左のサイドバーから銘柄を選択してください")
        
        # 使い方ガイド
        st.markdown("""
        ### 📋 使い方ガイド
        
        **1. 手動選択**
        - 3802銘柄から自由に最大12銘柄を選択
        - 検索機能で銘柄名・コードから簡単検索
        
        **2. ウォッチリスト**
        - 保存済みのリストから一括選択
        
        **3. 人気銘柄**
        - トヨタ、ソニーなど主要12銘柄を一括選択
        
        **4. 業種別**
        - 選択した業種の銘柄から12銘柄を一括選択
        """)
    
    # フッター
    st.markdown("---")
    st.markdown("💡 各チャートはドラッグで拡大、休日は自動で詰めて表示されます")

if __name__ == "__main__":
    main()

