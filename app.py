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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data():
    """株式データを読み込む"""
    try:
        df = pd.read_csv('data_j.csv')
        # 必要な列を選択してクリーニング
        df = df[['コード', '銘柄名', '市場・商品区分', '33業種区分']].copy()
        df = df.rename(columns={
            'コード': 'code',
            '銘柄名': 'name',
            '市場・商品区分': 'market',
            '33業種区分': 'sector'
        })
        # プライム、スタンダード、グロースのみフィルタ
        df = df[df['market'].isin(['プライム（内国株式）', 'スタンダード（内国株式）', 'グロース（内国株式）'])]
        # コードを文字列に変換し、4桁に統一
        df['code'] = df['code'].astype(str).str.zfill(4)
        df['ticker'] = df['code'] + '.T'
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

@st.cache_data(ttl=300)  # 5分間キャッシュ
def get_stock_data(ticker, period='1mo'):
    """株価データを取得"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        
        # VWAPバンド計算
        df = calculate_vwap_bands(df)
        return df
    except Exception as e:
        st.error(f"株価データの取得エラー ({ticker}): {e}")
        return None

def create_chart(df, ticker, name):
    """トレーディングビュー風チャートを作成"""
    if df is None or df.empty:
        return None
    
    # サブプロットを作成（価格チャートと出来高）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{name} ({ticker})", "出来高"]
    )
    
    # ローソク足チャート
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="価格",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # VWAP
    if 'vwap' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='#ffaa00', width=2)
            ),
            row=1, col=1
        )
        
        # VWAPバンド（1σ）
        if 'vwap_upper_1' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vwap_upper_1'],
                    mode='lines',
                    name='VWAP+1σ',
                    line=dict(color='#888888', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vwap_lower_1'],
                    mode='lines',
                    name='VWAP-1σ',
                    line=dict(color='#888888', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(136,136,136,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # VWAPバンド（2σ）
        if 'vwap_upper_2' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vwap_upper_2'],
                    mode='lines',
                    name='VWAP+2σ',
                    line=dict(color='#cccccc', width=1, dash='dot'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vwap_lower_2'],
                    mode='lines',
                    name='VWAP-2σ',
                    line=dict(color='#cccccc', width=1, dash='dot'),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 出来高
    colors = ['#ff4444' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#00ff88' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="出来高",
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # レイアウト更新
    fig.update_layout(
        title="",
        xaxis_rangeslider_visible=False,
        height=400,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
    
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
    st.markdown('<h1 class="main-header">📈 日本株リアルタイムチャート</h1>', unsafe_allow_html=True)
    
    # データ読み込み
    stock_df = load_stock_data()
    if stock_df.empty:
        st.error("データファイルが見つかりません。data_j.xlsファイルをアップロードしてください。")
        return
    
    # サイドバー
    st.sidebar.header("銘柄選択")
    
    # ウォッチリスト機能
    st.sidebar.subheader("ウォッチリスト")
    watchlist_names = get_watchlist_names()
    
    # ウォッチリスト選択
    if watchlist_names:
        selected_watchlist = st.sidebar.selectbox("保存済みリスト", ["選択してください"] + watchlist_names)
        if selected_watchlist != "選択してください":
            if st.sidebar.button("リスト読み込み"):
                st.session_state.selected_tickers = load_watchlist(selected_watchlist)
                st.rerun()
    
    # 銘柄選択
    search_method = st.sidebar.radio("検索方法", ["銘柄名", "銘柄コード", "業種", "市場"])
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if search_method == "銘柄名":
        companies = stock_df['name'].unique()
        selected_company = st.sidebar.selectbox("銘柄名で選択", ["選択してください"] + sorted(companies))
        if selected_company != "選択してください":
            ticker_info = stock_df[stock_df['name'] == selected_company].iloc[0]
            ticker = ticker_info['ticker']
    
    elif search_method == "銘柄コード":
        code = st.sidebar.text_input("銘柄コード（4桁）", placeholder="例: 7203")
        if code and len(code) == 4:
            ticker_info = stock_df[stock_df['code'] == code]
            if not ticker_info.empty:
                ticker = ticker_info.iloc[0]['ticker']
                selected_company = ticker_info.iloc[0]['name']
            else:
                st.sidebar.error("該当する銘柄が見つかりません")
                ticker = None
        else:
            ticker = None
    
    elif search_method == "業種":
        sectors = stock_df['sector'].unique()
        selected_sector = st.sidebar.selectbox("業種で選択", ["選択してください"] + sorted([s for s in sectors if pd.notna(s)]))
        if selected_sector != "選択してください":
            sector_companies = stock_df[stock_df['sector'] == selected_sector]['name'].tolist()
            selected_company = st.sidebar.selectbox("銘柄選択", ["選択してください"] + sorted(sector_companies))
            if selected_company != "選択してください":
                ticker_info = stock_df[stock_df['name'] == selected_company].iloc[0]
                ticker = ticker_info['ticker']
        else:
            ticker = None
    
    elif search_method == "市場":
        markets = stock_df['market'].unique()
        selected_market = st.sidebar.selectbox("市場で選択", ["選択してください"] + sorted(markets))
        if selected_market != "選択してください":
            market_companies = stock_df[stock_df['market'] == selected_market]['name'].tolist()
            selected_company = st.sidebar.selectbox("銘柄選択", ["選択してください"] + sorted(market_companies))
            if selected_company != "選択してください":
                ticker_info = stock_df[stock_df['name'] == selected_company].iloc[0]
                ticker = ticker_info['ticker']
        else:
            ticker = None
    
    # 銘柄追加
    if st.sidebar.button("銘柄を追加") and 'ticker' in locals() and ticker:
        if ticker not in st.session_state.selected_tickers and len(st.session_state.selected_tickers) < 12:
            st.session_state.selected_tickers.append(ticker)
            st.rerun()
        elif len(st.session_state.selected_tickers) >= 12:
            st.sidebar.warning("最大12銘柄まで選択できます")
    
    # 選択された銘柄表示
    if st.session_state.selected_tickers:
        st.sidebar.subheader("選択中の銘柄")
        for i, ticker in enumerate(st.session_state.selected_tickers):
            stock_info = stock_df[stock_df['ticker'] == ticker]
            if not stock_info.empty:
                name = stock_info.iloc[0]['name']
                col1, col2 = st.sidebar.columns([3, 1])
                col1.write(f"{name} ({ticker[:-2]})")
                if col2.button("削除", key=f"del_{i}"):
                    st.session_state.selected_tickers.remove(ticker)
                    st.rerun()
    
    # ウォッチリスト保存
    st.sidebar.subheader("リスト保存")
    watchlist_name = st.sidebar.text_input("リスト名", placeholder="例: 注目銘柄")
    if st.sidebar.button("保存") and watchlist_name and st.session_state.selected_tickers:
        save_watchlist(watchlist_name, st.session_state.selected_tickers)
        st.sidebar.success("保存しました！")
    
    # 全削除ボタン
    if st.sidebar.button("全銘柄削除"):
        st.session_state.selected_tickers = []
        st.rerun()
    
    # メインエリア
    if not st.session_state.selected_tickers:
        st.info("サイドバーから銘柄を選択してください（最大12銘柄）")
        return
    
    # チャート表示
    st.subheader(f"選択銘柄: {len(st.session_state.selected_tickers)}/12")
    
    # 4列×3行のレイアウト
    rows = 3
    cols = 4
    
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(st.session_state.selected_tickers):
                ticker = st.session_state.selected_tickers[idx]
                stock_info = stock_df[stock_df['ticker'] == ticker]
                
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                    
                    # 株価データ取得
                    with columns[col]:
                        with st.spinner(f"データ取得中... {name}"):
                            df = get_stock_data(ticker)
                            if df is not None:
                                fig = create_chart(df, ticker[:-2], name)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                    
                                    # 最新価格表示
                                    current_price = df['Close'].iloc[-1]
                                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                                    change = current_price - prev_price
                                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                                    
                                    color = "green" if change >= 0 else "red"
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 5px;">
                                        <span style="font-size: 16px; font-weight: bold;">¥{current_price:,.0f}</span><br>
                                        <span style="color: {color}; font-size: 12px;">
                                            {change:+.0f} ({change_pct:+.1f}%)
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error(f"データ取得エラー: {name}")

if __name__ == "__main__":
    main()

