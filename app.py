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
        
        # 連番でX軸を作成（休日の空白を詰める）
        df['x_axis'] = range(len(df))
        
        # VWAPバンド計算
        df = calculate_vwap_bands(df)
        return df
    except Exception as e:
        st.error(f"株価データの取得エラー ({ticker}): {e}")
        return None

def get_multiple_stock_data(tickers, period='1mo', interval='1d'):
    """複数銘柄の株価データを一括取得"""
    stock_data_list = []
    for ticker in tickers:
        data = get_stock_data(ticker, period, interval)
        stock_data_list.append(data)
    return stock_data_list

def create_multi_chart(stock_data_list, tickers, names, timeframe='1d'):
    """4列×3行の12銘柄チャートを作成"""
    if not stock_data_list or all(df is None for df in stock_data_list):
        return None

    # サブプロットを作成（4列×3行）
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{name} ({ticker})" for ticker, name in zip(tickers, names)],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    for i, (df, ticker, name) in enumerate(zip(stock_data_list, tickers, names)):
        if df is None or df.empty:
            continue
            
        row = (i // 4) + 1
        col = (i % 4) + 1

        # ローソク足チャート
        fig.add_trace(
            go.Candlestick(
                x=df['x_axis'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=f"{name}",
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
                    x=df['x_axis'],
                    y=df['vwap'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='#FFD93D', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（1σ）
        if 'vwap_upper_1' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['x_axis'],
                    y=df['vwap_upper_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.4)', width=0.5, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['x_axis'],
                    y=df['vwap_lower_1'],
                    mode='lines',
                    line=dict(color='rgba(106, 90, 205, 0.4)', width=0.5, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(106, 90, 205, 0.05)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（2σ）
        if 'vwap_upper_2' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['x_axis'],
                    y=df['vwap_upper_2'],
                    mode='lines',
                    line=dict(color='rgba(255, 107, 107, 0.3)', width=0.5, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['x_axis'],
                    y=df['vwap_lower_2'],
                    mode='lines',
                    line=dict(color='rgba(255, 107, 107, 0.3)', width=0.5, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 107, 0.02)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # X軸のラベルを日付に設定（最初と最後と中央のみ）
        if len(df) > 0:
            tick_vals = [0, len(df)//2, len(df)-1]
            tick_texts = [
                df.index[0].strftime('%m/%d'),
                df.index[len(df)//2].strftime('%m/%d'),
                df.index[-1].strftime('%m/%d')
            ]
            
            fig.update_xaxes(
                tickvals=tick_vals,
                ticktext=tick_texts,
                row=row, col=col
            )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>日本株マルチチャート - {timeframe}</b>",
            font=dict(size=18, color='#2C3E50'),
            x=0.5
        ),
        height=800,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=9, family="Arial, sans-serif"),
        margin=dict(l=10, r=10, t=60, b=20),
        dragmode='pan',
        showlegend=False
    )

    # 全てのX軸とY軸のグリッド設定
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor='rgba(128,128,128,0.2)'
    )

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
        <h1>📈 日本株リアルタイムチャート</h1>
        <p>プライム・スタンダード・グロース市場対応</p>
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
        
        # 銘柄選択
        st.subheader("📊 表示銘柄選択")
        
        # ウォッチリスト機能
        watchlist_names = get_watchlist_names()
        selected_watchlist = None
        
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "保存済みウォッチリスト",
                [""] + watchlist_names
            )
        
        # デフォルト銘柄（主要12銘柄）
        default_tickers = [
            '6501.T',  # 日立製作所
            '7203.T',  # トヨタ自動車
            '9432.T',  # NTT
            '6758.T',  # ソニーグループ
            '9984.T',  # ソフトバンクグループ
            '8058.T',  # 三菱商事
            '8035.T',  # 東京エレクトロン
            '6861.T',  # キーエンス
            '8591.T',  # オリックス
            '4568.T',  # 第一三共
            '6098.T',  # リクルートホールディングス
            '4519.T'   # 中外製薬
        ]
        
        display_tickers = default_tickers
        
        # ウォッチリストが選択されている場合
        if selected_watchlist:
            watchlist_tickers = load_watchlist(selected_watchlist)
            if len(watchlist_tickers) >= 12:
                display_tickers = watchlist_tickers[:12]
            elif watchlist_tickers:
                # 不足分をデフォルトで補完
                display_tickers = watchlist_tickers + default_tickers[:12-len(watchlist_tickers)]
        
        # 表示予定の銘柄名取得
        display_names = []
        for ticker in display_tickers:
            match = stock_df[stock_df['ticker'] == ticker]
            if not match.empty:
                display_names.append(match.iloc[0]['name'])
            else:
                display_names.append(ticker.replace('.T', ''))
        
        # 表示銘柄リスト
        st.write("**表示予定銘柄:**")
        for i, (ticker, name) in enumerate(zip(display_tickers, display_names)):
            st.write(f"{i+1:2d}. {ticker.replace('.T', '')} {name[:15]}...")
        
        # 新しいウォッチリスト作成
        with st.expander("新しいウォッチリスト作成"):
            new_watchlist_name = st.text_input("リスト名")
            
            # 銘柄検索・追加
            search_term = st.text_input("銘柄検索", placeholder="銘柄名またはコードを入力")
            
            if search_term:
                filtered_df = stock_df[
                    stock_df['name'].str.contains(search_term, na=False, case=False) |
                    stock_df['code'].str.contains(search_term, na=False, case=False)
                ].head(10)
                
                selected_stocks = st.multiselect(
                    "追加する銘柄を選択",
                    options=[f"{row['code']} {row['name']}" for _, row in filtered_df.iterrows()],
                    format_func=lambda x: x
                )
                
                if st.button("ウォッチリスト作成"):
                    if new_watchlist_name and selected_stocks:
                        # 選択された銘柄からティッカーを抽出
                        new_tickers = []
                        for stock in selected_stocks:
                            code = stock.split(' ')[0]
                            new_tickers.append(f"{code}.T")
                        
                        save_watchlist(new_watchlist_name, new_tickers)
                        st.success(f"'{new_watchlist_name}'を作成しました")
                        st.rerun()
                    else:
                        st.warning("リスト名と銘柄を選択してください")
    
    # メインエリア
    st.subheader(f"📊 マルチチャート表示 - {selected_timeframe}")
    
    # データ取得とチャート表示
    with st.spinner("チャートを読み込み中..."):
        stock_data_list = get_multiple_stock_data(display_tickers, period, interval)
        
        if stock_data_list and any(df is not None for df in stock_data_list):
            chart = create_multi_chart(stock_data_list, display_tickers, display_names, selected_timeframe)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
                
                # サマリー情報
                col1, col2, col3 = st.columns(3)
                
                successful_count = sum(1 for df in stock_data_list if df is not None)
                
                with col1:
                    st.metric("表示銘柄数", f"{successful_count}/12")
                
                with col2:
                    st.metric("データ期間", period)
                
                with col3:
                    st.metric("更新間隔", "5分毎")
                
                # 個別銘柄情報
                if st.checkbox("📈 個別銘柄詳細"):
                    cols = st.columns(4)
                    for i, (df, ticker, name) in enumerate(zip(stock_data_list[:12], display_tickers, display_names)):
                        if df is not None and not df.empty:
                            with cols[i % 4]:
                                latest = df.iloc[-1]
                                prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Close']
                                change = latest['Close'] - prev_close
                                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                                
                                st.metric(
                                    f"{name[:10]}...",
                                    f"¥{latest['Close']:,.0f}",
                                    f"{change_pct:+.1f}%"
                                )
        else:
            st.error("データの取得に失敗しました")
    
    # フッター
    st.markdown("---")
    st.markdown("💡 チャートはドラッグで期間変更、ピンチで拡大縮小が可能です")

if __name__ == "__main__":
    main()
