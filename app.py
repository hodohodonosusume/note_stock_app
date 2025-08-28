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
        
        # VWAPバンド計算
        df = calculate_vwap_bands(df)
        return df
    except Exception as e:
        st.error(f"株価データの取得エラー ({ticker}): {e}")
        return None

def create_chart(df, ticker, name, timeframe='1d'):
    """改善されたトレーディングビュー風チャートを作成"""
    if df is None or df.empty:
        return None

    # サブプロットを作成
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{name} ({ticker}) - {timeframe}", ""]
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
            increasing={'line': {'color': '#00D4AA'}, 'fillcolor': '#00D4AA'},
            decreasing={'line': {'color': '#FF6B6B'}, 'fillcolor': '#FF6B6B'},
            showlegend=False
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
                line=dict(color='#FFD93D', width=2),
                showlegend=False,
                hovertemplate='<b>VWAP</b><br>' +
                             '日時: %{x}<br>' +
                             '価格: ¥%{y:,.0f}<br>' +
                             '<extra></extra>'
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
                line=dict(color='rgba(255, 107, 107, 0.6)', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['vwap_lower_2'],
                mode='lines',
                name='VWAP-2σ',
                line=dict(color='rgba(255, 107, 107, 0.6)', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.05)',
                showlegend=False,
                hoverinfo='skip'
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
                line=dict(color='rgba(106, 90, 205, 0.8)', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['vwap_lower_1'],
                mode='lines',
                name='VWAP-1σ',
                line=dict(color='rgba(106, 90, 205, 0.8)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(106, 90, 205, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # 出来高バー
    colors = ['#FF6B6B' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#00D4AA' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="出来高",
            marker_color=colors,
            opacity=0.7,
            showlegend=False,
            hovertemplate='<b>出来高</b><br>' +
                         '日時: %{x}<br>' +
                         '出来高: %{y:,}<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>{name} ({ticker})</b>",
            font=dict(size=20, color='#2C3E50'),
            x=0.5
        ),
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=12, family="Arial, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
        dragmode='pan',  # ドラッグで期間変更を有効化
        showlegend=False  # 凡例を無効化
    )

    # X軸の設定
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(128,128,128,0.2)',
        showspikes=True,
        spikecolor="orange",
        spikesnap="cursor",
        spikemode="across"
    )
    
    # Y軸の設定
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(128,128,128,0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(128,128,128,0.2)',
        row=2, col=1
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
        
        # 銘柄検索
        search_term = st.text_input("🔍 銘柄検索", placeholder="銘柄名またはコードを入力")
        
        # 銘柄一覧フィルタリング
        filtered_df = stock_df.copy()
        if search_term:
            filtered_df = stock_df[
                stock_df['name'].str.contains(search_term, na=False, case=False) |
                stock_df['code'].str.contains(search_term, na=False, case=False)
            ]
        
        # 市場区分フィルタ
        markets = st.multiselect(
            "🏪 市場区分",
            ['プライム（内国株式）', 'スタンダード（内国株式）', 'グロース（内国株式）'],
            default=['プライム（内国株式）']
        )
        
        if markets:
            filtered_df = filtered_df[filtered_df['market'].isin(markets)]
        
        # 業種フィルタ
        sectors = filtered_df['sector'].unique()
        selected_sectors = st.multiselect(
            "🏭 業種",
            sorted([s for s in sectors if pd.notna(s)]),
            default=[]
        )
        
        if selected_sectors:
            filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
        
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
        
        # ウォッチリスト機能
        st.subheader("⭐ ウォッチリスト")
        
        # 既存のウォッチリスト選択
        watchlist_names = get_watchlist_names()
        selected_watchlist = None
        
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "保存済みリスト",
                [""] + watchlist_names
            )
        
        # 新しいウォッチリスト作成
        with st.expander("新しいリスト作成"):
            new_watchlist_name = st.text_input("リスト名")
            if st.button("作成"):
                if new_watchlist_name:
                    save_watchlist(new_watchlist_name, [])
                    st.success(f"'{new_watchlist_name}'を作成しました")
                    st.rerun()
    
    # メインエリア
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📊 銘柄一覧")
        
        # ウォッチリストから銘柄を表示
        watchlist_tickers = []
        if selected_watchlist:
            watchlist_tickers = load_watchlist(selected_watchlist)
            if watchlist_tickers:
                st.write(f"**{selected_watchlist}** の銘柄:")
                watchlist_df = stock_df[stock_df['ticker'].isin(watchlist_tickers)]
                for _, row in watchlist_df.iterrows():
                    if st.button(f"{row['code']} {row['name']}", key=f"wl_{row['ticker']}"):
                        st.session_state['selected_ticker'] = row['ticker']
                        st.session_state['selected_name'] = row['name']
                st.divider()
        
        # フィルタされた銘柄表示
        st.write("**検索結果:**")
        display_count = min(20, len(filtered_df))
        
        for _, row in filtered_df.head(display_count).iterrows():
            col_btn, col_add = st.columns([4, 1])
            
            with col_btn:
                if st.button(
                    f"{row['code']} {row['name'][:20]}{'...' if len(row['name']) > 20 else ''}",
                    key=f"btn_{row['ticker']}"
                ):
                    st.session_state['selected_ticker'] = row['ticker']
                    st.session_state['selected_name'] = row['name']
            
            with col_add:
                if selected_watchlist and st.button("➕", key=f"add_{row['ticker']}"):
                    current_list = load_watchlist(selected_watchlist)
                    if row['ticker'] not in current_list:
                        current_list.append(row['ticker'])
                        save_watchlist(selected_watchlist, current_list)
                        st.success("追加完了")
                        st.rerun()
        
        if len(filtered_df) > display_count:
            st.info(f"他 {len(filtered_df) - display_count} 銘柄")
    
    with col2:
        # 選択された銘柄のチャート表示
        if 'selected_ticker' in st.session_state:
            ticker = st.session_state['selected_ticker']
            name = st.session_state['selected_name']
            
            st.subheader(f"📈 チャート - {selected_timeframe}")
            
            # データ取得とチャート表示
            with st.spinner("チャートを読み込み中..."):
                stock_data = get_stock_data(ticker, period, interval)
                
                if stock_data is not None and not stock_data.empty:
                    chart = create_chart(stock_data, ticker, name, selected_timeframe)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # 最新の株価情報
                        latest = stock_data.iloc[-1]
                        prev_close = stock_data.iloc[-2]['Close'] if len(stock_data) > 1 else latest['Close']
                        change = latest['Close'] - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric("現在値", f"¥{latest['Close']:,.0f}", f"{change:+,.0f}")
                        
                        with col_m2:
                            st.metric("変動率", f"{change_pct:+.2f}%")
                        
                        with col_m3:
                            st.metric("出来高", f"{latest['Volume']:,}")
                        
                        with col_m4:
                            if 'vwap' in stock_data.columns:
                                vwap = latest['vwap']
                                st.metric("VWAP", f"¥{vwap:,.0f}")
                else:
                    st.error("データの取得に失敗しました")
        else:
            st.info("左側から銘柄を選択してください")
    
    # フッター
    st.markdown("---")
    st.markdown("💡 チャートはドラッグで期間変更、ピンチで拡大縮小が可能です")

if __name__ == "__main__":
    main()

