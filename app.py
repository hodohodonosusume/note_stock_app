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
.selected-stock {
    background: #e8f5e8;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.2rem 0;
    border-left: 3px solid #28a745;
}
.watchlist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem;
    margin: 0.2rem 0;
    background: #f8f9fa;
    border-radius: 5px;
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
        df['display'] = df['code'] + ' ' + df['name']
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

def create_multi_chart(tickers_data, timeframe='1d'):
    """12銘柄の4列×3行チャートを作成（90日データで最新20日表示）"""
    if not tickers_data:
        return None

    # 4列×3行のサブプロット作成
    fig = make_subplots(
        rows=3, cols=4,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[f"{data['name'][:15]} ({data['code']})" for data in tickers_data[:12]]
    )

    for i, stock_data in enumerate(tickers_data[:12]):
        if stock_data['data'] is None or stock_data['data'].empty:
            continue
        
        df = stock_data['data']
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        # 最新20日分を表示用に取得
        display_df = df.tail(20) if len(df) > 20 else df
        
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
        if 'vwap' in display_df.columns:
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
        if 'vwap_upper_2' in display_df.columns:
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
                    fillcolor='rgba(255, 107, 107, 0.05)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # VWAPバンド（1σ）
        if 'vwap_upper_1' in display_df.columns:
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
                    fillcolor='rgba(106, 90, 205, 0.1)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>📈 日本株マルチチャート - {timeframe} (90日データ/最新20日表示)</b>",
            font=dict(size=18, color='#2C3E50'),
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
    # セッション状態の初期化
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""

    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>📈 日本株マルチチャート</h1>
        <p>12銘柄同時表示 - 3802銘柄対応・90日データ・ウォッチリスト機能</p>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    stock_df = load_stock_data()
    
    if stock_df.empty:
        st.error("株式データの読み込みに失敗しました。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 銘柄選択")
        
        # 現在の選択銘柄表示
        st.subheader(f"📊 選択中の銘柄 ({len(st.session_state.selected_stocks)}/12)")
        
        if st.session_state.selected_stocks:
            for i, selected in enumerate(st.session_state.selected_stocks):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="selected-stock">{selected}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.selected_stocks.pop(i)
                        st.rerun()
        else:
            st.info("銘柄が選択されていません")
        
        if st.button("🗑️ 全て削除"):
            st.session_state.selected_stocks = []
            st.rerun()
        
        st.divider()
        
        # 銘柄検索・追加
        st.subheader("🔍 銘柄検索・追加")
        
        search_term = st.text_input(
            "銘柄名またはコードで検索", 
            value=st.session_state.search_term,
            placeholder="例: トヨタ, 7203"
        )
        st.session_state.search_term = search_term
        
        # 検索結果表示
        if search_term:
            filtered_df = stock_df[
                stock_df['name'].str.contains(search_term, na=False, case=False) |
                stock_df['code'].str.contains(search_term, na=False, case=False)
            ].head(20)
            
            st.write(f"**検索結果: {len(filtered_df)}件**")
            
            for _, row in filtered_df.iterrows():
                display_name = row['display']
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"{row['code']} {row['name'][:20]}{'...' if len(row['name']) > 20 else ''}")
                
                with col2:
                    if display_name not in st.session_state.selected_stocks:
                        if len(st.session_state.selected_stocks) < 12:
                            if st.button("➕", key=f"add_{row['ticker']}"):
                                st.session_state.selected_stocks.append(display_name)
                                st.rerun()
                        else:
                            st.write("🔒")
                    else:
                        st.write("✅")
        
        st.divider()
        
        # 時間足設定
        st.subheader("⏰ 時間足設定")
        timeframe_options = {
            '5分足': ('5d', '5m'),
            '日足': ('3mo', '1d'),
            '週足': ('6mo', '1wk'),
            '月足': ('2y', '1mo')
        }
        
        selected_timeframe = st.selectbox(
            "時間足",
            options=list(timeframe_options.keys()),
            index=1
        )
        
        period, interval = timeframe_options[selected_timeframe]
        
        st.divider()
        
        # ウォッチリスト機能
        st.subheader("⭐ ウォッチリスト管理")
        
        # ウォッチリスト作成
        with st.expander("📝 新規作成"):
            new_watchlist_name = st.text_input("リスト名")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("空のリスト作成"):
                    if new_watchlist_name:
                        save_watchlist(new_watchlist_name, [])
                        st.success(f"'{new_watchlist_name}'を作成しました")
                        st.rerun()
            
            with col2:
                if st.button("現在の選択で作成"):
                    if new_watchlist_name and st.session_state.selected_stocks:
                        save_watchlist(new_watchlist_name, st.session_state.selected_stocks)
                        st.success(f"'{new_watchlist_name}'を作成しました")
                        st.rerun()
        
        # 既存ウォッチリスト管理
        watchlist_names = get_watchlist_names()
        if watchlist_names:
            st.write("**既存のウォッチリスト:**")
            
            for watchlist_name in watchlist_names:
                with st.expander(f"📋 {watchlist_name}"):
                    watchlist_items = load_watchlist(watchlist_name)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📥 読込", key=f"load_{watchlist_name}"):
                            st.session_state.selected_stocks = watchlist_items[:12]
                            st.rerun()
                    
                    with col2:
                        if st.button("💾 上書", key=f"save_{watchlist_name}"):
                            save_watchlist(watchlist_name, st.session_state.selected_stocks)
                            st.success("保存しました")
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️ 削除", key=f"delete_{watchlist_name}"):
                            os.remove(f'watchlists/{watchlist_name}.json')
                            st.success("削除しました")
                            st.rerun()
                    
                    if watchlist_items:
                        st.write(f"銘柄数: {len(watchlist_items)}")
                        for item in watchlist_items[:5]:
                            st.write(f"• {item}")
                        if len(watchlist_items) > 5:
                            st.write(f"...他 {len(watchlist_items) - 5} 銘柄")
    
    # メインエリア
    if st.session_state.selected_stocks:
        st.subheader(f"📊 マルチチャート表示 - {selected_timeframe}")
        
        with st.spinner("チャートを読み込み中..."):
            # 各銘柄のデータを取得
            tickers_data = []
            progress_bar = st.progress(0)
            
            for i, selected_stock in enumerate(st.session_state.selected_stocks):
                # display形式から code を抽出
                code = selected_stock.split(' ')[0]
                ticker = code + '.T'
                
                stock_info = stock_df[stock_df['code'] == code]
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                else:
                    name = selected_stock
                
                stock_data = get_stock_data(ticker, period, interval)
                
                tickers_data.append({
                    'ticker': ticker,
                    'name': name,
                    'code': code,
                    'data': stock_data
                })
                
                progress_bar.progress((i + 1) / len(st.session_state.selected_stocks))
            
            progress_bar.empty()
            
            # マルチチャート作成
            multi_chart = create_multi_chart(tickers_data, selected_timeframe)
            
            if multi_chart:
                st.plotly_chart(multi_chart, use_container_width=True)
                
                # 銘柄一覧と最新価格表示
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
                
                # データ期間情報
                st.info("💡 90日分のデータを取得し、最新20本を表示。チャートをドラッグして過去データも確認できます")
            else:
                st.error("チャートの作成に失敗しました")
    else:
        st.info("左のサイドバーから銘柄を選択してください（最大12銘柄）")
    
    # フッター
    st.markdown("---")
    st.markdown("💡 **機能**: 検索→追加→ウォッチリスト保存→マルチチャート表示 | **データ**: 90日分取得・20日表示・ドラッグ可能")

if __name__ == "__main__":
    main()

