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
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.stock-item {
    padding: 0.5rem;
    margin: 0.2rem 0;
    border-radius: 5px;
    border: 1px solid #ddd;
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
    """PineScript準拠のVWAP バンドを計算"""
    if len(df) < period:
        return df
    
    # 典型価格 (HLC3)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # 価格×出来高
    df['price_volume'] = df['typical_price'] * df['Volume']
    
    # 指定期間のVWAP計算（移動平均ベース）
    sum_pv = df['price_volume'].rolling(window=period).sum()
    sum_vol = df['Volume'].rolling(window=period).sum()
    df['vwap'] = sum_pv / sum_vol
    
    # VWAP基準の偏差
    df['deviation'] = df['typical_price'] - df['vwap']
    df['squared_dev'] = df['deviation'] ** 2
    
    # 加重標準偏差計算
    df['weighted_squared_dev'] = df['squared_dev'] * df['Volume']
    sum_weighted_squared_dev = df['weighted_squared_dev'].rolling(window=period).sum()
    df['variance'] = sum_weighted_squared_dev / sum_vol
    df['std_dev'] = np.sqrt(df['variance'])
    
    # バンド計算
    df['vwap_upper_1'] = df['vwap'] + df['std_dev']
    df['vwap_lower_1'] = df['vwap'] - df['std_dev']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['std_dev']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['std_dev']
    
    return df

@st.cache_data(ttl=300)
def get_stock_data(ticker, period='3mo', interval='1d'):
    """株価データを取得（90日分取得）"""
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
    """12銘柄の4列×3行チャート、20日表示で過去90日スクロール可能"""
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

    colors = ['#00D4AA', '#FF6B6B', '#FFD93D', '#6A5ACD', '#FF69B4', '#32CD32',
              '#FF4500', '#1E90FF', '#DC143C', '#00CED1', '#9370DB', '#FFA500']

    for i, stock_data in enumerate(tickers_data[:12]):
        if stock_data['data'] is None or stock_data['data'].empty:
            continue
        
        df = stock_data['data']
        
        # 最新20日分を表示用に切り出し
        display_df = df.tail(display_days)
        
        row = (i // 4) + 1
        col = (i % 4) + 1
        color = colors[i % len(colors)]
        
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
                    line=dict(color='#FFD93D', width=2),
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

        # VWAPバンド（2σ）- 新機能！
        if 'vwap_upper_2' in display_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=display_df['vwap_upper_2'],
                    mode='lines',
                    line=dict(color='rgba(255, 107, 107, 0.8)', width=1, dash='dot'),
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
                    line=dict(color='rgba(255, 107, 107, 0.8)', width=1, dash='dot'),
                    fillcolor='rgba(255, 107, 107, 0.05)',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    # レイアウト更新
    fig.update_layout(
        title=dict(
            text=f"<b>📈 日本株マルチチャート (最新{display_days}日表示) - {timeframe}</b>",
            font=dict(size=18, color='#2C3E50'),
            x=0.5
        ),
        height=900,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=9, family="Arial, sans-serif"),
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
        <h1>📈 日本株マルチチャート (改良版)</h1>
        <p>12銘柄同時表示・90日データ・VWAP2σバンド対応</p>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    stock_df = load_stock_data()
    
    if stock_df.empty:
        st.error("株式データの読み込みに失敗しました。")
        return
    
    # セッション状態の初期化
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []
    if 'current_watchlist' not in st.session_state:
        st.session_state.current_watchlist = ""
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 銘柄選択 & 設定")
        
        # === 銘柄検索・選択セクション ===
        st.subheader("🔍 銘柄検索・選択")
        
        # 検索フィルタ
        search_term = st.text_input("🔍 銘柄名/コード検索", 
                                  placeholder="例：トヨタ、7203",
                                  key="search_input")
        
        # 市場区分フィルタ
        markets = st.multiselect(
            "🏪 市場区分",
            ['プライム（内国株式）', 'スタンダード（内国株式）', 'グロース（内国株式）'],
            default=['プライム（内国株式）'],
            key="market_filter"
        )
        
        # 業種フィルタ
        sectors = sorted([s for s in stock_df['sector'].unique() if pd.notna(s)])
        selected_sectors = st.multiselect(
            "🏭 業種",
            sectors,
            default=[],
            key="sector_filter"
        )
        
        # データフィルタリング
        filtered_df = stock_df.copy()
        if markets:
            filtered_df = filtered_df[filtered_df['market'].isin(markets)]
        if selected_sectors:
            filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
        if search_term:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(search_term, na=False, case=False) |
                filtered_df['code'].str.contains(search_term, na=False, case=False)
            ]
        
        # 選択可能な銘柄リストを作成
        available_options = []
        for _, row in filtered_df.iterrows():
            available_options.append(f"{row['code']} - {row['name']}")
        
        # 現在選択されている銘柄（最大12個）
        st.markdown("---")
        st.subheader("📊 選択中の銘柄")
        
        # 現在の選択を表示
        current_selection = st.multiselect(
            "選択銘柄（最大12個）",
            available_options,
            default=[opt for opt in available_options if any(stock in opt for stock in st.session_state.selected_stocks)],
            max_selections=12,
            key="stock_multiselect"
        )
        
        # セッション状態を更新
        st.session_state.selected_stocks = [opt.split(' - ')[0] + '.T' for opt in current_selection]
        
        # クリア・リセットボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 全てクリア"):
                st.session_state.selected_stocks = []
                st.rerun()
        
        with col2:
            if st.button("🔄 リフレッシュ"):
                st.cache_data.clear()
                st.rerun()
        
        # === ウォッチリスト管理 ===
        st.markdown("---")
        st.subheader("⭐ ウォッチリスト管理")
        
        # 既存ウォッチリスト選択
        watchlist_names = get_watchlist_names()
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "保存済みリスト",
                [""] + watchlist_names,
                key="watchlist_selector"
            )
            
            if selected_watchlist and selected_watchlist != st.session_state.current_watchlist:
                # ウォッチリスト読み込み
                watchlist_tickers = load_watchlist(selected_watchlist)
                st.session_state.selected_stocks = watchlist_tickers[:12]  # 最大12個
                st.session_state.current_watchlist = selected_watchlist
                st.rerun()
            
            # ウォッチリスト操作ボタン
            if selected_watchlist:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 現在の選択で上書き"):
                        save_watchlist(selected_watchlist, st.session_state.selected_stocks)
                        st.success(f"'{selected_watchlist}'を更新しました")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ リスト削除"):
                        try:
                            os.remove(f'watchlists/{selected_watchlist}.json')
                            st.success(f"'{selected_watchlist}'を削除しました")
                            st.session_state.current_watchlist = ""
                            time.sleep(1)
                            st.rerun()
                        except:
                            st.error("削除に失敗しました")
        
        # 新しいウォッチリスト作成
        with st.expander("🆕 新しいリスト作成"):
            new_watchlist_name = st.text_input("リスト名", key="new_watchlist_name")
            if st.button("作成 & 現在の選択を保存"):
                if new_watchlist_name and st.session_state.selected_stocks:
                    save_watchlist(new_watchlist_name, st.session_state.selected_stocks)
                    st.success(f"'{new_watchlist_name}'を作成しました")
                    st.session_state.current_watchlist = new_watchlist_name
                    time.sleep(1)
                    st.rerun()
                elif not new_watchlist_name:
                    st.error("リスト名を入力してください")
                else:
                    st.error("銘柄を選択してください")
        
        # === 表示設定 ===
        st.markdown("---")
        st.subheader("⏰ 表示設定")
        
        timeframe_options = {
            '日足': ('3mo', '1d'),  # 90日分取得
            '週足': ('1y', '1wk'),
            '月足': ('2y', '1mo')
        }
        
        selected_timeframe = st.selectbox(
            "時間足",
            options=list(timeframe_options.keys()),
            index=0,
            key="timeframe_selector"
        )
        
        # 表示日数設定
        display_days = st.slider(
            "チャート表示日数",
            min_value=10,
            max_value=90,
            value=20,
            step=5,
            help="チャートに表示する直近の日数"
        )
        
        period, interval = timeframe_options[selected_timeframe]
    
    # === メインエリア ===
    if st.session_state.selected_stocks:
        st.subheader(f"📊 選択銘柄数: {len(st.session_state.selected_stocks)}/12")
        
        # 選択された銘柄の表示
        selected_names = []
        for ticker in st.session_state.selected_stocks:
            code = ticker.replace('.T', '')
            stock_info = stock_df[stock_df['code'] == code]
            if not stock_info.empty:
                selected_names.append(f"{code}: {stock_info.iloc[0]['name']}")
            else:
                selected_names.append(f"{code}: 不明")
        
        st.write("**選択中:** " + " | ".join(selected_names))
        
        with st.spinner("チャートを読み込み中..."):
            # 各銘柄のデータを取得
            tickers_data = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(st.session_state.selected_stocks):
                code = ticker.replace('.T', '')
                stock_info = stock_df[stock_df['code'] == code]
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                else:
                    name = code
                
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
            multi_chart = create_multi_chart(tickers_data, selected_timeframe, display_days)
            
            if multi_chart:
                st.plotly_chart(multi_chart, use_container_width=True)
                
                # チャート情報
                st.info(f"💡 **表示情報**: 最新{display_days}日分を表示中 | 過去90日分のデータを取得済み | ドラッグで拡大・移動可能")
                
                # 銘柄別最新価格（簡潔版）
                st.subheader("💰 最新価格")
                
                cols = st.columns(min(4, len(tickers_data)))
                for i, stock_data in enumerate(tickers_data):
                    with cols[i % len(cols)]:
                        if stock_data['data'] is not None and not stock_data['data'].empty:
                            latest = stock_data['data'].iloc[-1]
                            prev_close = stock_data['data'].iloc[-2]['Close'] if len(stock_data['data']) > 1 else latest['Close']
                            change = latest['Close'] - prev_close
                            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            # VWAPとの比較
                            vwap_diff = ""
                            if 'vwap' in stock_data['data'].columns:
                                vwap = latest['vwap']
                                vwap_ratio = ((latest['Close'] - vwap) / vwap * 100) if vwap != 0 else 0
                                vwap_diff = f"VWAP比: {vwap_ratio:+.1f}%"
                            
                            st.metric(
                                label=f"{stock_data['code']}",
                                value=f"¥{latest['Close']:,.0f}",
                                delta=f"{change_pct:+.2f}%",
                                help=vwap_diff
                            )
                        else:
                            st.metric(
                                label=f"{stock_data['code']}",
                                value="データなし",
                                delta=None
                            )
            else:
                st.error("チャートの作成に失敗しました")
    else:
        # 銘柄未選択時の案内
        st.info("👈 左側のサイドバーから銘柄を選択してください")
        
        # おすすめ銘柄の表示
        st.subheader("🌟 おすすめ銘柄（クイック選択）")
        
        popular_stocks = {
            '大型株': ['7203.T', '6758.T', '8306.T', '6861.T'],  # トヨタ、ソニー、三菱UFJ、キーエンス
            'IT関連': ['9984.T', '4689.T', '3659.T', '4385.T'],  # ソフトバンクG、GMO、ネクソン、メルカリ
            '金融': ['8316.T', '8411.T', '8001.T', '8058.T']     # 三井住友FG、みずほFG、伊藤忠、三菱商事
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (category, tickers) in enumerate(popular_stocks.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{category}**")
                if st.button(f"{category}を選択", key=f"quick_{category}"):
                    st.session_state.selected_stocks = tickers
                    st.rerun()
    
    # フッター
    st.markdown("---")
    st.markdown("""
    **📈 機能一覧:**
    - ✅ 3802銘柄から最大12個選択
    - ✅ 休日を詰めた表示
    - ✅ 90日分データ取得・20日表示
    - ✅ VWAPバンド（1σ・2σ対応）
    - ✅ ウォッチリスト保存・読み込み
    - ✅ ドラッグ・ズーム対応
    """)

if __name__ == "__main__":
    main()
