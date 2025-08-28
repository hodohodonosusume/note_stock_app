import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(
    page_title="📈 Stock Chart Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """データを読み込む"""
    try:
        df = pd.read_csv('data_j.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 休日（土日）を削除
        df = df[df['Date'].dt.dayofweek < 5].copy()
        
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return None

def resample_data(df, timeframe):
    """データを指定された時間軸にリサンプル"""
    if timeframe == "5分足":
        # 5分足の場合は元データをそのまま使用（実際のアプリでは5分足データが必要）
        return df
    
    df_resampled = df.set_index('Date')
    
    if timeframe == "日足":
        rule = 'D'
    elif timeframe == "週足":
        rule = 'W'
    elif timeframe == "月足":
        rule = 'M'
    else:
        rule = 'D'
    
    # OHLCV データのリサンプル
    resampled = df_resampled.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'VWAP': 'mean'
    }).dropna()
    
    return resampled.reset_index()

def calculate_vwap_bands(df, window=20):
    """VWAPバンドを計算"""
    df = df.copy()
    
    # 移動平均VWAP
    df['VWAP_MA'] = df['VWAP'].rolling(window=window).mean()
    
    # 標準偏差
    df['VWAP_std'] = df['VWAP'].rolling(window=window).std()
    
    # 1σ、2σバンド
    df['VWAP_upper_1sigma'] = df['VWAP_MA'] + df['VWAP_std']
    df['VWAP_lower_1sigma'] = df['VWAP_MA'] - df['VWAP_std']
    df['VWAP_upper_2sigma'] = df['VWAP_MA'] + 2 * df['VWAP_std']
    df['VWAP_lower_2sigma'] = df['VWAP_MA'] - 2 * df['VWAP_std']
    
    return df

def create_chart(df, timeframe):
    """インタラクティブチャートを作成"""
    
    # VWAPバンドを計算
    df = calculate_vwap_bands(df)
    
    # サブプロットを作成（価格チャート + 出来高）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=None,
        row_width=[0.7, 0.3]
    )
    
    # カラーパレット
    colors = {
        'candle_up': '#26A69A',
        'candle_down': '#EF5350',
        'vwap': '#FF6B35',
        'vwap_1sigma': '#4ECDC4',
        'vwap_2sigma': '#95E1D3',
        'volume': '#A8DADC'
    }
    
    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="",
            increasing_line_color=colors['candle_up'],
            decreasing_line_color=colors['candle_down'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWAP 2σバンド（薄い色）
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['VWAP_upper_2sigma'],
            mode='lines',
            line=dict(color=colors['vwap_2sigma'], width=1, dash='dot'),
            name="VWAP +2σ",
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['VWAP_lower_2sigma'],
            mode='lines',
            line=dict(color=colors['vwap_2sigma'], width=1, dash='dot'),
            name="VWAP -2σ",
            fill='tonexty',
            fillcolor=f"rgba(149, 225, 211, 0.1)",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWAP 1σバンド
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['VWAP_upper_1sigma'],
            mode='lines',
            line=dict(color=colors['vwap_1sigma'], width=1.5, dash='dash'),
            name="VWAP +1σ",
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['VWAP_lower_1sigma'],
            mode='lines',
            line=dict(color=colors['vwap_1sigma'], width=1.5, dash='dash'),
            name="VWAP -1σ",
            fill='tonexty',
            fillcolor=f"rgba(78, 205, 196, 0.15)",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['VWAP'],
            mode='lines',
            line=dict(color=colors['vwap'], width=2),
            name="VWAP",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 出来高
    colors_volume = [colors['candle_up'] if close >= open else colors['candle_down'] 
                    for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color=colors_volume,
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=f"📈 Stock Price Chart ({timeframe})",
            x=0.5,
            font=dict(size=24, color='#2E86AB', family="Arial, sans-serif")
        ),
        xaxis_title="",
        yaxis_title="Price (¥)",
        template="plotly_white",
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=12, color="#333333")
    )
    
    # Y軸の設定
    fig.update_yaxes(
        title_text="Price (¥)",
        title_font=dict(size=14, color='#2E86AB'),
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        title_font=dict(size=14, color='#2E86AB'),
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )
    
    # X軸の設定
    fig.update_xaxes(
        gridcolor='rgba(128, 128, 128, 0.2)',
        title_font=dict(size=14, color='#2E86AB'),
        row=2, col=1
    )
    
    # インタラクティブ機能の設定
    fig.update_layout(
        dragmode='pan',  # パン操作を有効
        scrollZoom=True,  # スクロールズームを有効
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=30, label="30D", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(count=180, label="6M", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ],
                bgcolor='rgba(46, 134, 171, 0.1)',
                bordercolor='#2E86AB',
                font=dict(color='#2E86AB')
            )
        )
    )
    
    return fig

# メインアプリ
def main():
    st.title("📈 Stock Chart Analyzer")
    
    # データ読み込み
    df = load_data()
    
    if df is None:
        st.error("データファイルが見つかりません。data_j.csvを確認してください。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ Chart Settings")
        
        # 時間軸選択
        timeframe = st.selectbox(
            "📊 Time Frame",
            ["日足", "週足", "月足", "5分足"],
            index=0,
            help="チャートの時間軸を選択してください"
        )
        
        # 期間選択
        st.subheader("📅 Period")
        
        max_date = df['Date'].max()
        min_date = df['Date'].min()
        
        period_options = {
            "過去1週間": 7,
            "過去1ヶ月": 30,
            "過去3ヶ月": 90,
            "過去6ヶ月": 180,
            "過去1年": 365,
            "全期間": None
        }
        
        selected_period = st.selectbox(
            "期間を選択",
            list(period_options.keys()),
            index=2
        )
        
        if period_options[selected_period]:
            start_date = max_date - timedelta(days=period_options[selected_period])
            df_filtered = df[df['Date'] >= start_date].copy()
        else:
            df_filtered = df.copy()
        
        # 統計情報
        st.subheader("📊 Statistics")
        if not df_filtered.empty:
            latest_price = df_filtered['Close'].iloc[-1]
            price_change = df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[-2] if len(df_filtered) > 1 else 0
            price_change_pct = (price_change / df_filtered['Close'].iloc[-2] * 100) if len(df_filtered) > 1 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💰 Current Price", f"¥{latest_price:,.0f}")
            with col2:
                st.metric("📈 Change", f"¥{price_change:,.0f}", f"{price_change_pct:+.2f}%")
            
            st.metric("📊 Volume", f"{df_filtered['Volume'].iloc[-1]:,}")
            st.metric("🎯 VWAP", f"¥{df_filtered['VWAP'].iloc[-1]:,.0f}")
    
    # メインチャート
    if not df_filtered.empty:
        # データをリサンプル
        df_resampled = resample_data(df_filtered, timeframe)
        
        if not df_resampled.empty:
            # チャート作成
            fig = create_chart(df_resampled, timeframe)
            
            # チャート表示
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'scrollZoom': True
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # 使用方法
            with st.expander("💡 操作方法"):
                st.markdown("""
                **📱 チャート操作方法：**
                - 🖱️ **マウスホイール**: ズームイン/アウト
                - 🖱️ **ドラッグ**: チャートの移動（パン）
                - 🎯 **ダブルクリック**: 元の表示に戻る
                - 📊 **上部ボタン**: 期間選択（7D, 30D, 3M, 6M, All）
                
                **📈 チャート要素：**
                - 🕯️ **ローソク足**: 株価の動き（緑：上昇、赤：下降）
                - 📊 **VWAP**: オレンジライン（出来高加重平均価格）
                - 📏 **1σバンド**: 点線（標準偏差1倍）
                - 📏 **2σバンド**: 薄い帯状エリア（標準偏差2倍）
                - 📊 **出来高**: 下部の棒グラフ
                """)
        else:
            st.warning("選択した期間にデータがありません。")
    else:
        st.warning("選択した期間にデータがありません。")

if __name__ == "__main__":
    main()
