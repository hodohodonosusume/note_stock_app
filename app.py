import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(
    page_title="株価分析アプリ", 
    page_icon="📈", 
    layout="wide"
)

# タイトル
st.title("📈 株価分析アプリ")

# サイドバーでファイルアップロード
st.sidebar.header("データ読み込み")

# CSVファイル読み込み
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_j.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # 休日・空白データを削除
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Volume'] > 0]  # 出来高0のデータも除外
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None

# データ読み込み
df = load_data()

if df is not None:
    # サイドバーで時間軸選択
    st.sidebar.header("チャート設定")
    timeframe = st.sidebar.selectbox(
        "時間軸を選択",
        ["日足", "週足", "月足", "5分足"],
        index=0
    )
    
    # 表示期間選択
    period_days = st.sidebar.slider("表示期間（日）", 30, len(df), 200)
    
    # VWAPバンド設定
    show_vwap_bands = st.sidebar.checkbox("VWAPバンド表示", True)
    
    # データの時間軸変換
    @st.cache_data
    def resample_data(df, timeframe):
        df_resampled = df.copy()
        df_resampled.set_index('Date', inplace=True)
        
        if timeframe == "週足":
            rule = 'W'
        elif timeframe == "月足":
            rule = 'M'
        elif timeframe == "5分足":
            # 5分足は元データが日足の場合は意味がないが、デモ用に実装
            rule = '5T'
        else:  # 日足
            rule = 'D'
        
        if timeframe != "日足":
            ohlc_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            df_resampled = df_resampled.resample(rule).agg(ohlc_dict)
            df_resampled = df_resampled.dropna()
        
        df_resampled.reset_index(inplace=True)
        return df_resampled
    
    # VWAP計算
    @st.cache_data
    def calculate_vwap_with_bands(df):
        df = df.copy()
        
        # Typical Price
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Cumulative TP * Volume
        df['TP_Volume'] = df['TP'] * df['Volume']
        df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        
        # VWAP
        df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
        
        # VWAPからの偏差計算（簡易版）
        df['Price_VWAP_Diff'] = df['TP'] - df['VWAP']
        df['Price_VWAP_Diff_Sq'] = df['Price_VWAP_Diff'] ** 2
        
        # 移動平均を使った標準偏差の近似
        window = min(20, len(df))
        df['VWAP_Std'] = df['Price_VWAP_Diff_Sq'].rolling(window=window).mean() ** 0.5
        
        # VWAPバンド
        df['VWAP_Upper_1std'] = df['VWAP'] + df['VWAP_Std']
        df['VWAP_Lower_1std'] = df['VWAP'] - df['VWAP_Std']
        df['VWAP_Upper_2std'] = df['VWAP'] + 2 * df['VWAP_Std']
        df['VWAP_Lower_2std'] = df['VWAP'] - 2 * df['VWAP_Std']
        
        return df
    
    # データ処理
    df_chart = resample_data(df, timeframe)
    df_chart = calculate_vwap_with_bands(df_chart)
    
    # 表示期間でフィルタ
    df_display = df_chart.tail(period_days)
    
    # チャート作成
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=['価格チャート', '出来高']
    )
    
    # ローソク足チャート
    fig.add_trace(
        go.Candlestick(
            x=df_display['Date'],
            open=df_display['Open'],
            high=df_display['High'],
            low=df_display['Low'],
            close=df_display['Close'],
            name='価格',
            showlegend=False,  # 凡例非表示
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ),
        row=1, col=1
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(
            x=df_display['Date'],
            y=df_display['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='#ff9800', width=2),
            showlegend=False  # 凡例非表示
        ),
        row=1, col=1
    )
    
    # VWAPバンド
    if show_vwap_bands:
        # 2σバンド
        fig.add_trace(
            go.Scatter(
                x=df_display['Date'],
                y=df_display['VWAP_Upper_2std'],
                mode='lines',
                name='VWAP+2σ',
                line=dict(color='rgba(255, 152, 0, 0.3)', width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_display['Date'],
                y=df_display['VWAP_Lower_2std'],
                mode='lines',
                name='VWAP-2σ',
                line=dict(color='rgba(255, 152, 0, 0.3)', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 1σバンド
        fig.add_trace(
            go.Scatter(
                x=df_display['Date'],
                y=df_display['VWAP_Upper_1std'],
                mode='lines',
                name='VWAP+1σ',
                line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_display['Date'],
                y=df_display['VWAP_Lower_1std'],
                mode='lines',
                name='VWAP-1σ',
                line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.15)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 出来高チャート
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df_display['Close'], df_display['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df_display['Date'],
            y=df_display['Volume'],
            name='出来高',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # レイアウト設定（おしゃれ＆機能的）
    fig.update_layout(
        title=dict(
            text=f'{timeframe}チャート',
            x=0.5,
            font=dict(size=24, color='#2c3e50')
        ),
        xaxis_rangeslider_visible=False,  # レンジスライダー非表示
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa',
        font=dict(color='#2c3e50'),
        margin=dict(l=50, r=50, t=80, b=50),
        # ホバー時の十字線
        hovermode='x unified',
    )
    
    # X軸設定
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='#bdc3c7'
    )
    
    # Y軸設定
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='#bdc3c7'
    )
    
    # インタラクティブ機能設定
    config = {
        'scrollZoom': True,  # ホイールでズーム
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'stock_chart',
            'height': 700,
            'width': 1200,
            'scale': 2
        }
    }
    
    # チャート表示
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # 操作方法の説明
    st.info("""
    **📊 チャート操作方法:**
    - **ズーム**: マウスホイールでズームイン/アウト
    - **パン**: ドラッグで移動
    - **期間選択**: チャート上でドラッグして範囲選択
    - **リセット**: ダブルクリックで元の表示範囲に戻る
    """)
    
    # 統計情報表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("現在価格", f"¥{df_display['Close'].iloc[-1]:,.0f}")
    
    with col2:
        price_change = df_display['Close'].iloc[-1] - df_display['Close'].iloc[-2]
        price_change_pct = (price_change / df_display['Close'].iloc[-2]) * 100
        st.metric("前日比", f"¥{price_change:+,.0f}", f"{price_change_pct:+.2f}%")
    
    with col3:
        st.metric("VWAP", f"¥{df_display['VWAP'].iloc[-1]:,.0f}")
    
    with col4:
        avg_volume = df_display['Volume'].tail(20).mean()
        st.metric("平均出来高(20日)", f"{avg_volume:,.0f}")

else:
    st.error("データファイル 'data_j.csv' が見つかりません。ファイルをアップロードしてください。")
    
    # ファイルアップロード機能
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success("ファイルがアップロードされました！")
        st.dataframe(df_uploaded.head())

