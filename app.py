import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="📈 Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """CSVデータを読み込み（自動カラム検出付き）"""
    try:
        # CSVファイルを読み込み
        df = pd.read_csv('data_j.csv', encoding='utf-8')
        
        # デバッグ情報を表示
        st.sidebar.markdown("### 📋 CSVファイル情報")
        st.sidebar.write(f"**カラム数**: {len(df.columns)}")
        st.sidebar.write(f"**行数**: {len(df)}")
        st.sidebar.write("**カラム名**:")
        for i, col in enumerate(df.columns):
            st.sidebar.write(f"{i+1}. `{col}`")
        
        # 日付カラムを自動検出
        date_column = None
        possible_date_names = ['Date', 'date', 'DATE', '日付', 'Timestamp', 'timestamp', 'Time', 'time']
        
        for col in df.columns:
            if col in possible_date_names:
                date_column = col
                break
        
        # 日付カラムが見つからない場合、最初のカラムを使用
        if date_column is None:
            date_column = df.columns
            st.warning(f"日付カラムが見つかりません。'{date_column}'を日付として使用します。")
        
        # 日付カラムを変換
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            st.error(f"'{date_column}'カラムを日付に変換できませんでした。")
            return None
        
        # 標準カラム名にリネーム
        df = df.rename(columns={date_column: 'Date'})
        
        # 価格カラムを自動検出
        price_column = None
        possible_price_names = ['Price', 'price', 'PRICE', '価格', 'Close', 'close', 'CLOSE', '終値']
        
        for col in df.columns:
            if col in possible_price_names:
                price_column = col
                break
        
        if price_column is None:
            # 数値カラムから最初のものを選択
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_column = numeric_cols
                st.warning(f"価格カラムが見つかりません。'{price_column}'を価格として使用します。")
            else:
                st.error("数値カラムが見つかりません。")
                return None
        
        # 標準カラム名にリネーム
        if price_column != 'Price':
            df = df.rename(columns={price_column: 'Price'})
        
        # ボリュームカラムを自動検出（オプション）
        volume_column = None
        possible_volume_names = ['Volume', 'volume', 'VOLUME', '出来高', 'Vol', 'vol']
        
        for col in df.columns:
            if col in possible_volume_names:
                volume_column = col
                break
        
        if volume_column is None:
            # ボリュームがない場合はダミーデータを作成
            df['Volume'] = 1000000  # 固定値
            st.info("出来高カラムが見つかりません。固定値を使用します。")
        else:
            df = df.rename(columns={volume_column: 'Volume'})
        
        # VWAPカラムを自動検出（オプション）
        vwap_column = None
        possible_vwap_names = ['VWAP', 'vwap', 'Vwap']
        
        for col in df.columns:
            if col in possible_vwap_names:
                vwap_column = col
                break
        
        if vwap_column is None:
            # VWAPがない場合は価格をベースに計算
            df['VWAP'] = df['Price']
            st.info("VWAPカラムが見つかりません。価格をベースに使用します。")
        else:
            df = df.rename(columns={vwap_column: 'VWAP'})
        
        # 休日・土日を除外してデータを詰める
        df = df.dropna(subset=['Price'])  # 価格がNaNの行を削除
        df = df.reset_index(drop=True)
        
        st.sidebar.success("✅ データ読み込み完了!")
        return df
        
    except FileNotFoundError:
        st.error("❌ data_j.csvファイルが見つかりません。GitHubリポジトリにファイルがアップロードされているか確認してください。")
        return None
    except Exception as e:
        st.error(f"❌ データ読み込みエラー: {str(e)}")
        st.markdown("### 🔍 トラブルシューティング")
        st.markdown("""
        1. **ファイル名を確認**: `data_j.csv`が正しい名前か確認
        2. **エンコーディングを変更**: CSVファイルがUTF-8で保存されているか確認
        3. **カラム名を確認**: 日付、価格、出来高のカラム名を確認
        """)
        return None

def resample_data(df, timeframe):
    """指定された時間足にリサンプリング"""
    df_copy = df.copy()
    df_copy.set_index('Date', inplace=True)
    
    if timeframe == '5分足':
        freq = '5T'
    elif timeframe == '日足':
        freq = 'D'
    elif timeframe == '週足':
        freq = 'W'
    elif timeframe == '月足':
        freq = 'M'
    else:
        return df_copy.reset_index()
    
    # OHLC形式にリサンプル
    resampled = df_copy.resample(freq).agg({
        'Price': ['first', 'max', 'min', 'last'],
        'Volume': 'sum',
        'VWAP': 'mean'
    }).dropna()
    
    # カラム名を整理
    resampled.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
    resampled = resampled.reset_index()
    
    return resampled

def calculate_vwap_bands(df, period=20):
    """VWAPバンドを計算"""
    df = df.copy()
    
    # 移動平均VWAP
    df['VWAP_MA'] = df['VWAP'].rolling(window=period).mean()
    
    # 標準偏差を計算
    df['VWAP_STD'] = df['VWAP'].rolling(window=period).std()
    
    # 1σと2σバンド
    df['VWAP_Upper_1'] = df['VWAP_MA'] + df['VWAP_STD']
    df['VWAP_Lower_1'] = df['VWAP_MA'] - df['VWAP_STD']
    df['VWAP_Upper_2'] = df['VWAP_MA'] + (df['VWAP_STD'] * 2)
    df['VWAP_Lower_2'] = df['VWAP_MA'] - (df['VWAP_STD'] * 2)
    
    return df

def create_advanced_chart(df, timeframe):
    """高機能なインタラクティブチャートを作成"""
    
    # VWAPバンドを計算
    df = calculate_vwap_bands(df)
    
    # サブプロット作成（価格チャート + ボリューム）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('', ''),
        row_width=[0.7, 0.3]
    )
    
    # カラーパレット（おしゃれな色）
    colors = {
        'candle_up': '#26a69a',
        'candle_down': '#ef5350', 
        'vwap': '#ff9800',
        'band_1': 'rgba(103, 126, 234, 0.3)',
        'band_2': 'rgba(103, 126, 234, 0.15)',
        'volume': '#9c27b0'
    }
    
    # ローソク足チャート
    if timeframe != '日足' or 'Open' in df.columns:
        fig.add_trace(
            go.Candlestick(
                x=df['Date'] if 'Date' in df.columns else df.index,
                open=df['Open'] if 'Open' in df.columns else df['Price'],
                high=df['High'] if 'High' in df.columns else df['Price'],
                low=df['Low'] if 'Low' in df.columns else df['Price'],
                close=df['Close'] if 'Close' in df.columns else df['Price'],
                name='Price',
                increasing_line_color=colors['candle_up'],
                decreasing_line_color=colors['candle_down'],
                showlegend=False
            ),
            row=1, col=1
        )
    else:
        # 日足の場合は線グラフ
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Price'],
                mode='lines',
                name='Price',
                line=dict(color=colors['candle_up'], width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # VWAPバンド（2σ）
    fig.add_trace(
        go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['VWAP_Upper_2'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['VWAP_Lower_2'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor=colors['band_2'],
            name='2σ Band',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWAPバンド（1σ）
    fig.add_trace(
        go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['VWAP_Upper_1'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['VWAP_Lower_1'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor=colors['band_1'],
            name='1σ Band',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWAP線
    fig.add_trace(
        go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color=colors['vwap'], width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # ボリューム
    fig.add_trace(
        go.Bar(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors['volume'],
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # レイアウト設定（高機能 & おしゃれ）
    fig.update_layout(
        title=dict(
            text=f'<b>株価チャート ({timeframe})</b>',
            x=0.5,
            font=dict(size=24, color='#2c3e50')
        ),
        height=700,
        showlegend=False,  # 凡例を非表示
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Helvetica, Arial, sans-serif")
    )
    
    # X軸とY軸のスタイル設定
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False
    )
    
    # インタラクティブ機能の設定
    fig.update_layout(
        dragmode='zoom',  # ドラッグでズーム
        scrollZoom=True,  # ホイールでズーム
        doubleClick='reset',  # ダブルクリックでリセット
        showTips=True
    )
    
    return fig

def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>📈 Advanced Stock Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    df = load_data()
    if df is None:
        st.stop()
    
    # サイドバー
    st.sidebar.markdown("## ⚙️ チャート設定")
    
    # 時間足選択
    timeframe = st.sidebar.selectbox(
        "📊 時間足を選択",
        ["日足", "週足", "月足", "5分足"],
        index=0
    )
    
    # 表示期間設定
    st.sidebar.markdown("### 📅 表示期間")
    
    # 期間プリセット
    period_preset = st.sidebar.selectbox(
        "期間プリセット",
        ["カスタム", "1ヶ月", "3ヶ月", "6ヶ月", "1年", "全期間"]
    )
    
    if period_preset != "カスタム":
        end_date = df['Date'].max()
        if period_preset == "1ヶ月":
            start_date = end_date - timedelta(days=30)
        elif period_preset == "3ヶ月":
            start_date = end_date - timedelta(days=90)
        elif period_preset == "6ヶ月":
            start_date = end_date - timedelta(days=180)
        elif period_preset == "1年":
            start_date = end_date - timedelta(days=365)
        else:  # 全期間
            start_date = df['Date'].min()
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['Date'].min().date())
        with col2:
            end_date = st.date_input("終了日", df['Date'].max().date())
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # データフィルタリング
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    if filtered_df.empty:
        st.error("選択した期間にデータがありません。")
        st.stop()
    
    # 時間足でリサンプリング
    if timeframe != "日足":
        filtered_df = resample_data(filtered_df, timeframe)
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = filtered_df['Price'].iloc[-1] if 'Price' in filtered_df.columns else filtered_df['Close'].iloc[-1]
    prev_price = filtered_df['Price'].iloc[-2] if len(filtered_df) > 1 and 'Price' in filtered_df.columns else filtered_df['Close'].iloc[-2] if len(filtered_df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    with col1:
        st.metric(
            label="現在価格",
            value=f"¥{current_price:,.0f}",
            delta=f"{price_change:+.0f} ({price_change_pct:+.1f}%)"
        )
    
    with col2:
        high_price = filtered_df['High'].max() if 'High' in filtered_df.columns else filtered_df['Price'].max()
        st.metric(label="期間最高値", value=f"¥{high_price:,.0f}")
    
    with col3:
        low_price = filtered_df['Low'].min() if 'Low' in filtered_df.columns else filtered_df['Price'].min()
        st.metric(label="期間最安値", value=f"¥{low_price:,.0f}")
    
    with col4:
        avg_volume = filtered_df['Volume'].mean()
        st.metric(label="平均出来高", value=f"{avg_volume:,.0f}")
    
    # チャート表示
    st.markdown("## 📊 インタラクティブチャート")
    st.markdown("""
    **🖱️ 操作方法:**
    - **ホイール**: ズームイン・アウト
    - **ドラッグ**: チャート移動
    - **ダブルクリック**: ズームリセット
    - **ボックス選択**: 範囲ズーム
    """)
    
    fig = create_advanced_chart(filtered_df, timeframe)
    
    # チャート表示（設定でインタラクティブ機能を有効化）
    st.plotly_chart(
        fig, 
        use_container_width=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'autoScale2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'scrollZoom': True,  # ホイールズーム有効
            'doubleClick': 'reset'  # ダブルクリックでリセット
        }
    )
    
    # データテーブル
    with st.expander("📋 データテーブル", expanded=False):
        st.dataframe(
            filtered_df.tail(50),
            use_container_width=True,
            height=300
        )
    
    # フッター
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
        "📈 Advanced Stock Analysis Dashboard | Powered by Streamlit & Plotly"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
