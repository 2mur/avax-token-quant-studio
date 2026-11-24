"""
Feature Engineering Pipeline for NOCHILL-WAVAX (Dune ID: 4834833)
duneml_s2_features.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def compute_stoch(series, window):
    low_min = series.rolling(window, min_periods=1).min()
    high_max = series.rolling(window, min_periods=1).max()
    stoch = (series - low_min) / (high_max - low_min + 1e-9) * 100.0
    return stoch.fillna(50.0)

def build_feature_pipeline():
    print("Step 1: Loading prices and aligning continuous 1h timeline...")
    with open(config.PRICES_JSON, 'r', encoding='utf-8') as f:
        prices_data = json.load(f)
    
    price_df = pd.DataFrame(prices_data)
    price_df['time'] = pd.to_datetime(price_df['time'])
    if price_df['time'].dt.tz is not None:
        price_df['time'] = price_df['time'].dt.tz_convert(None)
    
    price_df.sort_values('time', inplace=True)
    price_df.drop_duplicates(subset=['time'], inplace=True)
    price_df.set_index('time', inplace=True)
    
    # Continuous 1h grid
    full_idx = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='1h')
    df = price_df.reindex(full_idx)
    df.index.name = 'time'
    df['price'] = df['price'].ffill().bfill()
    
    print("Step 2: Processing raw DEX trades and volume features...")
    trades_df = pd.read_csv(config.TRADES_CSV)
    trades_df['block_time'] = pd.to_datetime(trades_df['block_time'])
    if trades_df['block_time'].dt.tz is not None:
        trades_df['block_time'] = trades_df['block_time'].dt.tz_convert(None)
        
    # Filter usd_value > 1 (and buysell_pair if trade_type column is absent)
    if 'trade_type' in trades_df.columns:
        trades_filtered = trades_df[trades_df['usd_value'] > 1.0].copy()
    else:
        valid_pairs = ['NOCHILL-WAVAX', 'WAVAX-NOCHILL']
        trades_filtered = trades_df[
            trades_df['buysell_pair'].isin(valid_pairs) &
            (trades_df['usd_value'] > 1.0)
        ].copy()

    trades_filtered['time_hour'] = trades_filtered['block_time'].dt.floor('h')

    # Group by hourly trades
    def agg_trades(group):
        if 'trade_type' in group.columns:
            buy_v = group[group['trade_type'] == 'BUY']['usd_value'].sum()
            sell_v = group[group['trade_type'] == 'SELL']['usd_value'].sum()
        else:
            buy_v = group[group['buysell_pair'] == 'NOCHILL-WAVAX']['usd_value'].sum()
            sell_v = group[group['buysell_pair'] == 'WAVAX-NOCHILL']['usd_value'].sum()
        cnt = len(group)
        return pd.Series({
            'buy_vol': buy_v,
            'sell_vol': sell_v,
            'tx_count': cnt
        })
        
    trades_hourly = trades_filtered.groupby('time_hour').apply(agg_trades, include_groups=False)
    trades_hourly.index.name = 'time'
    
    # Join volume features
    df = df.join(trades_hourly, how='left')
    df['buy_vol'] = df['buy_vol'].fillna(0.0)
    df['sell_vol'] = df['sell_vol'].fillna(0.0)
    df['tx_count'] = df['tx_count'].fillna(0.0)
    df['total_vol'] = df['buy_vol'] + df['sell_vol']
    df['net_flow'] = df['buy_vol'] - df['sell_vol']
    
    print("Step 3: Calculating momentum and shock indicators...")
    df['price_change_1h'] = df['price'].pct_change().fillna(0.0)
    df['volatility_1h'] = df['price_change_1h'].rolling(2, min_periods=1).std().fillna(0.0)
    df['volatility_12h'] = df['price_change_1h'].rolling(12, min_periods=1).std().fillna(0.0)
    df['vol_shock'] = df['total_vol'] / (df['total_vol'].rolling(24, min_periods=1).mean() + 1.0)
    
    print("Step 4: Joining PnL metrics, calculating NUPL and PCA...")
    pnl_df = pd.read_csv(config.PNL_FEATURES_CSV)
    pnl_time_col = 'time' if 'time' in pnl_df.columns else 'block_time'
    pnl_df['time'] = pd.to_datetime(pnl_df[pnl_time_col])
    if pnl_df['time'].dt.tz is not None:
        pnl_df['time'] = pnl_df['time'].dt.tz_convert(None)
    pnl_df.drop_duplicates(subset=['time'], inplace=True)
    pnl_df.set_index('time', inplace=True)
    
    # Identify bin columns
    bin_cols = [c for c in pnl_df.columns if c.startswith('bin_')]
    base_pnl_cols = [
        'active_users', 'tokens_bought', 'tokens_sold', 'net_tokens', 
        'transfer_tokens_sold', 'unrealized_profit', 'unrealized_loss', 'market_cap'
    ]
    pnl_cols_to_join = [c for c in base_pnl_cols + bin_cols if c in pnl_df.columns]
    
    df = df.join(pnl_df[pnl_cols_to_join], how='left')
    for c in pnl_cols_to_join:
        df[c] = df[c].ffill().bfill().fillna(0.0)
        
    # NUPL
    df['nupl'] = (df['unrealized_profit'] - df['unrealized_loss']) / (df['market_cap'] + 1.0)
    df['nupl_14d'] = df['nupl'].rolling(14 * 24, min_periods=1).mean()
    df['nupl_30d'] = df['nupl'].rolling(30 * 24, min_periods=1).mean()
    
    # PCA on PnL columns
    pca_features = [c for c in pnl_cols_to_join if c in df.columns]
    scaler = StandardScaler()
    pnl_scaled = scaler.fit_transform(df[pca_features].fillna(0.0))
    pca = PCA(n_components=config.N_PCA_COMP)
    pca_result = pca.fit_transform(pnl_scaled)
    for i in range(config.N_PCA_COMP):
        df[f'pca_pnl_{i+1}'] = pca_result[:, i]
        
    print("Step 5: Generating multi-timeframe (MTF) indicators...")
    mtf_cols = []
    days_list = [3, 7, 14, 30, 60]
    for d in days_list:
        w = d * 24
        ma_col = f'ma_{d}d'
        bb_up_col = f'bb_up_{d}d'
        bb_low_col = f'bb_low_{d}d'
        rsi_col = f'rsi_{d}d'
        stoch_col = f'stoch_{d}d'
        roc_col = f'roc_{d}d'
        
        df[ma_col] = df['price'].rolling(w, min_periods=1).mean()
        roll_std = df['price'].rolling(w, min_periods=1).std().fillna(0.0)
        df[bb_up_col] = df[ma_col] + 2.0 * roll_std
        df[bb_low_col] = df[ma_col] - 2.0 * roll_std
        df[rsi_col] = compute_rsi(df['price'], w)
        df[stoch_col] = compute_stoch(df['price'], w)
        df[roc_col] = (df['price'].pct_change(w) * 100.0).fillna(0.0)
        
        mtf_cols.extend([ma_col, bb_up_col, bb_low_col, rsi_col, stoch_col, roc_col])
        
    print("Step 6: Calculating log transforms and forward return targets...")
    df['log_price'] = np.log(np.maximum(df['price'], 1e-8))
    df['log_return'] = np.log(df['price'] / df['price'].shift(1)).fillna(0.0)
    
    for h in [1, 6, 12]:
        df[f'future_return_{h}h'] = (df['price'].shift(-h) / df['price'] - 1.0).fillna(0.0)
        df[f'log_future_return_{h}h'] = np.log(np.maximum(df['price'].shift(-h), 1e-8) / np.maximum(df['price'], 1e-8)).fillna(0.0)
        
    print("Step 7: Computing SMA Velocities (_v)...")
    vel_cols = [
        'buy_vol', 'sell_vol', 'tx_count', 'total_vol', 'net_flow',
        'price_change_1h', 'volatility_1h', 'volatility_12h', 'vol_shock',
        'active_users', 'tokens_bought', 'tokens_sold', 'net_tokens', 'transfer_tokens_sold'
    ]
    for i in range(config.N_PCA_COMP):
        vel_cols.append(f'pca_pnl_{i+1}')
    vel_cols.extend(mtf_cols)
    
    vel_dict = {}
    for col in vel_cols:
        if col in df.columns:
            sma = df[col].rolling(config.SMA_VELOCITY_WINDOW, min_periods=1).mean()
            vel_dict[f'{col}_v'] = df[col] - sma
    df = pd.concat([df, pd.DataFrame(vel_dict, index=df.index)], axis=1)
            
    print("Step 8: Exporting full dataset and 12h dashboard payload...")
    # Reset index so 'time' is a column
    df_out = df.reset_index()
    df_out['time'] = df_out['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save training_data_4834833.csv
    df_out.to_csv(config.TRAINING_DATA_CSV, index=False)
    print(f"Saved full 1h training dataset ({len(df_out)} rows, {df_out.shape[1]} cols) to {config.TRAINING_DATA_CSV}")
    
    # Downsample to 12h for dashboard_features_4834833.json
    df_12h = df.resample(config.DASHBOARD_RESOLUTION).last().reset_index()
    df_12h['time'] = df_12h['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    dashboard_payload = {
        "metadata": {
            "token": "NOCHILL-WAVAX",
            "dune_id": config.DUNE_ID,
            "resolution": config.DASHBOARD_RESOLUTION,
            "generated_rows": len(df_12h),
            "columns": list(df_12h.columns)
        },
        "data": df_12h.to_dict(orient="records")
    }
    
    with open(config.DASHBOARD_FEATURES_JSON, 'w', encoding='utf-8') as f:
        json.dump(dashboard_payload, f, indent=2)
    print(f"Saved 12h downsampled dashboard payload ({len(df_12h)} rows) to {config.DASHBOARD_FEATURES_JSON}")

if __name__ == "__main__":
    build_feature_pipeline()
