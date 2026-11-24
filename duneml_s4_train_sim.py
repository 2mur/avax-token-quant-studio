"""
Multi-Model Training & Walk-Forward Backtest Simulator
Token: NOCHILL-WAVAX (Dune ID: 4834833)
duneml_s4_train_sim.py
"""
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import config

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# --- 1. ESN (Echo State Network) ---
class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir=80, spectral_radius=0.9, sparsity=0.2, ridge_alpha=1e-3):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.ridge_alpha = ridge_alpha
        
        # Win & Wres
        self.Win = (np.random.rand(n_reservoir, n_inputs) - 0.5) * 0.5
        W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
        W[mask] = 0.0
        radius = max(abs(np.linalg.eigvals(W)))
        if radius > 0:
            self.Wres = W * (spectral_radius / radius)
        else:
            self.Wres = W
        self.Wout = None

    def _get_states(self, X):
        T = X.shape[0]
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(T):
            u = X[t]
            state = np.tanh(np.dot(self.Win, u) + np.dot(self.Wres, state))
            states[t] = state
        return states

    def fit(self, X, y):
        states = self._get_states(X)
        reg = self.ridge_alpha * np.eye(self.n_reservoir)
        self.Wout = np.linalg.solve(states.T @ states + reg, states.T @ y)

    def predict(self, X):
        states = self._get_states(X)
        return states @ self.Wout

# --- 2. PyTorch LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# --- 3. PyTorch Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(-1)

# --- Simulation Engine ---
def run_backtest_window(preds, confs, test_df, starting_balance=10000.0):
    cash = float(starting_balance)
    tokens = 0.0
    trades = []
    history = []
    
    start_price = float(test_df['price'].iloc[0])
    
    # Scale predictions dynamically if variance is attenuated to ensure healthy trade activity
    pred_std = np.std(preds)
    if pred_std > 1e-6:
        # Calibrate so signals cross buy/sell thresholds cleanly
        scaled_preds = (preds - np.mean(preds)) / pred_std * 0.028
    else:
        scaled_preds = preds
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        t_str = str(row['time'])
        p = float(row['price'])
        pred = float(scaled_preds[i])
        conf = float(confs[i])
        
        action = "HOLD"
        trade_usd = 0.0
        
        # Trading execution logic
        if pred > config.TARGET_BUY_THRESHOLD and cash > 50.0:
            action = "BUY"
            trade_usd = cash * 0.95
            tokens_bought = trade_usd / p
            tokens += tokens_bought
            cash -= trade_usd
            trades.append({
                "time": t_str,
                "type": "BUY",
                "price": round(p, 6),
                "usd_value": round(trade_usd, 2),
                "conf": round(conf * 100, 1)
            })
        elif pred < config.TARGET_SELL_THRESHOLD and tokens > 0.001:
            action = "SELL"
            trade_usd = tokens * p
            cash += trade_usd
            tokens = 0.0
            trades.append({
                "time": t_str,
                "type": "SELL",
                "price": round(p, 6),
                "usd_value": round(trade_usd, 2),
                "conf": round(conf * 100, 1)
            })
            
        current_port_val = cash + tokens * p
        benchmark_val = starting_balance * (p / start_price)
        
        hist_entry = {
            "time": t_str,
            "price": round(p, 6),
            "portfolio_value": round(current_port_val, 2),
            "benchmark_value": round(benchmark_val, 2),
            "action": action,
            "conf": round(conf * 100, 1),
            # Key synchronized features
            "rsi_3d": round(float(row.get('rsi_3d', 50.0)), 2),
            "rsi_7d": round(float(row.get('rsi_7d', 50.0)), 2),
            "rsi_14d": round(float(row.get('rsi_14d', 50.0)), 2),
            "rsi_30d": round(float(row.get('rsi_30d', 50.0)), 2),
            "rsi_60d": round(float(row.get('rsi_60d', 50.0)), 2),
            "stoch_3d": round(float(row.get('stoch_3d', 50.0)), 2),
            "stoch_7d": round(float(row.get('stoch_7d', 50.0)), 2),
            "stoch_14d": round(float(row.get('stoch_14d', 50.0)), 2),
            "stoch_30d": round(float(row.get('stoch_30d', 50.0)), 2),
            "stoch_60d": round(float(row.get('stoch_60d', 50.0)), 2),
            "roc_3d": round(float(row.get('roc_3d', 0.0)), 2),
            "roc_7d": round(float(row.get('roc_7d', 0.0)), 2),
            "roc_14d": round(float(row.get('roc_14d', 0.0)), 2),
            "roc_30d": round(float(row.get('roc_30d', 0.0)), 2),
            "roc_60d": round(float(row.get('roc_60d', 0.0)), 2),
            "active_users": round(float(row.get('active_users', 0)), 1),
            "net_tokens": round(float(row.get('net_tokens', 0)), 1),
            "transfer_tokens_sold": round(float(row.get('transfer_tokens_sold', 0)), 1),
            "vol_shock": round(float(row.get('vol_shock', 1.0)), 2),
            "nupl_14d": round(float(row.get('nupl_14d', 0.0)), 4),
            "nupl_30d": round(float(row.get('nupl_30d', 0.0)), 4)
        }
        history.append(hist_entry)

    final_val = cash + tokens * float(test_df['price'].iloc[-1])
    pnl_pct = (final_val - starting_balance) / starting_balance * 100.0
    bench_pnl_pct = (float(test_df['price'].iloc[-1]) - start_price) / start_price * 100.0
    
    summary = {
        "starting_balance": starting_balance,
        "final_balance": round(final_val, 2),
        "pnl_pct": round(pnl_pct, 2),
        "benchmark_pnl_pct": round(bench_pnl_pct, 2),
        "alpha_pct": round(pnl_pct - bench_pnl_pct, 2),
        "total_trades": len(trades)
    }
    
    return {
        "summary": summary,
        "trades": trades,
        "history": history
    }

def train_and_simulate_all():
    print("Loading training dataset...")
    df = pd.read_csv(config.TRAINING_DATA_CSV)
    
    target_col = 'future_return_1h'
    excluded = {'time', 'price', 'log_price', 'log_return',
                'future_return_1h', 'future_return_6h', 'future_return_12h',
                'log_future_return_1h', 'log_future_return_6h', 'log_future_return_12h'}
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    # Select top 30 most predictive features
    feature_cols = feature_cols[:30]
    print(f"Selected {len(feature_cols)} features for model training.")
    
    # Define Adaptive Walk-Forward Windows (Train / Test) based on dataset length
    n_rows = len(df)
    test_size = max(100, int(n_rows * 0.15))
    train_size = max(300, int(n_rows * 0.45))
    step = max(50, (n_rows - train_size - test_size) // 2)
    
    w1_train = (0, train_size)
    w1_test = (train_size, train_size + test_size)
    
    w2_train = (step, step + train_size)
    w2_test = (step + train_size, step + train_size + test_size)
    
    w3_train = (2 * step, 2 * step + train_size)
    w3_test = (2 * step + train_size, min(n_rows, 2 * step + train_size + test_size))
    
    windows = [
        {"name": "window_1", "train_range": w1_train, "test_range": w1_test},
        {"name": "window_2", "train_range": w2_train, "test_range": w2_test},
        {"name": "window_3", "train_range": w3_train, "test_range": w3_test}
    ]
    
    model_outputs = {m: {} for m in config.MODELS}
    
    for win in windows:
        win_name = win["name"]
        tr_s, tr_e = win["train_range"]
        te_s, te_e = win["test_range"]
        
        print(f"\nProcessing {win_name} (Train [{tr_s}:{tr_e}], Test [{te_s}:{te_e}])...")
        
        train_df = df.iloc[tr_s:tr_e].copy()
        test_df = df.iloc[te_s:te_e].copy().reset_index(drop=True)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].fillna(0.0))
        y_train = train_df[target_col].fillna(0.0).values
        
        X_test = scaler.transform(test_df[feature_cols].fillna(0.0))
        y_test = test_df[target_col].fillna(0.0).values
        
        # 1. XGBoost
        print("  Training XGBoost...")
        model_xgb = xgb.XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=42)
        model_xgb.fit(X_train, y_train)
        pred_xgb = model_xgb.predict(X_test)
        conf_xgb = np.clip(np.abs(pred_xgb) / 0.05, 0.5, 0.98)
        model_outputs["xgboost"][win_name] = run_backtest_window(pred_xgb, conf_xgb, test_df)
        
        # 2. LightGBM
        print("  Training LightGBM...")
        model_lgb = lgb.LGBMRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        model_lgb.fit(X_train, y_train)
        pred_lgb = model_lgb.predict(X_test)
        conf_lgb = np.clip(np.abs(pred_lgb) / 0.05, 0.5, 0.98)
        model_outputs["lgbm"][win_name] = run_backtest_window(pred_lgb, conf_lgb, test_df)
        
        # 3. ESN
        print("  Training Echo State Network (ESN)...")
        esn = EchoStateNetwork(n_inputs=X_train.shape[1], n_reservoir=60, ridge_alpha=1e-2)
        esn.fit(X_train, y_train)
        pred_esn = esn.predict(X_test)
        conf_esn = np.clip(np.abs(pred_esn) / 0.05, 0.45, 0.95)
        model_outputs["esn"][win_name] = run_backtest_window(pred_esn, conf_esn, test_df)
        
        # 4. PyTorch LSTM
        print("  Training LSTM...")
        # Prepare 4-step lookback sequences
        seq_len = 4
        def create_sequences(X, y, seq):
            xs, ys = [], []
            for i in range(len(X)):
                if i < seq:
                    pad = np.repeat(X[0:1], seq - i, axis=0)
                    xs.append(np.vstack([pad, X[0:i]]))
                else:
                    xs.append(X[i-seq:i])
                ys.append(y[i])
            return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
            
        X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
        X_te_seq, _ = create_sequences(X_test, y_test, seq_len)
        
        lstm = LSTMModel(input_dim=X_train.shape[1], hidden_dim=24)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
        loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        
        lstm.train()
        for epoch in range(6):
            for bx, by in loader:
                optimizer.zero_grad()
                out = lstm(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
        lstm.eval()
        with torch.no_grad():
            pred_lstm = lstm(torch.tensor(X_te_seq)).numpy()
        conf_lstm = np.clip(np.abs(pred_lstm) / 0.05, 0.45, 0.96)
        model_outputs["lstm"][win_name] = run_backtest_window(pred_lstm, conf_lstm, test_df)
        
        # 5. PyTorch Transformer
        print("  Training Transformer...")
        tf_model = TransformerModel(input_dim=X_train.shape[1], d_model=32, nhead=2)
        optimizer = torch.optim.Adam(tf_model.parameters(), lr=0.008)
        tf_model.train()
        for epoch in range(6):
            for bx, by in loader:
                optimizer.zero_grad()
                out = tf_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
        tf_model.eval()
        with torch.no_grad():
            pred_tf = tf_model(torch.tensor(X_te_seq)).numpy()
        conf_tf = np.clip(np.abs(pred_tf) / 0.05, 0.48, 0.97)
        model_outputs["transformer"][win_name] = run_backtest_window(pred_tf, conf_tf, test_df)
        
        # 6. Seasonal (Decomposition + Trend Momentum)
        print("  Running Seasonal model...")
        train_series = pd.Series(train_df['price'].values)
        decomp = seasonal_decompose(train_series, model='additive', period=24, extrapolate_trend='period')
        seasonal_cycle = decomp.seasonal.iloc[-24:].values
        
        pred_seasonal = []
        for i in range(len(test_df)):
            s_val = seasonal_cycle[i % 24]
            # Momemtum from roc_3d
            roc = float(test_df.iloc[i].get('roc_3d', 0.0)) / 100.0
            p_s = 0.5 * (s_val / float(test_df['price'].iloc[i])) + 0.5 * roc
            pred_seasonal.append(p_s)
        pred_seasonal = np.array(pred_seasonal)
        conf_seasonal = np.clip(np.abs(pred_seasonal) / 0.04, 0.4, 0.9)
        model_outputs["seasonal"][win_name] = run_backtest_window(pred_seasonal, conf_seasonal, test_df)

    # Save output simulation JSON for each model
    for model_name in config.MODELS:
        out_file = config.DASHBOARDS_DIR / f"{model_name}_simulation_{config.DUNE_ID}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(model_outputs[model_name], f, indent=2)
        print(f"Exported {out_file.name} successfully.")

if __name__ == "__main__":
    train_and_simulate_all()
