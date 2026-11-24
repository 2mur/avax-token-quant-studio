"""
Helper script to generate realistic mock Dune datasets for NOCHILL-WAVAX (Dune ID: 4834833)
Generates:
1. dune-databoards/prices_4834833.json
2. dune-data/dune_query_4834833_latest.csv
3. dune-data/pnl_features_4834833.csv
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config

def generate_all_mock_data(n_days=120, seed=42):
    np.random.seed(seed)
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    n_hours = n_days * 24
    time_index = [start_date + timedelta(hours=i) for i in range(n_hours)]
    
    print(f"Generating mock data for {n_hours} hours ({n_days} days)...")

    # 1. Price generation (Geometric Brownian Motion with mean-reversion and regime shifts)
    dt = 1.0 / 24.0
    prices = [0.0085]
    regime = 0.05
    for i in range(1, n_hours):
        # Shift regime occasionally
        if i % 180 == 0:
            regime = np.random.choice([-0.25, -0.08, 0.05, 0.22, 0.45])
        shock = 0.0
        if np.random.rand() < 0.03: # occasional pump/dump spikes
            shock = np.random.choice([-0.06, 0.08])
        drift = regime * dt
        volatility = 0.04 * np.sqrt(dt)
        shock_vol = 0.015 * np.random.randn()
        ret = drift + shock_vol + shock
        p_next = max(0.0005, prices[-1] * (1.0 + ret))
        prices.append(round(p_next, 6))

    # Save prices_4834833.json
    price_records = [
        {"time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": p}
        for t, p in zip(time_index, prices)
    ]
    with open(config.PRICES_JSON, "w", encoding="utf-8") as f:
        json.dump(price_records, f, indent=2)
    print(f"Saved {len(price_records)} price points to {config.PRICES_JSON}")

    # 2. Trades generation (raw DEX trades)
    trades = []
    pairs = ['NOCHILL-WAVAX', 'WAVAX-NOCHILL']
    for idx, t in enumerate(time_index):
        p = prices[idx]
        # Number of trades this hour
        n_trades = np.random.randint(6, 26)
        # Momentum bias based on recent price change
        p_prev = prices[idx - 1] if idx > 0 else p
        ret = (p - p_prev) / p_prev
        buy_prob = np.clip(0.5 + ret * 8.0, 0.2, 0.8)
        
        for _ in range(n_trades):
            sec_offset = np.random.randint(0, 3600)
            tx_time = t + timedelta(seconds=sec_offset)
            is_buy = np.random.rand() < buy_prob
            pair = 'NOCHILL-WAVAX' if is_buy else 'WAVAX-NOCHILL'
            # USD value log-normal distribution
            usd_val = float(np.random.lognormal(mean=5.2, sigma=1.3))
            if np.random.rand() < 0.02: # occasional micro-trade or dust
                usd_val = round(np.random.uniform(0.1, 0.95), 2)
            else:
                usd_val = round(max(1.5, usd_val), 2)

            trades.append({
                "block_time": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "buysell_pair": pair,
                "usd_value": usd_val
            })
            
    # Add a few noisy pairs to test filtering
    for _ in range(20):
        t_rand = start_date + timedelta(hours=np.random.randint(0, n_hours))
        trades.append({
            "block_time": t_rand.strftime("%Y-%m-%d %H:%M:%S"),
            "buysell_pair": "USDC-WAVAX",
            "usd_value": 150.0
        })

    trades_df = pd.DataFrame(trades)
    trades_df.sort_values("block_time", inplace=True)
    trades_df.to_csv(config.TRADES_CSV, index=False)
    print(f"Saved {len(trades_df)} DEX trades to {config.TRADES_CSV}")

    # 3. PnL features generation
    pnl_records = []
    base_mcap = 8_500_000.0
    for idx, t in enumerate(time_index):
        p = prices[idx]
        mcap = base_mcap * (p / prices[0])
        users = int(max(10, np.random.poisson(lam=45) * (1.0 + abs(ret) * 10)))
        
        # Token volumes
        tk_bought = float(np.random.uniform(200_000, 2_500_000))
        tk_sold = float(np.random.uniform(180_000, 2_400_000))
        net_tk = tk_bought - tk_sold
        trans_sold = float(np.random.uniform(10_000, 500_000))
        
        # PnL distribution
        unrealized_prof = max(5_000.0, mcap * np.random.uniform(0.08, 0.22))
        unrealized_loss = max(3_000.0, mcap * np.random.uniform(0.04, 0.18))
        
        # Cohort distribution bins (summing to 100%)
        # e.g., heavy loss, moderate loss, breakeven, moderate profit, heavy profit
        profit_sentiment = (p - min(prices[:idx+1])) / (max(prices[:idx+1]) - min(prices[:idx+1]) + 1e-6)
        w_hl = np.clip(0.35 - 0.25 * profit_sentiment + np.random.normal(0, 0.02), 0.05, 0.6)
        w_ml = np.clip(0.25 - 0.10 * profit_sentiment + np.random.normal(0, 0.02), 0.05, 0.4)
        w_be = np.clip(0.20 + np.random.normal(0, 0.02), 0.05, 0.4)
        w_mp = np.clip(0.12 + 0.15 * profit_sentiment + np.random.normal(0, 0.02), 0.05, 0.4)
        w_hp = np.clip(0.08 + 0.20 * profit_sentiment + np.random.normal(0, 0.02), 0.05, 0.4)
        tot = w_hl + w_ml + w_be + w_mp + w_hp
        
        pnl_records.append({
            "time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "active_users": users,
            "tokens_bought": round(tk_bought, 2),
            "tokens_sold": round(tk_sold, 2),
            "net_tokens": round(net_tk, 2),
            "transfer_tokens_sold": round(trans_sold, 2),
            "unrealized_profit": round(unrealized_prof, 2),
            "unrealized_loss": round(unrealized_loss, 2),
            "market_cap": round(mcap, 2),
            "bin_heavy_loss": round((w_hl / tot) * 100, 2),
            "bin_moderate_loss": round((w_ml / tot) * 100, 2),
            "bin_breakeven": round((w_be / tot) * 100, 2),
            "bin_moderate_profit": round((w_mp / tot) * 100, 2),
            "bin_heavy_profit": round((w_hp / tot) * 100, 2)
        })
        
    pnl_df = pd.DataFrame(pnl_records)
    pnl_df.to_csv(config.PNL_FEATURES_CSV, index=False)
    print(f"Saved {len(pnl_df)} hourly PnL feature rows to {config.PNL_FEATURES_CSV}")
    print("Mock Dune data generation complete!")

if __name__ == "__main__":
    generate_all_mock_data()
