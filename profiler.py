import os
import sys
import json
import time
import math
import requests
import pandas as pd

ZERO_ADDR = "0x0000000000000000000000000000000000000000"

KNOWN_INFRA = {
    ZERO_ADDR: "ZERO_MINT_BURN",
    "0xdef171fe48cf0115b1d80b88dc8eab59176fee57": "PARASWAP_ROUTER",
    "0x60ae616a2155ee3d9a68541ba4544862310933d4": "TRADERJOE_ROUTER",
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1INCH_ROUTER",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1INCH_V5_ROUTER",
    "0xe54ca86531e17ef3616d22ca28b0d458b6c89106": "PANGOLIN_ROUTER",
}

_market_cache = {}  # addr_lower -> {"ts": float, "data": dict}


def fetch_dexscreener_market(token_address: str) -> dict:
    """
    Fetches live USD price, liquidity, and DEX pair addresses from DexScreener (60s TTL cache).
    Gracefully returns zeros if offline or unlisted.
    """
    addr_lower = token_address.lower()
    now = time.time()
    cached = _market_cache.get(addr_lower)
    if cached and (now - cached["ts"]) < 60:
        return cached["data"]

    result = {
        "price_usd": 0.0,
        "liquidity_usd": 0.0,
        "fdv_usd": 0.0,
        "pair_addresses": [],
        "primary_dex": "N/A",
    }

    try:
        url = f"https://api.dexscreener.com/token-pairs/v1/avalanche/{token_address}"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            pairs = resp.json()
            if isinstance(pairs, list) and len(pairs) > 0:
                # Sort pairs by USD liquidity descending
                pairs.sort(key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0), reverse=True)
                top = pairs[0]
                result["price_usd"] = float(top.get("priceUsd") or 0.0)
                result["liquidity_usd"] = sum(
                    float((p.get("liquidity") or {}).get("usd") or 0.0) for p in pairs
                )
                result["fdv_usd"] = float(top.get("fdv") or 0.0)
                result["primary_dex"] = str(top.get("dexId") or "N/A").upper()
                result["pair_addresses"] = [
                    str(p.get("pairAddress")).lower() for p in pairs if p.get("pairAddress")
                ]
    except Exception:
        pass

    _market_cache[addr_lower] = {"ts": now, "data": result}
    return result


def compute_gini(balances: list[float]) -> float:
    """Computes Gini inequality coefficient (0 = equal distribution, 1 = single holder)."""
    positive = sorted(b for b in balances if b > 0)
    n = len(positive)
    if n < 2:
        return 0.0
    total = sum(positive)
    if total <= 0:
        return 0.0
    cumulative = sum((i + 1) * val for i, val in enumerate(positive))
    return round((2.0 * cumulative) / (n * total) - (n + 1.0) / n, 4)


def profile_token_csv(
    csv_path: str,
    token_address: str = "",
    override_price_usd: float | None = None,
    limit: int = 250,
) -> dict:
    """
    Scans a token transfer CSV from disk and returns a full token profile + ranked holder list.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"summary": {}, "users": []}

    # Infer token decimals from raw_value / amount
    decimals = 18
    valid_rows = df[df["amount"] > 0]
    if not valid_rows.empty:
        r_val = int(valid_rows["raw_value"].iloc[0])
        a_val = float(valid_rows["amount"].iloc[0])
        if a_val > 0 and r_val > 0:
            decimals = max(0, min(36, round(math.log10(r_val / a_val))))
    scale = 10 ** decimals

    market = fetch_dexscreener_market(token_address) if token_address else {
        "price_usd": 0.0, "liquidity_usd": 0.0, "fdv_usd": 0.0, "pair_addresses": [], "primary_dex": "N/A"
    }
    price_usd = float(override_price_usd) if (override_price_usd is not None and override_price_usd >= 0) else market["price_usd"]
    pool_set = set(market.get("pair_addresses", []))

    wallets = {}
    seen_gas = set()
    minted_raw = 0
    burned_raw = 0
    total_volume_raw = 0
    total_gas_wei = 0

    first_block_global = int(df["blockNumber"].min())
    early_cutoff_block = first_block_global + 5000

    for ts, blk, tx_hash, sender, receiver, raw_val_str, g_used, g_price in zip(
        df["unix_ts"],
        df["blockNumber"],
        df["hash"],
        df["from"],
        df["to"],
        df["raw_value"],
        df["gasUsed"],
        df["gasPrice"],
    ):
        ts = int(ts)
        blk = int(blk)
        tx_hash = str(tx_hash)
        sender = str(sender).lower()
        receiver = str(receiver).lower()
        raw_val = int(raw_val_str)
        gas_cost_wei = int(g_used) * int(g_price)

        total_volume_raw += raw_val

        if sender == ZERO_ADDR:
            minted_raw += raw_val
        else:
            w = wallets.get(sender)
            if w is None:
                w = {
                    "address": sender,
                    "in_raw": 0,
                    "out_raw": 0,
                    "tx_in": 0,
                    "tx_out": 0,
                    "first_ts": ts,
                    "last_ts": ts,
                    "first_block": blk,
                    "gas_wei": 0,
                }
                wallets[sender] = w
            w["out_raw"] += raw_val
            w["tx_out"] += 1
            if ts < w["first_ts"]:
                w["first_ts"] = ts
            if ts > w["last_ts"]:
                w["last_ts"] = ts
            if blk < w["first_block"]:
                w["first_block"] = blk
            if (tx_hash, sender) not in seen_gas:
                seen_gas.add((tx_hash, sender))
                w["gas_wei"] += gas_cost_wei
                total_gas_wei += gas_cost_wei

        if receiver == ZERO_ADDR:
            burned_raw += raw_val
        else:
            w = wallets.get(receiver)
            if w is None:
                w = {
                    "address": receiver,
                    "in_raw": 0,
                    "out_raw": 0,
                    "tx_in": 0,
                    "tx_out": 0,
                    "first_ts": ts,
                    "last_ts": ts,
                    "first_block": blk,
                    "gas_wei": 0,
                }
                wallets[receiver] = w
            w["in_raw"] += raw_val
            w["tx_in"] += 1
            if ts < w["first_ts"]:
                w["first_ts"] = ts
            if ts > w["last_ts"]:
                w["last_ts"] = ts
            if blk < w["first_block"]:
                w["first_block"] = blk

    # Fallback circulating supply if mint event wasn't captured yet
    inferred_supply_raw = minted_raw - burned_raw
    if inferred_supply_raw <= 0:
        inferred_supply_raw = sum(max(0, w["in_raw"] - w["out_raw"]) for w in wallets.values())
    circulating_supply = max(1.0, inferred_supply_raw / scale)

    users = []
    active_balances = []

    for addr, w in wallets.items():
        net_raw = w["in_raw"] - w["out_raw"]
        bal = max(0.0, net_raw / scale)
        in_amt = w["in_raw"] / scale
        out_amt = w["out_raw"] / scale
        net_token_pnl = net_raw / scale
        total_tx = w["tx_in"] + w["tx_out"]
        supply_pct = (bal / circulating_supply) * 100.0
        gas_avax = w["gas_wei"] / 1e18
        hold_hours = round((w["last_ts"] - w["first_ts"]) / 3600.0, 2)

        # USD metrics anchored to spot price
        position_usd = bal * price_usd
        net_flow_usd = net_token_pnl * price_usd
        realized_extracted_usd = (out_amt - in_amt) * price_usd

        # Behavioral classification
        if addr in KNOWN_INFRA:
            tag = KNOWN_INFRA[addr]
        elif addr in pool_set or (total_tx >= 400 and w["tx_in"] > 50 and w["tx_out"] > 50):
            tag = "POOL / HUB"
        elif w["tx_out"] > 200 and w["tx_in"] <= 5:
            tag = "DISTRIBUTOR"
        elif supply_pct >= 1.0:
            tag = "WHALE"
        elif w["first_block"] <= early_cutoff_block and bal > 0:
            tag = "EARLY / SNIPER"
        elif bal == 0 and out_amt > 0:
            tag = "EXITED"
        elif w["tx_out"] == 0 and w["tx_in"] > 0:
            tag = "DIAMOND HAND"
        else:
            tag = "TRADER"

        if bal > 0 and tag not in ("POOL / HUB", "PARASWAP_ROUTER", "TRADERJOE_ROUTER", "1INCH_ROUTER", "1INCH_V5_ROUTER"):
            active_balances.append(bal)

        users.append({
            "address": addr,
            "tag": tag,
            "balance": round(bal, 4),
            "supply_pct": round(supply_pct, 4),
            "position_usd": round(position_usd, 2),
            "inflow": round(in_amt, 4),
            "outflow": round(out_amt, 4),
            "net_token_pnl": round(net_token_pnl, 4),
            "net_flow_usd": round(net_flow_usd, 2),
            "realized_extracted_usd": round(realized_extracted_usd, 2),
            "tx_total": total_tx,
            "tx_in": w["tx_in"],
            "tx_out": w["tx_out"],
            "gas_avax": round(gas_avax, 5),
            "hold_hours": hold_hours,
            "first_ts": w["first_ts"],
            "last_ts": w["last_ts"],
        })

    # Rank strictly by token ownership descending
    users.sort(key=lambda u: u["balance"], reverse=True)
    for idx, u in enumerate(users, start=1):
        u["rank"] = idx

    non_infra_users = [u for u in users if "ROUTER" not in u["tag"] and u["tag"] != "POOL / HUB"]
    top10_supply_pct = round(sum(u["supply_pct"] for u in non_infra_users[:10]), 2)

    summary = {
        "symbol": str(df["tokenSymbol"].iloc[0]),
        "total_transfers": int(len(df)),
        "unique_wallets": len(wallets),
        "active_holders": len(active_balances),
        "exited_wallets": sum(1 for u in users if u["tag"] == "EXITED"),
        "circulating_supply": round(circulating_supply, 2),
        "total_volume": round(total_volume_raw / scale, 2),
        "total_gas_avax": round(total_gas_wei / 1e18, 4),
        "top10_concentration_pct": top10_supply_pct,
        "gini_coefficient": compute_gini(active_balances),
        "price_usd": price_usd,
        "liquidity_usd": round(market["liquidity_usd"], 2),
        "primary_dex": market["primary_dex"],
    }

    return {
        "summary": summary,
        "users": users[:limit],
        "total_ranked": len(users),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profiler.py <path_to_csv> [token_address]")
        sys.exit(1)
    csv_arg = sys.argv[1]
    addr_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    report = profile_token_csv(csv_arg, token_address=addr_arg, limit=15)
    print(json.dumps(report["summary"], indent=2))
    print("\nTOP 15 HOLDERS:")
    for u in report["users"]:
        print(
            f"#{u['rank']:<3} {u['address']} | {u['tag']:<14} | "
            f"BAL: {u['balance']:>14,.2f} ({u['supply_pct']:>6.2f}%) | "
            f"TXS: {u['tx_total']:>5} | GAS: {u['gas_avax']:.4f} AVAX"
        )