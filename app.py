"""
AVAX Quant Terminal & Blockchain Sourcing Platform
Integrated Flask Application combining:
1. Real-time Blockchain Transaction Sourcing & Indexer (from Hydrogen Project)
2. Token Profiler & Wealth Distribution Analysis
3. High-Frequency AMM Price Reconstruction
4. Quant Feature Engineering Pipeline (duneml_s2_features)
5. Exploratory Data Analysis & Manifold Engine (duneml_s3_eda)
6. 6-Model Walk-Forward Machine Learning Simulator (duneml_s4_train_sim)
"""
import os
import sys
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

import config
import sourcing_blockchain
from profiler import profile_token_csv

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, "state.json")

DASHBOARDS_DIR = config.DASHBOARDS_DIR

ROUTESCAN_API_URL = "https://api.routescan.io/v2/network/mainnet/evm/43114/etherscan/api"
ROUTESCAN_API_KEY = "placeholder"

state_lock = threading.Lock()
workers = {}

# Background pipeline execution state
pipeline_lock = threading.Lock()
pipeline_state = {
    "status": "IDLE",  # IDLE, RUNNING, COMPLETED, ERROR
    "step": "",
    "progress": 0,
    "address": config.DEFAULT_NOCHILL_ADDRESS,
    "stage": "all",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "logs": []
}


def load_state() -> dict:
    with state_lock:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}


def save_state(state: dict) -> None:
    with state_lock:
        tmp_file = STATE_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_file, STATE_FILE)


def append_worker_log(addr: str, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {message}"
    if addr in workers:
        workers[addr]["logs"].append(entry)
        if len(workers[addr]["logs"]) > 60:
            workers[addr]["logs"] = workers[addr]["logs"][-60:]


def append_pipeline_log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {message}"
    with pipeline_lock:
        pipeline_state["logs"].append(entry)
        if len(pipeline_state["logs"]) > 100:
            pipeline_state["logs"] = pipeline_state["logs"][-100:]


def compute_completion_metrics(coin: dict) -> dict:
    now_ts = int(time.time())
    first_ts = coin.get("first_tx_ts")
    last_ts = coin.get("last_tx_ts")
    collected = coin.get("collected_count", 0)
    last_updated = coin.get("last_updated_ts", now_ts)

    if not first_ts or not last_ts or collected == 0:
        return {
            "time_pct": 0.0,
            "tx_pct": 0.0,
            "est_total_txs": 0,
            "est_new_txs_since_update": 0,
            "seconds_since_update": max(0, now_ts - last_updated),
        }

    total_lifespan = max(1, now_ts - first_ts)
    covered_span = max(0, min(last_ts - first_ts, total_lifespan))
    time_pct = round((covered_span / total_lifespan) * 100.0, 2)

    tx_rate_per_sec = collected / max(1, covered_span)
    uncollected_gap_sec = max(0, now_ts - last_ts)

    if coin.get("status") == "SYNCED" and uncollected_gap_sec < 15:
        est_new_txs = 0
        time_pct = 100.0
    else:
        est_new_txs = int(round(tx_rate_per_sec * uncollected_gap_sec))

    est_total_txs = collected + est_new_txs
    tx_pct = round((collected / max(1, est_total_txs)) * 100.0, 2)

    return {
        "time_pct": min(100.0, time_pct),
        "tx_pct": min(100.0, tx_pct),
        "est_total_txs": est_total_txs,
        "est_new_txs_since_update": est_new_txs,
        "seconds_since_update": max(0, now_ts - last_updated),
    }


def load_seen_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["hash", "from", "to", "raw_value"], dtype=str)
        return set(zip(df["hash"], df["from"], df["to"], df["raw_value"]))
    except Exception:
        return set()


def indexer_worker(addr: str, pause_event: threading.Event) -> None:
    addr_lower = addr.lower()
    state = load_state()
    coin = state.get(addr_lower, {
        "address": addr,
        "symbol": "UNKNOWN",
        "name": "Unidentified Token",
        "first_tx_ts": None,
        "last_tx_ts": None,
        "last_block": 0,
        "page_cursor": 1,
        "collected_count": 0,
        "status": "RUNNING",
        "csv_file": f"{addr_lower[:10]}_txs.csv",
        "last_updated_ts": int(time.time()),
    })

    coin["status"] = "RUNNING"
    state[addr_lower] = coin
    save_state(state)

    csv_path = os.path.join(DATA_DIR, coin["csv_file"])
    seen_keys = load_seen_keys(csv_path)
    append_worker_log(addr_lower, f"INITIALIZING SWEEP FROM BLOCK {coin['last_block']}...")

    offset = 1000
    current_block = coin["last_block"]
    page = coin.get("page_cursor", 1)
    session = requests.Session()

    while not pause_event.is_set():
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": addr,
            "startblock": current_block,
            "endblock": 99999999,
            "page": page,
            "offset": offset,
            "sort": "asc",
            "apikey": ROUTESCAN_API_KEY,
        }

        try:
            resp = session.get(ROUTESCAN_API_URL, params=params, timeout=20)
            if resp.status_code == 429:
                append_worker_log(addr_lower, "RATE LIMITED (429) — BACKING OFF 2S")
                time.sleep(2)
                continue
            payload = resp.json()
        except Exception as exc:
            append_worker_log(addr_lower, f"RPC/API ERROR: {exc} — RETRYING IN 3S")
            time.sleep(3)
            continue

        status_flag = payload.get("status")
        results = payload.get("result", [])

        if status_flag != "1" or not isinstance(results, list) or len(results) == 0:
            if coin["collected_count"] > 0:
                coin["status"] = "SYNCED"
                coin["last_updated_ts"] = int(time.time())
                state[addr_lower] = coin
                save_state(state)
                append_worker_log(addr_lower, "CHAIN HEAD REACHED. ALL AVAILABLE TXS ARCHIVED.")
            else:
                coin["status"] = "ERROR"
                state[addr_lower] = coin
                save_state(state)
                append_worker_log(addr_lower, f"NO TRANSFERS FOUND ({payload.get('message', 'EMPTY')}).")
            break

        if coin["symbol"] == "UNKNOWN" and results[0].get("tokenSymbol"):
            coin["symbol"] = results[0].get("tokenSymbol", "TOKEN")
            coin["name"] = results[0].get("tokenName", coin["symbol"])
            safe_sym = "".join(c for c in coin["symbol"] if c.isalnum()) or "TOKEN"
            new_csv_name = f"{safe_sym}_{addr_lower[:8]}.csv"
            new_csv_path = os.path.join(DATA_DIR, new_csv_name)
            if os.path.exists(csv_path) and csv_path != new_csv_path:
                os.replace(csv_path, new_csv_path)
            coin["csv_file"] = new_csv_name
            csv_path = new_csv_path

        new_rows = []
        max_block_in_batch = current_block

        for tx in results:
            blk = int(tx.get("blockNumber", 0))
            ts = int(tx.get("timeStamp", 0))
            if blk > max_block_in_batch:
                max_block_in_batch = blk

            key = (tx.get("hash", ""), tx.get("from", ""), tx.get("to", ""), str(tx.get("value", "0")))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            decimals = int(tx.get("tokenDecimal") or 18)
            raw_val = int(tx.get("value") or 0)
            amount = raw_val / (10 ** decimals)

            if coin["first_tx_ts"] is None or (ts > 0 and ts < coin["first_tx_ts"]):
                coin["first_tx_ts"] = ts
            if coin["last_tx_ts"] is None or ts > coin["last_tx_ts"]:
                coin["last_tx_ts"] = ts

            new_rows.append({
                "timestamp_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "unix_ts": ts,
                "blockNumber": blk,
                "hash": tx.get("hash"),
                "from": tx.get("from"),
                "to": tx.get("to"),
                "tokenSymbol": coin["symbol"],
                "amount": amount,
                "raw_value": str(raw_val),
                "gasUsed": tx.get("gasUsed"),
                "gasPrice": tx.get("gasPrice"),
            })

        if new_rows:
            df_batch = pd.DataFrame(new_rows)
            write_header = not os.path.exists(csv_path)
            df_batch.to_csv(csv_path, mode="a", header=write_header, index=False)
            coin["collected_count"] += len(new_rows)
            if addr_lower in workers:
                workers[addr_lower]["session_fetched"] = workers[addr_lower].get("session_fetched", 0) + len(new_rows)

        if len(results) < offset:
            coin["last_block"] = max_block_in_batch
            coin["page_cursor"] = 1
            coin["status"] = "SYNCED"
            coin["last_updated_ts"] = int(time.time())
            state[addr_lower] = coin
            save_state(state)
            append_worker_log(
                addr_lower,
                f"+{len(new_rows)} TXS (BLK {max_block_in_batch}) — REACHED LATEST BLOCK.",
            )
            break
        else:
            if max_block_in_batch > current_block:
                current_block = max_block_in_batch
                page = 1
            else:
                page += 1

            coin["last_block"] = current_block
            coin["page_cursor"] = page
            coin["last_updated_ts"] = int(time.time())
            state[addr_lower] = coin
            save_state(state)
            append_worker_log(
                addr_lower,
                f"CAPTURED +{len(new_rows)} TXS | TOTAL: {coin['collected_count']:,} | BLK {current_block}",
            )

        for _ in range(2):
            if pause_event.is_set():
                break
            time.sleep(0.1)

    if pause_event.is_set():
        state = load_state()
        if addr_lower in state:
            state[addr_lower]["status"] = "PAUSED"
            state[addr_lower]["last_updated_ts"] = int(time.time())
            save_state(state)
        append_worker_log(addr_lower, "EXECUTION HALTED BY OPERATOR. STATE & CSV WRITTEN TO DISK.")


# --- BACKGROUND ML PIPELINE RUNNER ---

def run_ml_pipeline_worker(target_address: str = None, stage: str = "all"):
    """Background worker to transform blockchain transfers and/or run ML pipeline stages for any token."""
    addr_clean = (target_address or config.DEFAULT_NOCHILL_ADDRESS).strip().lower()
    stage_clean = (stage or "all").strip().lower()

    state = load_state()
    coin = state.get(addr_clean, {})
    symbol = coin.get("symbol", "TOKEN")
    csv_name = coin.get("csv_file")

    with pipeline_lock:
        pipeline_state["status"] = "RUNNING"
        pipeline_state["address"] = addr_clean
        pipeline_state["stage"] = stage_clean
        pipeline_state["step"] = f"Preparing {stage_clean.upper()} pipeline for {symbol}..."
        pipeline_state["progress"] = 5
        pipeline_state["started_at"] = time.time()
        pipeline_state["completed_at"] = None
        pipeline_state["error"] = None
        pipeline_state["logs"] = []

    append_pipeline_log(f"INITIATING {stage_clean.upper()} FOR {symbol} ({addr_clean})...")
    try:
        config.set_active_token_paths(addr_clean)

        # Stage 1: Ingest & Transform
        if stage_clean in ("all", "sourcing"):
            with pipeline_lock:
                pipeline_state["step"] = f"Reconstructing Prices & On-Chain Data for {symbol}..."
                pipeline_state["progress"] = 15 if stage_clean == "all" else 35

            csv_file = Path(DATA_DIR) / csv_name if csv_name else (config.BASE_DIR / "data" / "NOCHILL_0xacfb89.csv")
            if not csv_file.exists() and addr_clean == config.DEFAULT_NOCHILL_ADDRESS:
                csv_file = config.HYDROGEN_NOCHILL_CSV

            raw_df = sourcing_blockchain.load_raw_blockchain_transfers(csv_file)
            append_pipeline_log(f"Loaded {len(raw_df):,} transfers for {symbol}. Reconstructing DEX trades...")
            sourcing_blockchain.transform_transfers_to_prices_and_features(
                raw_df, token_address=addr_clean, symbol=symbol
            )
            append_pipeline_log("Reconstructed DEX trades, 1h price grid, and wallet PnL cohorts.")

        # Stage 2: Feature Engineering
        if stage_clean in ("all", "features"):
            with pipeline_lock:
                pipeline_state["step"] = f"Building Multi-Timeframe Features for {symbol}..."
                pipeline_state["progress"] = 40 if stage_clean == "all" else 50
            append_pipeline_log("Running feature engineering (duneml_s2_features.py)...")
            import duneml_s2_features
            duneml_s2_features.build_feature_pipeline()
            append_pipeline_log("Generated 117 features and 12h dashboard payload.")

        # Stage 3: EDA & Manifold Generation
        if stage_clean in ("all", "eda"):
            with pipeline_lock:
                pipeline_state["step"] = f"Running Statistical Analysis & Manifolds for {symbol}..."
                pipeline_state["progress"] = 65 if stage_clean == "all" else 60
            append_pipeline_log("Running statistical analysis (duneml_s3_eda.py)...")
            import duneml_s3_eda
            duneml_s3_eda.run_eda_pipeline()
            append_pipeline_log("Saved eda_features.json with 3D manifolds and correlation matrix.")

        # Stage 4: Multi-Model Walk-Forward Simulation
        if stage_clean in ("all", "train"):
            with pipeline_lock:
                pipeline_state["step"] = f"Training & Backtesting 6 ML Models for {symbol}..."
                pipeline_state["progress"] = 85 if stage_clean == "all" else 70
            append_pipeline_log("Running model training & backtests (duneml_s4_train_sim.py)...")
            import duneml_s4_train_sim
            duneml_s4_train_sim.train_and_simulate_all()
            append_pipeline_log("All 6 models trained and backtested across walk-forward windows.")

        with pipeline_lock:
            pipeline_state["status"] = "COMPLETED"
            pipeline_state["step"] = f"{symbol} ({stage_clean.upper()}) Complete"
            pipeline_state["progress"] = 100
            pipeline_state["completed_at"] = time.time()
        append_pipeline_log(f"PIPELINE {stage_clean.upper()} FINISHED SUCCESSFULLY.")

    except Exception as exc:
        import traceback
        err_str = f"{exc}\n{traceback.format_exc()}"
        print(f"[PIPELINE ERROR]: {err_str}", file=sys.stderr)
        with pipeline_lock:
            pipeline_state["status"] = "ERROR"
            pipeline_state["step"] = f"Pipeline Failed: {exc}"
            pipeline_state["error"] = str(exc)
        append_pipeline_log(f"ERROR: {exc}")


# --- FLASK ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/coins", methods=["GET"])
def list_coins():
    state = load_state()
    enriched = []
    for addr_lower, coin in state.items():
        is_alive = addr_lower in workers and workers[addr_lower]["thread"].is_alive()
        if coin.get("status") == "RUNNING" and not is_alive:
            coin["status"] = "PAUSED"

        metrics = compute_completion_metrics(coin)
        logs = workers[addr_lower]["logs"] if addr_lower in workers else []
        enriched.append({**coin, "metrics": metrics, "logs": logs})

    enriched.sort(key=lambda x: x.get("last_updated_ts", 0), reverse=True)
    return jsonify(enriched)


@app.route("/api/start", methods=["POST"])
def start_collection():
    data = request.get_json(force=True)
    raw_addr = (data.get("address") or "").strip()
    if not raw_addr.startswith("0x") or len(raw_addr) != 42:
        return jsonify({"error": "Invalid Avalanche C-Chain 0x contract address."}), 400

    addr_lower = raw_addr.lower()
    if addr_lower in workers and workers[addr_lower]["thread"].is_alive():
        return jsonify({"status": "ALREADY_RUNNING", "address": raw_addr})

    pause_event = threading.Event()
    t = threading.Thread(target=indexer_worker, args=(raw_addr, pause_event), daemon=True)
    workers[addr_lower] = {"thread": t, "pause_event": pause_event, "logs": [], "session_fetched": 0}
    t.start()

    return jsonify({"status": "STARTED", "address": raw_addr})


@app.route("/api/pause", methods=["POST"])
def pause_collection():
    data = request.get_json(force=True)
    raw_addr = (data.get("address") or "").strip().lower()
    session_fetched = 0
    if raw_addr in workers:
        session_fetched = workers[raw_addr].get("session_fetched", 0)
        workers[raw_addr]["pause_event"].set()

    state = load_state()
    total_saved = 0
    symbol = "TOKEN"
    if raw_addr in state:
        state[raw_addr]["status"] = "PAUSED"
        state[raw_addr]["last_updated_ts"] = int(time.time())
        save_state(state)
        total_saved = state[raw_addr].get("collected_count", 0)
        symbol = state[raw_addr].get("symbol", "TOKEN")

    return jsonify({
        "status": "PAUSED",
        "address": raw_addr,
        "symbol": symbol,
        "session_fetched": session_fetched,
        "total_saved": total_saved,
    })


@app.route("/api/profile/<address>", methods=["GET"])
def get_token_profile(address: str):
    state = load_state()
    coin = state.get(address.lower())
    if not coin:
        return jsonify({"error": "Token not found in state.json"}), 404

    csv_path = os.path.join(DATA_DIR, coin["csv_file"])
    if not os.path.exists(csv_path):
        return jsonify({"error": f"CSV file missing: {coin['csv_file']}"}), 404

    price_param = request.args.get("price")
    override_price = float(price_param) if (price_param is not None and price_param != "") else None
    limit = min(1000, max(50, int(request.args.get("limit", 300))))

    profile = profile_token_csv(
        csv_path=csv_path,
        token_address=coin["address"],
        override_price_usd=override_price,
        limit=limit,
    )
    return jsonify({
        "address": coin["address"],
        "name": coin["name"],
        "symbol": coin["symbol"],
        **profile,
    })


@app.route("/api/download/<address>", methods=["GET"])
def download_csv(address: str):
    state = load_state()
    coin = state.get(address.lower())
    if not coin:
        return jsonify({"error": "Token not found"}), 404
    csv_path = os.path.join(DATA_DIR, coin["csv_file"])
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not generated yet"}), 404
    return send_file(csv_path, as_attachment=True, download_name=coin["csv_file"])


# --- ML PIPELINE & ANALYSIS API ENDPOINTS ---

@app.route("/api/run_pipeline", methods=["POST"])
def trigger_pipeline():
    """Trigger specific stage or full pipeline retraining on latest data for a token."""
    with pipeline_lock:
        if pipeline_state["status"] == "RUNNING":
            return jsonify({"status": "ALREADY_RUNNING", "message": "Pipeline is already in progress."}), 409

    payload = request.get_json(silent=True) or {}
    target_address = payload.get("address") or config.DEFAULT_NOCHILL_ADDRESS
    stage = payload.get("stage") or "all"

    t = threading.Thread(target=run_ml_pipeline_worker, args=(target_address, stage), daemon=True)
    t.start()
    return jsonify({
        "status": "STARTED",
        "address": target_address,
        "stage": stage,
        "message": f"Pipeline ({stage}) started in background."
    })


@app.route("/api/pipeline_status", methods=["GET"])
def get_pipeline_status():
    """Poll the status and logs of the running pipeline."""
    with pipeline_lock:
        return jsonify(pipeline_state)


@app.route("/api/data/simulation/<model>", methods=["GET"])
def get_simulation_payload(model: str):
    """Returns the simulation JSON for the requested model and optional token address."""
    safe_model = model.lower()
    address = request.args.get("address")
    _, boards_dir = config.get_token_dirs(address)
    file_path = boards_dir / f"{safe_model}_simulation_{config.DUNE_ID}.json"
    if not file_path.exists():
        return jsonify({"error": f"Simulation payload for {safe_model} not found.", "not_ready": True}), 404
    with open(file_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/data/features", methods=["GET"])
def get_features_payload():
    """Returns the 12h downsampled feature dashboard payload for optional token address."""
    address = request.args.get("address")
    _, boards_dir = config.get_token_dirs(address)
    file_path = boards_dir / f"dashboard_features_{config.DUNE_ID}.json"
    if not file_path.exists():
        return jsonify({"error": "Features payload not found.", "not_ready": True}), 404
    with open(file_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/data/eda", methods=["GET"])
def get_eda_payload():
    """Returns the complete EDA payload (correlations, manifolds, histograms) for optional token address."""
    address = request.args.get("address")
    _, boards_dir = config.get_token_dirs(address)
    file_path = boards_dir / "eda_features.json"
    if not file_path.exists():
        return jsonify({"error": "EDA payload not found.", "not_ready": True}), 404
    with open(file_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/dashboards/<path:filename>")
def serve_dashboards(filename):
    """Direct static access to standalone dashboard HTMLs if needed."""
    return send_from_directory(str(DASHBOARDS_DIR), filename)


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"[AVAX-QUANT-TERMINAL] Local server spinning up at {url}")
    app.run(host="127.0.0.1", port=5000, debug=False)
