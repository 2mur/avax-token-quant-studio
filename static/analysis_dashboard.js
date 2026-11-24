const { useState, useEffect, useMemo, useRef } = React;
const {
  ResponsiveContainer, ComposedChart, LineChart, BarChart, AreaChart,
  ScatterChart, Scatter, Line, Bar, Area, XAxis, YAxis, ZAxis, Tooltip,
  Legend, CartesianGrid, ReferenceLine, Brush, Cell
} = window.Recharts;

const MODELS = ["XGBOOST", "LGBM", "ESN", "LSTM", "TRANSFORMER", "SEASONAL"];
const PERIODS = ["3d", "7d", "14d", "30d", "60d"];

// Custom Trade Cross Marker Component (Only data points are colored)
const TradeCrossDot = (props) => {
  const { cx, cy, payload } = props;
  if (!payload || payload.action === "HOLD") return null;
  const isBuy = payload.action === "BUY";
  const color = isBuy ? "var(--data-green)" : "var(--data-red)";
  const size = 6;
  return (
    <g>
      <line x1={cx - size} y1={cy} x2={cx + size} y2={cy} stroke={color} strokeWidth={2.5} />
      <line x1={cx} y1={cy - size} x2={cx} y2={cy + size} stroke={color} strokeWidth={2.5} />
      <circle cx={cx} cy={cy} r={size + 2} stroke={color} strokeWidth={1} fill="none" opacity={0.6} />
    </g>
  );
};

function QuantAnalysisRoot() {
  // Active Token State (updated when user clicks a token card)
  const [selectedToken, setSelectedToken] = useState(
    window.selectedToken || {
      address: "0xAcFb898Cff266E53278cC0124fC2C7C94C8cB9a5",
      symbol: "NOCHILL",
      name: "AVAX HAS NO CHILL"
    }
  );

  useEffect(() => {
    const onTokenSelected = (e) => {
      if (e.detail && e.detail.address) {
        setSelectedToken(e.detail);
      }
    };
    window.addEventListener("tokenSelected", onTokenSelected);
    return () => window.removeEventListener("tokenSelected", onTokenSelected);
  }, []);

  // Analysis Navigation: 'terminal', 'features_ts', 'eda'
  const [analysisView, setAnalysisView] = useState("terminal");

  // Pipeline execution state
  const [pipelineStatus, setPipelineStatus] = useState({ status: "IDLE", step: "", progress: 0, logs: [] });
  const [isTriggering, setIsTriggering] = useState(false);
  const [dataMissingNotice, setDataMissingNotice] = useState(false);

  // --- Terminal View State ---
  const [selectedModel, setSelectedModel] = useState("XGBOOST");
  const [selectedWindow, setSelectedWindow] = useState("window_1");
  const [simData, setSimData] = useState(null);
  const [availableWindows, setAvailableWindows] = useState(["window_1", "window_2", "window_3"]);
  const [loadingSim, setLoadingSim] = useState(true);

  // Terminal Bottom Row Toggles
  const [oscType, setOscType] = useState("RSI");
  const [oscPeriod, setOscPeriod] = useState("14D");
  const [onChainMetric, setOnChainMetric] = useState("USERS");
  const [macroMetric, setMacroMetric] = useState("SHOCK");

  // --- Features TS View State ---
  const [featTsData, setFeatTsData] = useState([]);
  const [priceScale, setPriceScale] = useState("LINEAR");
  const [valPeriod, setValPeriod] = useState("14d");
  const [showMA, setShowMA] = useState(true);
  const [showBB, setShowBB] = useState(true);
  const [rsiPeriod, setRsiPeriod] = useState("14d");
  const [rsiMode, setRsiMode] = useState("BASE");
  const [stochPeriod, setStochPeriod] = useState("14d");
  const [stochMode, setStochMode] = useState("BASE");
  const [rocPeriod, setRocPeriod] = useState("14d");
  const [rocMode, setRocMode] = useState("BASE");
  const [volMetric, setVolMetric] = useState("TX");
  const [volMode, setVolMode] = useState("BASE");

  // --- EDA View State ---
  const [edaData, setEdaData] = useState(null);
  const [activeEdaTab, setActiveEdaTab] = useState("scatter");
  const [activeTarget, setActiveTarget] = useState("future_return_1h");
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [featureSearch, setFeatureSearch] = useState("");
  const [histMode, setHistMode] = useState("GROUPED");
  const [pcaFilter, setPcaFilter] = useState("ALL");
  const [selectedPcaPair, setSelectedPcaPair] = useState(null);
  const [selectedManifoldPair, setSelectedManifoldPair] = useState(null);
  const [corrFeature, setCorrFeature] = useState(null);

  // 1. Poll Pipeline Status
  const prevStatusRef = useRef("IDLE");
  const fetchPipelineStatus = () => {
    fetch("/api/pipeline_status")
      .then((r) => r.json())
      .then((data) => {
        setPipelineStatus(data);
        if (prevStatusRef.current === "RUNNING" && data.status === "COMPLETED") {
          loadSimulationData();
          loadFeaturesTsData();
          loadEdaData();
        }
        prevStatusRef.current = data.status;
      })
      .catch((err) => console.error("Pipeline poll error:", err));
  };

  useEffect(() => {
    fetchPipelineStatus();
    const interval = setInterval(fetchPipelineStatus, 2500);
    return () => clearInterval(interval);
  }, [selectedToken.address, selectedModel]);

  // 2. Trigger Pipeline Stage or Full Pipeline
  const triggerPipelineStage = (stage = "all") => {
    setIsTriggering(true);
    fetch("/api/run_pipeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ address: selectedToken.address, stage })
    })
      .then((r) => r.json())
      .then(() => {
        setIsTriggering(false);
        fetchPipelineStatus();
      })
      .catch((err) => {
        console.error("Pipeline trigger error:", err);
        setIsTriggering(false);
      });
  };

  // 3. Load Simulation Payload
  const loadSimulationData = () => {
    setLoadingSim(true);
    const addrQ = encodeURIComponent(selectedToken.address);
    fetch(`/api/data/simulation/${selectedModel.toLowerCase()}?address=${addrQ}`)
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok || d.not_ready || d.error) {
          setSimData(null);
          setDataMissingNotice(true);
          setLoadingSim(false);
          return;
        }
        setDataMissingNotice(false);
        setSimData(d);
        const wins = Object.keys(d);
        setAvailableWindows(wins);
        if (!wins.includes(selectedWindow)) {
          setSelectedWindow(wins[0] || "window_1");
        }
        setLoadingSim(false);
      })
      .catch((err) => {
        console.error("Simulation load error:", err);
        setLoadingSim(false);
      });
  };

  const loadFeaturesTsData = () => {
    const addrQ = encodeURIComponent(selectedToken.address);
    fetch(`/api/data/features?address=${addrQ}`)
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok || d.not_ready || d.error) {
          setFeatTsData([]);
          return;
        }
        const rows = (d.data || []).slice(-200);
        setFeatTsData(rows);
      })
      .catch((err) => console.error("Features TS error:", err));
  };

  const loadEdaData = () => {
    const addrQ = encodeURIComponent(selectedToken.address);
    fetch(`/api/data/eda?address=${addrQ}`)
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok || d.not_ready || d.error) {
          setEdaData(null);
          return;
        }
        setEdaData(d);
        if (d.targets?.length > 0) setActiveTarget(d.targets[0]);
        if (d.features?.length > 0) {
          setSelectedFeature(d.features[0]);
          setCorrFeature(d.features[0]);
        }
        if (d.r2_groups?.pca_base?.length > 0) setSelectedPcaPair(d.r2_groups.pca_base[0]);
        if (d.r2_groups?.umap_phate?.length > 0) setSelectedManifoldPair(d.r2_groups.umap_phate[0]);
      })
      .catch((err) => console.error("EDA load error:", err));
  };

  useEffect(() => {
    loadSimulationData();
  }, [selectedModel, selectedToken.address]);

  useEffect(() => {
    loadFeaturesTsData();
    loadEdaData();
  }, [selectedToken.address]);

  // Current Window Data
  const currentWindow = useMemo(() => {
    if (!simData || !simData[selectedWindow]) return null;
    return simData[selectedWindow];
  }, [simData, selectedWindow]);

  const summary = currentWindow?.summary || {
    starting_balance: 10000,
    final_balance: 10000,
    pnl_pct: 0,
    alpha_pct: 0,
    total_trades: 0
  };
  const history = currentWindow?.history || [];
  const trades = currentWindow?.trades || [];

  // Terminal Bottom Panel Keys
  const currentOscKey = useMemo(() => {
    const p = oscPeriod.toLowerCase();
    if (oscType === "RSI") return `rsi_${p}`;
    if (oscType === "STOCH") return `stoch_${p}`;
    return `roc_${p}`;
  }, [oscType, oscPeriod]);

  const currentOnChainKey = useMemo(() => {
    if (onChainMetric === "USERS") return "active_users";
    if (onChainMetric === "NET TOK") return "net_tokens";
    return "transfer_tokens_sold";
  }, [onChainMetric]);

  const currentMacroKey = useMemo(() => {
    if (macroMetric === "SHOCK") return "vol_shock";
    if (macroMetric === "NUPL_14D") return "nupl_14d";
    return "nupl_30d";
  }, [macroMetric]);

  return (
    <div>
      {/* Main Action Bar: Separate Task Buttons + Run All Stages */}
      <div className="content-block" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 15 }}>
          <div style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 8 }}>
              ANALYSIS TARGET: <strong style={{ color: 'var(--text-color)' }}>{selectedToken.symbol}</strong> ({selectedToken.address})
            </div>

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <button
                className="run-pipeline-btn"
                onClick={() => triggerPipelineStage("all")}
                disabled={pipelineStatus.status === "RUNNING" || isTriggering}
              >
                {pipelineStatus.status === "RUNNING" ? "PIPELINE RUNNING..." : "RUN ALL STAGES"}
              </button>
              <button
                className="mono-btn"
                onClick={() => triggerPipelineStage("sourcing")}
                disabled={pipelineStatus.status === "RUNNING" || isTriggering}
              >
                1. Reconstruct Prices
              </button>
              <button
                className="mono-btn"
                onClick={() => triggerPipelineStage("features")}
                disabled={pipelineStatus.status === "RUNNING" || isTriggering}
              >
                2. Build Features
              </button>
              <button
                className="mono-btn"
                onClick={() => triggerPipelineStage("eda")}
                disabled={pipelineStatus.status === "RUNNING" || isTriggering}
              >
                3. Statistical Analysis
              </button>
              <button
                className="mono-btn"
                onClick={() => triggerPipelineStage("train")}
                disabled={pipelineStatus.status === "RUNNING" || isTriggering}
              >
                4. Train & Backtest Models
              </button>
            </div>
          </div>

          {/* Status Indicator */}
          <div style={{ textAlign: 'right', minWidth: 240 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end', marginBottom: 6 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>PIPELINE:</span>
              <span className="status-badge" style={{ borderColor: 'var(--text-color)' }}>
                {pipelineStatus.status}
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {pipelineStatus.step || "Ready"}
            </div>
          </div>
        </div>

        {/* Live Progress Bar */}
        {pipelineStatus.status === "RUNNING" && (
          <div style={{ marginTop: 15 }}>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${pipelineStatus.progress}%` }}></div>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
              <span>{pipelineStatus.step}</span>
              <span>{pipelineStatus.progress}%</span>
            </div>
          </div>
        )}

        {/* Recent Pipeline Logs */}
        {pipelineStatus.logs && pipelineStatus.logs.length > 0 && (
          <div style={{ marginTop: 12, padding: '8px 10px', background: 'var(--bg-color)', border: '1px solid var(--border-color)', fontSize: '0.72rem', color: 'var(--text-muted)', maxHeight: 80, overflowY: 'auto' }}>
            {pipelineStatus.logs.slice(-4).map((line, idx) => (
              <div key={idx}>{line}</div>
            ))}
          </div>
        )}
      </div>

      {/* Notice if selected token has not been analyzed yet */}
      {dataMissingNotice && (
        <div className="content-block" style={{ marginBottom: 20, borderColor: '#faa094' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#ffb4a8', marginBottom: 4 }}>
            NO ML ANALYSIS FOUND FOR {selectedToken.symbol}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Click <strong>RUN ALL STAGES</strong> above to reconstruct prices from on-chain transfers, build indicators, run statistical analysis, and backtest all 6 models for {selectedToken.symbol}.
          </div>
        </div>
      )}

      {/* Analysis View Sub-Navigation */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '1px solid var(--border-color)', paddingBottom: 12, flexWrap: 'wrap' }}>
        <button
          className={`mono-btn ${analysisView === "terminal" ? "mono-btn-active" : ""}`}
          onClick={() => setAnalysisView("terminal")}
        >
          1. Model Performance & Signals
        </button>
        <button
          className={`mono-btn ${analysisView === "features_ts" ? "mono-btn-active" : ""}`}
          onClick={() => setAnalysisView("features_ts")}
        >
          2. Price & Indicator Time Series
        </button>
        <button
          className={`mono-btn ${analysisView === "eda" ? "mono-btn-active" : ""}`}
          onClick={() => setAnalysisView("eda")}
        >
          3. Feature Analysis & Correlations
        </button>
      </div>

      {/* SUB-VIEW 1: SOP PERFORMANCE TERMINAL */}
      {analysisView === "terminal" && (
        <div>
          {/* Model & Window Selectors */}
          <div className="content-block" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>AI MODEL:</span>
              {MODELS.map((m) => (
                <button
                  key={m}
                  onClick={() => setSelectedModel(m)}
                  className={`mono-btn ${selectedModel === m ? "mono-btn-active" : ""}`}
                >
                  {m}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>WALK-FORWARD:</span>
              {availableWindows.map((w) => (
                <button
                  key={w}
                  onClick={() => setSelectedWindow(w)}
                  className={`mono-btn ${selectedWindow === w ? "mono-btn-active" : ""}`}
                >
                  [{w.toUpperCase()}]
                </button>
              ))}
            </div>
          </div>

          {/* 5 KPI Cards (Monochrome containers, Colored data points) */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>STARTING CAPITAL</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4 }}>
                ${summary.starting_balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>BASE ALLOCATION</div>
            </div>

            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>AI PORTFOLIO VALUE</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4 }}>
                ${summary.final_balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>MARK-TO-MARKET</div>
            </div>

            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>NET RETURN</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4, color: summary.pnl_pct >= 0 ? 'var(--data-green)' : 'var(--data-red)' }}>
                {summary.pnl_pct >= 0 ? "+" : ""}{summary.pnl_pct.toFixed(2)}%
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>TOTAL RETURN</div>
            </div>

            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>ALPHA (VS HODL)</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4, color: summary.alpha_pct >= 0 ? 'var(--data-green)' : 'var(--data-red)' }}>
                {summary.alpha_pct >= 0 ? "+" : ""}{summary.alpha_pct.toFixed(2)}%
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>VS PASSIVE HOLD</div>
            </div>

            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>EXECUTED SIGNALS</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4 }}>
                {summary.total_trades} TRADES
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>BUY & SELL SIGNALS</div>
            </div>
          </div>

          {/* Master Synchronized Chart + Execution Log Table */}
          <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: 16, marginBottom: 20 }}>
            {/* Master Chart */}
            <div className="content-block" style={{ margin: 0, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700 }}>
                  AI PORTFOLIO EQUITY VS HODL BENCHMARK & PRICE
                </div>
                <div style={{ display: 'flex', gap: 14, fontSize: '0.75rem' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 10, height: 2, background: '#ffffff' }}></span>
                    <span>PORTFOLIO ($)</span>
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 10, height: 2, borderTop: '1px dashed #888888' }}></span>
                    <span>HODL BENCH ($)</span>
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 10, height: 2, background: 'var(--data-blue)' }}></span>
                    <span style={{ color: 'var(--data-blue)' }}>PRICE ($)</span>
                  </span>
                </div>
              </div>

              <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={history} syncId="masterSync">
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="time"
                      tick={{ fill: '#888888', fontSize: 10 }}
                      tickFormatter={(t) => t.slice(5, 16)}
                      stroke="#444444"
                    />
                    <YAxis
                      yAxisId="valAxis"
                      tick={{ fill: '#f0f0f0', fontSize: 10 }}
                      stroke="#444444"
                      domain={['auto', 'auto']}
                      tickFormatter={(v) => `$${v.toFixed(0)}`}
                    />
                    <YAxis
                      yAxisId="priceAxis"
                      orientation="right"
                      tick={{ fill: 'var(--data-blue)', fontSize: 10 }}
                      stroke="#444444"
                      domain={['auto', 'auto']}
                      tickFormatter={(p) => `$${p.toFixed(6)}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1b1b1b',
                        borderColor: '#555555',
                        fontSize: '11px',
                        fontFamily: 'Roboto Mono'
                      }}
                    />
                    <Line
                      yAxisId="valAxis"
                      type="stepAfter"
                      dataKey="portfolio_value"
                      name="Portfolio Value"
                      stroke="#ffffff"
                      strokeWidth={2}
                      dot={<TradeCrossDot />}
                    />
                    <Line
                      yAxisId="valAxis"
                      type="monotone"
                      dataKey="benchmark_value"
                      name="Benchmark HODL"
                      stroke="#888888"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                    <Line
                      yAxisId="priceAxis"
                      type="monotone"
                      dataKey="price"
                      name="Price"
                      stroke="var(--data-blue)"
                      strokeWidth={1.2}
                      opacity={0.8}
                      dot={false}
                    />
                    <Brush
                      dataKey="time"
                      height={22}
                      stroke="#666666"
                      fill="#181818"
                      tickFormatter={(t) => t.slice(11, 16)}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Execution Log Table */}
            <div className="content-block" style={{ margin: 0, padding: 16, display: 'flex', flexDirection: 'column', maxHeight: 390 }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 8, marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                <span>EXECUTION LOG</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{trades.length} SIGNALS</span>
              </div>
              <div style={{ overflowY: 'auto', flex: 1 }}>
                {trades.length === 0 ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: 15, textAlign: 'center' }}>NO TRADES IN THIS WINDOW</div>
                ) : (
                  <table style={{ width: '100%', fontSize: '0.75rem', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', textAlign: 'left' }}>
                        <th style={{ padding: 4 }}>ACT</th>
                        <th style={{ padding: 4 }}>PRICE</th>
                        <th style={{ padding: 4, textAlign: 'right' }}>CONF</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trades.map((tr, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: 4, fontWeight: 'bold', color: tr.type === 'BUY' ? 'var(--data-green)' : 'var(--data-red)' }}>
                            {tr.type}
                          </td>
                          <td style={{ padding: 4, fontFamily: 'monospace' }}>${tr.price.toFixed(6)}</td>
                          <td style={{ padding: 4, textAlign: 'right', color: 'var(--text-muted)' }}>{tr.conf}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </div>

          {/* Bottom 3 Synchronized Panels */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            {/* Panel 1: Oscillators */}
            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>OSCILLATORS</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{currentOscKey}</span>
              </div>
              <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
                {["RSI", "STOCH", "ROC"].map((typ) => (
                  <button
                    key={typ}
                    onClick={() => setOscType(typ)}
                    className={`mono-btn ${oscType === typ ? "mono-btn-active" : ""}`}
                    style={{ padding: '2px 8px', fontSize: '0.7rem', flex: 1 }}
                  >
                    {typ}
                  </button>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 4, marginBottom: 10 }}>
                {["3D", "7D", "14D", "30D", "60D"].map((per) => (
                  <button
                    key={per}
                    onClick={() => setOscPeriod(per)}
                    className={`mono-btn ${oscPeriod === per ? "mono-btn-active" : ""}`}
                    style={{ padding: '2px 6px', fontSize: '0.68rem', flex: 1 }}
                  >
                    {per}
                  </button>
                ))}
              </div>
              <div style={{ width: '100%', height: 160 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={history} syncId="masterSync">
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(11, 16)} stroke="#444444" />
                    <YAxis tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" domain={oscType === "ROC" ? ['auto', 'auto'] : [0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                    {oscType !== "ROC" ? (
                      <>
                        <ReferenceLine y={70} stroke="var(--data-red)" strokeDasharray="3 3" opacity={0.6} />
                        <ReferenceLine y={30} stroke="var(--data-green)" strokeDasharray="3 3" opacity={0.6} />
                      </>
                    ) : (
                      <ReferenceLine y={0} stroke="#ffffff" strokeDasharray="3 3" opacity={0.4} />
                    )}
                    <Line type="monotone" dataKey={currentOscKey} stroke="var(--data-green)" strokeWidth={1.5} dot={<TradeCrossDot />} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Panel 2: On-Chain Flow */}
            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>ON-CHAIN ACTIVITY</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{currentOnChainKey}</span>
              </div>
              <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
                {[{ label: "USERS", key: "USERS" }, { label: "NET TOK", key: "NET TOK" }, { label: "SOLD", key: "SOLD" }].map((item) => (
                  <button
                    key={item.key}
                    onClick={() => setOnChainMetric(item.key)}
                    className={`mono-btn ${onChainMetric === item.key ? "mono-btn-active" : ""}`}
                    style={{ padding: '4px 8px', fontSize: '0.72rem', flex: 1 }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div style={{ width: '100%', height: 185 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={history} syncId="masterSync">
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(11, 16)} stroke="#444444" />
                    <YAxis tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                    <Bar dataKey={currentOnChainKey} fill="var(--data-blue)" opacity={0.8} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Panel 3: Macro & NUPL */}
            <div className="content-block" style={{ margin: 0, padding: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>SYSTEMIC MACRO</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{currentMacroKey}</span>
              </div>
              <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
                {[{ label: "SHOCK", key: "SHOCK" }, { label: "NUPL 14D", key: "NUPL_14D" }, { label: "NUPL 30D", key: "NUPL_30D" }].map((item) => (
                  <button
                    key={item.key}
                    onClick={() => setMacroMetric(item.key)}
                    className={`mono-btn ${macroMetric === item.key ? "mono-btn-active" : ""}`}
                    style={{ padding: '4px 8px', fontSize: '0.72rem', flex: 1 }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div style={{ width: '100%', height: 185 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history} syncId="masterSync">
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(11, 16)} stroke="#444444" />
                    <YAxis tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                    {macroMetric === "SHOCK" && (
                      <ReferenceLine y={1.0} stroke="#ffffff" strokeDasharray="3 3" opacity={0.5} />
                    )}
                    <Area type="monotone" dataKey={currentMacroKey} stroke="var(--data-amber)" fill="rgba(251, 191, 36, 0.2)" strokeWidth={1.5} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 2: MULTI-HORIZON FEATURES TS */}
      {analysisView === "features_ts" && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          {/* 1. Valuation */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>1. ASSET VALUATION</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{priceScale} | MA ({valPeriod})</span>
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
              <button onClick={() => setPriceScale("LINEAR")} className={`mono-btn ${priceScale === "LINEAR" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>LIN</button>
              <button onClick={() => setPriceScale("LOG")} className={`mono-btn ${priceScale === "LOG" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>LOG</button>
              {PERIODS.map((p) => (
                <button key={p} onClick={() => setValPeriod(p)} className={`mono-btn ${valPeriod === p ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>{p.toUpperCase()}</button>
              ))}
              <button onClick={() => setShowMA(!showMA)} className={`mono-btn ${showMA ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>MA</button>
              <button onClick={() => setShowBB(!showBB)} className={`mono-btn ${showBB ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>BB</button>
            </div>
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} tickFormatter={(v) => priceScale === "LOG" ? v.toFixed(2) : `$${v.toFixed(6)}`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  <Line type="monotone" dataKey={priceScale === "LOG" ? "log_price" : "price"} stroke="#ffffff" strokeWidth={1.8} dot={false} />
                  {showMA && <Line type="monotone" dataKey={`ma_${valPeriod}`} stroke="var(--data-blue)" strokeWidth={1.2} dot={false} />}
                  {showBB && (
                    <>
                      <Line type="monotone" dataKey={`bb_up_${valPeriod}`} stroke="var(--data-green)" strokeDasharray="2 2" strokeWidth={1} dot={false} />
                      <Line type="monotone" dataKey={`bb_low_${valPeriod}`} stroke="var(--data-red)" strokeDasharray="2 2" strokeWidth={1} dot={false} />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 2. RSI */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>2. RELATIVE STRENGTH INDEX</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{rsiMode === 'VEL' ? `rsi_${rsiPeriod}_v` : `rsi_${rsiPeriod}`}</span>
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <button onClick={() => setRsiMode("BASE")} className={`mono-btn ${rsiMode === "BASE" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>BASE</button>
              <button onClick={() => setRsiMode("VEL")} className={`mono-btn ${rsiMode === "VEL" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>VELOCITY</button>
              {PERIODS.map((p) => (
                <button key={p} onClick={() => setRsiPeriod(p)} className={`mono-btn ${rsiPeriod === p ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>{p.toUpperCase()}</button>
              ))}
            </div>
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" domain={rsiMode === "BASE" ? [0, 100] : ['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  {rsiMode === "BASE" ? (
                    <>
                      <ReferenceLine y={70} stroke="var(--data-red)" strokeDasharray="3 3" opacity={0.6} />
                      <ReferenceLine y={30} stroke="var(--data-green)" strokeDasharray="3 3" opacity={0.6} />
                    </>
                  ) : (
                    <ReferenceLine y={0} stroke="#ffffff" strokeDasharray="3 3" opacity={0.4} />
                  )}
                  <Line type="monotone" dataKey={rsiMode === "VEL" ? `rsi_${rsiPeriod}_v` : `rsi_${rsiPeriod}`} stroke="var(--data-green)" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 3. Stochastic */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>3. STOCHASTIC OSCILLATOR (%K)</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{stochMode === 'VEL' ? `stoch_${stochPeriod}_v` : `stoch_${stochPeriod}`}</span>
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <button onClick={() => setStochMode("BASE")} className={`mono-btn ${stochMode === "BASE" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>BASE</button>
              <button onClick={() => setStochMode("VEL")} className={`mono-btn ${stochMode === "VEL" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>VEL</button>
              {PERIODS.map((p) => (
                <button key={p} onClick={() => setStochPeriod(p)} className={`mono-btn ${stochPeriod === p ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>{p.toUpperCase()}</button>
              ))}
            </div>
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" domain={stochMode === "BASE" ? [0, 100] : ['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  {stochMode === "BASE" && (
                    <>
                      <ReferenceLine y={80} stroke="var(--data-red)" strokeDasharray="3 3" opacity={0.6} />
                      <ReferenceLine y={20} stroke="var(--data-green)" strokeDasharray="3 3" opacity={0.6} />
                    </>
                  )}
                  <Line type="monotone" dataKey={stochMode === "VEL" ? `stoch_${stochPeriod}_v` : `stoch_${stochPeriod}`} stroke="#ffffff" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 4. ROC */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>4. RATE OF CHANGE (ROC %)</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{rocMode === 'VEL' ? `roc_${rocPeriod}_v` : `roc_${rocPeriod}`}</span>
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <button onClick={() => setRocMode("BASE")} className={`mono-btn ${rocMode === "BASE" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>BASE</button>
              <button onClick={() => setRocMode("VEL")} className={`mono-btn ${rocMode === "VEL" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>VEL</button>
              {PERIODS.map((p) => (
                <button key={p} onClick={() => setRocPeriod(p)} className={`mono-btn ${rocPeriod === p ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>{p.toUpperCase()}</button>
              ))}
            </div>
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  <ReferenceLine y={0} stroke="#ffffff" strokeDasharray="3 3" opacity={0.4} />
                  <Area type="monotone" dataKey={rocMode === "VEL" ? `roc_${rocPeriod}_v` : `roc_${rocPeriod}`} stroke="var(--data-blue)" fill="rgba(56, 189, 248, 0.2)" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 5. Network Volume */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>5. ON-CHAIN NETWORK FLOW</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{volMetric} ({volMode})</span>
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              {["TX", "USERS", "NET TOK", "SOLD"].map((m) => (
                <button key={m} onClick={() => setVolMetric(m)} className={`mono-btn ${volMetric === m ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>{m}</button>
              ))}
              <button onClick={() => setVolMode("BASE")} className={`mono-btn ${volMode === "BASE" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>BASE</button>
              <button onClick={() => setVolMode("VEL")} className={`mono-btn ${volMode === "VEL" ? "mono-btn-active" : ""}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>VEL</button>
            </div>
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  <Bar
                    dataKey={
                      volMetric === "USERS" ? (volMode === "VEL" ? "active_users_v" : "active_users") :
                      volMetric === "NET TOK" ? (volMode === "VEL" ? "net_tokens_v" : "net_tokens") :
                      volMetric === "SOLD" ? (volMode === "VEL" ? "transfer_tokens_sold_v" : "transfer_tokens_sold") :
                      (volMode === "VEL" ? "tx_count_v" : "tx_count")
                    }
                    fill="var(--data-green)"
                    opacity={0.8}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 6. Shock vs Returns */}
          <div className="content-block" style={{ margin: 0, padding: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>6. VOL SHOCK VS FORWARD RETURN</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>VOL_SHOCK × LOG_RETURN_12H</span>
            </div>
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={featTsData}>
                  <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                  <YAxis yAxisId="sAx" tick={{ fill: 'var(--data-amber)', fontSize: 9 }} stroke="#444444" domain={[0, 'auto']} />
                  <YAxis yAxisId="rAx" orientation="right" tick={{ fill: '#ffffff', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                  <ReferenceLine yAxisId="sAx" y={1.0} stroke="var(--data-amber)" strokeDasharray="2 2" opacity={0.5} />
                  <ReferenceLine yAxisId="rAx" y={0.0} stroke="#ffffff" strokeDasharray="2 2" opacity={0.5} />
                  <Bar yAxisId="sAx" dataKey="vol_shock" fill="var(--data-amber)" opacity={0.4} />
                  <Line yAxisId="rAx" type="stepAfter" dataKey="log_future_return_12h" stroke="#ffffff" strokeWidth={1.8} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 3: FEATURE EDA & MANIFOLDS */}
      {analysisView === "eda" && edaData && (
        <div>
          {/* Target Dropdown & EDA Tabs */}
          <div className="content-block" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>TARGET:</span>
              <select
                value={activeTarget}
                onChange={(e) => setActiveTarget(e.target.value)}
                style={{ background: 'var(--bg-color)', color: 'var(--text-color)', border: '1px solid var(--border-color)', padding: '4px 8px', fontFamily: 'Roboto Mono', fontSize: '0.8rem' }}
              >
                {edaData.targets?.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {[
                { id: "scatter", label: "[FEATURE VS RETURNS]" },
                { id: "hist", label: "[HISTOGRAMS]" },
                { id: "pca", label: "[PCA EMBEDDINGS]" },
                { id: "manifold", label: "[UMAP/PHATE]" },
                { id: "corr", label: "[CORRELATION]" },
                { id: "pnl_bins", label: "[PNL BINS]" }
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveEdaTab(t.id)}
                  className={`mono-btn ${activeEdaTab === t.id ? "mono-btn-active" : ""}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Scatter / Feature vs Returns */}
          {activeEdaTab === "scatter" && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 16 }}>
              <div className="content-block" style={{ margin: 0, padding: 14, maxHeight: 480, display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 6, marginBottom: 8 }}>
                  RANKED BY ROBUST R²
                </div>
                <input
                  type="text"
                  placeholder="Filter feature..."
                  value={featureSearch}
                  onChange={(e) => setFeatureSearch(e.target.value)}
                  style={{ background: 'var(--bg-color)', color: 'var(--text-color)', border: '1px solid var(--border-color)', padding: '4px 8px', fontSize: '0.75rem', fontFamily: 'monospace', marginBottom: 8 }}
                />
                <div style={{ overflowY: 'auto', flex: 1 }}>
                  {edaData.features
                    ?.filter((f) => f.toLowerCase().includes(featureSearch.toLowerCase()))
                    .map((f) => {
                      const met = edaData.feature_metrics?.[activeTarget]?.[f] || {};
                      return (
                        <div
                          key={f}
                          onClick={() => setSelectedFeature(f)}
                          style={{
                            padding: '6px 8px',
                            fontSize: '0.75rem',
                            cursor: 'pointer',
                            display: 'flex',
                            justifyContent: 'space-between',
                            background: selectedFeature === f ? 'var(--text-color)' : 'transparent',
                            color: selectedFeature === f ? 'var(--bg-color)' : 'inherit'
                          }}
                        >
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 150 }}>{f}</span>
                          <span style={{ fontWeight: 'bold', color: selectedFeature === f ? 'inherit' : 'var(--data-green)' }}>
                            {((met.robust_r2 || 0) * 100).toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>

              <div className="content-block" style={{ margin: 0, padding: 16, display: 'flex', flexDirection: 'column' }}>
                {/* Readout */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 12, textAlign: 'center' }}>
                  <div className="content-block" style={{ margin: 0, padding: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>MUTUAL INFO</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700 }}>{edaData.feature_metrics?.[activeTarget]?.[selectedFeature]?.mi?.toFixed(4) || "0.0000"}</div>
                  </div>
                  <div className="content-block" style={{ margin: 0, padding: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>STANDARD R²</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700 }}>{edaData.feature_metrics?.[activeTarget]?.[selectedFeature]?.r2?.toFixed(4) || "0.0000"}</div>
                  </div>
                  <div className="content-block" style={{ margin: 0, padding: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ROBUST R²</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--data-green)' }}>{edaData.feature_metrics?.[activeTarget]?.[selectedFeature]?.robust_r2?.toFixed(4) || "0.0000"}</div>
                  </div>
                </div>

                <div style={{ width: '100%', height: 350 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name={selectedFeature} tick={{ fill: '#888888', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} />
                      <YAxis type="number" dataKey="y" name={activeTarget} tick={{ fill: '#f0f0f0', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
                      <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                      <ReferenceLine y={0} stroke="#ffffff" strokeDasharray="3 3" opacity={0.4} />
                      <Scatter
                        name="Target > 0"
                        data={(edaData.scatter_data || []).filter(r => r[activeTarget] > 0).map(r => ({ x: r[selectedFeature], y: r[activeTarget] }))}
                        fill="var(--data-green)"
                        opacity={0.75}
                      />
                      <Scatter
                        name="Target <= 0"
                        data={(edaData.scatter_data || []).filter(r => r[activeTarget] <= 0).map(r => ({ x: r[selectedFeature], y: r[activeTarget] }))}
                        fill="var(--data-red)"
                        opacity={0.75}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Histograms Tab */}
          {activeEdaTab === "hist" && (
            <div className="content-block" style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700 }}>1%-TRIMMED HISTOGRAM: {selectedFeature}</div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button onClick={() => setHistMode("GROUPED")} className={`mono-btn ${histMode === "GROUPED" ? "mono-btn-active" : ""}`}>GROUPED</button>
                  <button onClick={() => setHistMode("STACKED")} className={`mono-btn ${histMode === "STACKED" ? "mono-btn-active" : ""}`}>STACKED</button>
                </div>
              </div>
              <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={edaData.hist_data?.[activeTarget]?.[selectedFeature] || []}>
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis dataKey="range" tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" angle={-25} textAnchor="end" height={45} />
                    <YAxis tick={{ fill: '#f0f0f0', fontSize: 10 }} stroke="#444444" />
                    <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="pos" name="Target > 0" fill="var(--data-green)" stackId={histMode === "STACKED" ? "a" : undefined} opacity={0.8} />
                    <Bar dataKey="neg" name="Target <= 0" fill="var(--data-red)" stackId={histMode === "STACKED" ? "a" : undefined} opacity={0.8} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* PCA Embeddings Tab */}
          {activeEdaTab === "pca" && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 16 }}>
              <div className="content-block" style={{ margin: 0, padding: 14, maxHeight: 460, display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 6, marginBottom: 8 }}>
                  PAIRWISE PCA (STRICT ISOLATION)
                </div>
                <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
                  {["ALL", "BASE", "VELOCITY"].map(m => (
                    <button key={m} onClick={() => setPcaFilter(m)} className={`mono-btn ${pcaFilter === m ? "mono-btn-active" : ""}`} style={{ flex: 1, padding: '2px 4px', fontSize: '0.68rem' }}>{m}</button>
                  ))}
                </div>
                <div style={{ overflowY: 'auto', flex: 1 }}>
                  {[...(pcaFilter !== "VELOCITY" ? edaData.r2_groups?.pca_base || [] : []), ...(pcaFilter !== "BASE" ? edaData.r2_groups?.pca_velocity || [] : [])].map((p, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedPcaPair(p)}
                      style={{
                        padding: '6px 8px',
                        fontSize: '0.72rem',
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        background: selectedPcaPair?.label === p.label ? 'var(--text-color)' : 'transparent',
                        color: selectedPcaPair?.label === p.label ? 'var(--bg-color)' : 'inherit'
                      }}
                    >
                      <span>{p.label}</span>
                      <span style={{ fontWeight: 'bold' }}>R²: {p.r2.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="content-block" style={{ margin: 0, padding: 16 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: 10 }}>
                  {selectedPcaPair?.label || "PCA PAIR SCATTER"} (R² = {selectedPcaPair?.r2?.toFixed(6)})
                </div>
                <div style={{ width: '100%', height: 350 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name={selectedPcaPair?.f1} tick={{ fill: '#888888', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} />
                      <YAxis type="number" dataKey="y" name={selectedPcaPair?.f2} tick={{ fill: '#f0f0f0', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                      <Scatter name="Points" data={(edaData.scatter_data || []).map(r => ({ x: r[selectedPcaPair?.f1], y: r[selectedPcaPair?.f2] }))} fill="var(--data-blue)" opacity={0.8} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Manifolds Tab */}
          {activeEdaTab === "manifold" && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 16 }}>
              <div className="content-block" style={{ margin: 0, padding: 14, maxHeight: 460, display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 6, marginBottom: 8 }}>
                  3D MANIFOLD PROJECTIONS
                </div>
                <div style={{ overflowY: 'auto', flex: 1 }}>
                  {edaData.r2_groups?.umap_phate?.map((p, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedManifoldPair(p)}
                      style={{
                        padding: '6px 8px',
                        fontSize: '0.72rem',
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        background: selectedManifoldPair?.label === p.label ? 'var(--text-color)' : 'transparent',
                        color: selectedManifoldPair?.label === p.label ? 'var(--bg-color)' : 'inherit'
                      }}
                    >
                      <span>{p.label}</span>
                      <span style={{ fontWeight: 'bold' }}>R²: {p.r2.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="content-block" style={{ margin: 0, padding: 16 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: 10 }}>
                  {selectedManifoldPair?.label || "MANIFOLD PROJECTION"}
                </div>
                <div style={{ width: '100%', height: 350 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name={selectedManifoldPair?.f1} tick={{ fill: '#888888', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} />
                      <YAxis type="number" dataKey="y" name={selectedManifoldPair?.f2} tick={{ fill: '#f0f0f0', fontSize: 10 }} stroke="#444444" domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                      <Scatter name="Manifold" data={(edaData.scatter_data || []).map(r => ({ x: r[selectedManifoldPair?.f1], y: r[selectedManifoldPair?.f2] }))} fill="var(--data-green)" opacity={0.8} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Correlation Tab */}
          {activeEdaTab === "corr" && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 16 }}>
              <div className="content-block" style={{ margin: 0, padding: 14, maxHeight: 460, display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 6, marginBottom: 8, color: 'var(--data-red)' }}>
                  REDUNDANT ALERT (>0.85)
                </div>
                <div style={{ overflowY: 'auto', flex: 1 }}>
                  {(edaData.redundant_pairs || []).map((rp, idx) => (
                    <div key={idx} onClick={() => setCorrFeature(rp.feature1)} style={{ padding: '6px 4px', fontSize: '0.72rem', borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                        <span>{rp.feature1}</span>
                        <span style={{ color: 'var(--data-red)' }}>{rp.correlation.toFixed(3)}</span>
                      </div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>↔ {rp.feature2}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="content-block" style={{ margin: 0, padding: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>CORRELATION PROFILE FOR: {corrFeature}</span>
                  <select value={corrFeature || ""} onChange={(e) => setCorrFeature(e.target.value)} style={{ background: 'var(--bg-color)', color: 'var(--text-color)', border: '1px solid var(--border-color)', padding: '2px 6px', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                    {edaData.features?.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
                <div style={{ width: '100%', height: 350 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={Object.entries(edaData.correlation_matrix?.[corrFeature] || {})
                        .filter(([k]) => k !== corrFeature)
                        .map(([k, v]) => ({ feature: k, correlation: v, abs: Math.abs(v) }))
                        .sort((a, b) => b.abs - a.abs)
                        .slice(0, 16)}
                      margin={{ top: 10, right: 30, left: 130, bottom: 10 }}
                    >
                      <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                      <XAxis type="number" domain={[-1, 1]} tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" />
                      <YAxis type="category" dataKey="feature" tick={{ fill: '#f0f0f0', fontSize: 9 }} stroke="#444444" width={120} />
                      <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                      <ReferenceLine x={0} stroke="#ffffff" opacity={0.4} />
                      <Bar dataKey="correlation">
                        {Object.entries(edaData.correlation_matrix?.[corrFeature] || {})
                          .filter(([k]) => k !== corrFeature)
                          .map(([k, v]) => ({ feature: k, correlation: v, abs: Math.abs(v) }))
                          .sort((a, b) => b.abs - a.abs)
                          .slice(0, 16)
                          .map((entry, idx) => (
                            <Cell key={idx} fill={Math.abs(entry.correlation) > 0.85 ? 'var(--data-red)' : entry.correlation >= 0 ? 'var(--data-green)' : 'var(--data-blue)'} />
                          ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* PnL Bins Tab */}
          {activeEdaTab === "pnl_bins" && (
            <div className="content-block" style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>100% STACKED PNL COHORT MIGRATION OVER TIME</span>
                <div style={{ display: 'flex', gap: 10, fontSize: '0.72rem' }}>
                  <span style={{ color: 'var(--data-red)' }}>HEAVY LOSS</span>
                  <span style={{ color: '#fb923c' }}>MOD LOSS</span>
                  <span style={{ color: '#94a3b8' }}>EVEN</span>
                  <span style={{ color: 'var(--data-blue)' }}>MOD PROFIT</span>
                  <span style={{ color: 'var(--data-green)' }}>HEAVY PROFIT</span>
                  <span style={{ color: '#ffffff', fontWeight: 'bold' }}>PRICE ($)</span>
                </div>
              </div>
              <div style={{ width: '100%', height: 350 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={edaData.bin_timeline || []}>
                    <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fill: '#888888', fontSize: 9 }} tickFormatter={(t) => t.slice(5, 10)} stroke="#444444" />
                    <YAxis yAxisId="bAx" tick={{ fill: '#888888', fontSize: 9 }} stroke="#444444" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <YAxis yAxisId="pAx" orientation="right" tick={{ fill: '#ffffff', fontSize: 9 }} stroke="#444444" domain={['auto', 'auto']} tickFormatter={(p) => `$${p.toFixed(6)}`} />
                    <Tooltip contentStyle={{ backgroundColor: '#1b1b1b', borderColor: '#555555', fontSize: '10px' }} />
                    <Area yAxisId="bAx" type="monotone" dataKey="bin_heavy_loss" stackId="1" fill="var(--data-red)" stroke="var(--data-red)" opacity={0.7} />
                    <Area yAxisId="bAx" type="monotone" dataKey="bin_moderate_loss" stackId="1" fill="#fb923c" stroke="#fb923c" opacity={0.7} />
                    <Area yAxisId="bAx" type="monotone" dataKey="bin_breakeven" stackId="1" fill="#94a3b8" stroke="#94a3b8" opacity={0.7} />
                    <Area yAxisId="bAx" type="monotone" dataKey="bin_moderate_profit" stackId="1" fill="var(--data-blue)" stroke="var(--data-blue)" opacity={0.7} />
                    <Area yAxisId="bAx" type="monotone" dataKey="bin_heavy_profit" stackId="1" fill="var(--data-green)" stroke="var(--data-green)" opacity={0.7} />
                    <Line yAxisId="pAx" type="monotone" dataKey="price" stroke="#ffffff" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('react-analysis-root'));
root.render(<QuantAnalysisRoot />);
