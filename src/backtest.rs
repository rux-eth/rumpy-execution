//! Backtest runner: ties together all execution modules into a single entry point.
//!
//! Consumes a single unified H1 parquet (built by `features build-execution`) containing:
//!   alpha_future, close, volume, market_cap, spread_bps, kappa, funding_rate
//! per (timestamp, symbol) at H1 resolution.
//!
//! D1 covariance is computed from D1 OHLCV (separate file) and reused 24x per day.
//!
//! This is the function that:
//!   - The CLI calls for `rumpy execution backtest`
//!   - The PyO3 bindings expose for Optuna tuning
//!   - Maestro calls in cross-validation mode

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::alpha::AlphaRow;
use crate::aim;
use crate::config::ExecutionConfig;
use crate::covariance;
use crate::diagnostics;
use crate::metrics::{self, TrialMetrics};
use crate::pipeline::{self, WeightOutput};
use crate::config::WalkForwardConfig;
use crate::walkforward::{self, FoldEval};

// ---------------------------------------------------------------------------
// Data paths
// ---------------------------------------------------------------------------

/// Paths to input data for a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestPaths {
    /// Unified H1 execution parquet (from `features build-execution`).
    /// Contains everything: timestamp, symbol, OHLCV, alpha_future, spread_bps, kappa, funding_rate.
    /// Covariance is derived from D1 closes extracted from this same file.
    pub execution_h1: PathBuf,
}

// ---------------------------------------------------------------------------
// Execution row (from unified parquet)
// ---------------------------------------------------------------------------

/// One row from the unified execution H1 parquet.
#[derive(Debug, Clone)]
pub(crate) struct ExecRow {
    timestamp: i64,
    symbol: String,
    close: f64,
    volume: f64,
    // Parsed to validate the input schema; not consumed by the pipeline.
    #[allow(dead_code)]
    market_cap: f64,
    alpha_future: f64,
    spread_bps: f64,
    kappa: f64,
    funding_rate: f64,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Full backtest result.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: TrialMetrics,
    pub fold_eval: FoldEval,
    pub weights: Vec<WeightOutput>,
    pub per_bar_returns: Vec<(i64, f64)>,
    /// A4 Layer 2: per-bar daily returns computed off `nav_liquid` (mark
    /// NAV minus liquidation haircut) instead of mark NAV. Use this series
    /// for net-of-haircut performance evaluation per Frazzini-Israel-
    /// Moskowitz 2018 / Novy-Marx-Velikov 2016 / Bangia et al. 1999.
    /// When `nav_liquid_{t-1} <= ε`, falls back to mark return (early-run
    /// thin-NAV bars) to avoid singularities.
    pub per_bar_returns_liquid: Vec<(i64, f64)>,
    pub solve_stats: pipeline::SolveStats,
    pub benchmark_sharpe: f64,
    /// Minimum margin ratio across all bars (distance to liquidation).
    pub min_margin_ratio: f64,
    /// Per-day cost decomposition aligned with `per_bar_returns`.
    pub cost_breakdown: Vec<DailyCosts>,
    /// Path-level CDaR(α=0.12) on the full concatenated daily return series.
    /// Diagnostic only — the tuner computes its own CDaR on the in-sample slice
    /// in Python (see train/scripts/tune_execution.py). Kept here so CLI runs
    /// can print a quick at-a-glance number.
    pub path_cdar: f64,
    /// Path-level MaxDD on the full concatenated additive return path.
    /// Diagnostic only.
    pub path_max_dd: f64,
    /// One-sided NAV cap mode: cumulative dollars skimmed across the run.
    /// Diagnostic; 0.0 when `nav_cap_usd = None`.
    pub cumulative_skimmed_usd: f64,
    /// One-sided NAV cap mode: per-bar skim events `(timestamp, amount)`.
    /// Only populated for bars with `skim > 0`. Empty when no cap.
    pub per_bar_skimmed: Vec<(i64, f64)>,
    /// A4 Layer 2 audit fields — per-bar haircut decomposition. All
    /// length = n_records (same alignment as `per_bar_returns`). Used by
    /// the liquidity-stack audit tooling to inspect formula correctness.
    pub per_bar_haircut: Vec<(i64, f64)>,
    pub per_bar_haircut_current: Vec<f64>,
    pub per_bar_haircut_last_known: Vec<f64>,
    pub per_bar_haircut_dlom: Vec<f64>,
    pub per_bar_n_dlom: Vec<u32>,
    pub per_bar_nav_liquid: Vec<f64>,
    /// A4 tradeability gate diagnostics — total trades refused across the run.
    pub n_rejected_trades: u64,
    /// A4 tradeability gate diagnostics — total |notional| refused.
    pub rejected_notional: f64,
}

/// Per-day cost decomposition. All values in dollars.
/// `commission`, `spread`, `impact` are positive numbers (costs deducted from
/// NAV during trading). `funding` is signed (positive = longs received).
#[derive(Debug, Clone)]
pub struct DailyCosts {
    pub commission: f64,
    pub spread: f64,
    pub impact: f64,
    pub funding: f64,
    pub turnover: f64,
    pub nav: f64,
}

// ---------------------------------------------------------------------------
// Preloaded data (shared across Optuna trials)
// ---------------------------------------------------------------------------

/// Interned symbol table: maps symbol strings to compact u32 IDs.
/// Eliminates ~104M duplicate String heap allocations across 7 lookups.
pub struct SymbolTable {
    pub(crate) to_id: HashMap<String, u32>,
    pub(crate) to_name: Vec<String>,
}

impl SymbolTable {
    fn new() -> Self {
        Self { to_id: HashMap::new(), to_name: Vec::new() }
    }

    fn intern(&mut self, sym: &str) -> u32 {
        if let Some(&id) = self.to_id.get(sym) {
            return id;
        }
        let id = self.to_name.len() as u32;
        self.to_name.push(sym.to_string());
        self.to_id.insert(sym.to_string(), id);
        id
    }
}

/// Read-only precomputed data shared across Optuna trials.
///
/// Alpha is daily by design — one row per (day, symbol). The H1 parquet is
/// loaded for execution-side lookups (price/spread/kappa/funding) needed in
/// Twap fill mode; sigma/ADV are built at D1 (halflife in days) and indexed
/// by day-floor.
///
/// Safe to share across threads: all inner types are Send + Sync.
/// Lookups use interned u32 symbol IDs instead of String keys (~3x less memory).
pub struct PreloadedData {
    /// Symbol intern table (~900 unique symbols → u32 IDs).
    pub(crate) symbols: SymbolTable,
    /// Daily alphas (one per (day, symbol)). Used by QP, gamma, aim, benchmark.
    pub(crate) alphas: Vec<AlphaRow>,
    /// H1-keyed cost lookups (needed for Twap intraday execution).
    pub(crate) spread_lookup: HashMap<(i64, u32), f64>,
    pub(crate) kappa_lookup: HashMap<(i64, u32), f64>,
    pub(crate) price_lookup: HashMap<(i64, u32), f64>,
    /// Funding-rate lookups, keyed at the cadence of the simulation's bar
    /// loop. The exchange settles hourly; what changes between fill modes is
    /// the bar interval (1h vs 1d), so the lookup value must match:
    ///   - `funding_lookup_twap`: per-hour rates (raw HL hourly), keyed by H1 ts.
    ///   - `funding_lookup_market`: per-day rates = SUM of 24 hourly rates,
    ///     keyed by day-floor. With constant intraday position in Market mode,
    ///     `Σ_h rate_h × pos = (Σ_h rate_h) × pos`, so applying the day-summed
    ///     rate once per daily bar is mathematically identical to 24 hourly
    ///     accruals against the same constant position.
    pub(crate) funding_lookup_twap: HashMap<(i64, u32), f64>,
    pub(crate) funding_lookup_market: HashMap<(i64, u32), f64>,
    /// Forward returns at D1 (for `score_weighted_benchmark`), keyed by day-floor.
    pub(crate) forward_returns: HashMap<(i64, u32), f64>,
    /// Sigma and ADV at D1, EWMA halflives in DAYS, keyed by day-floor.
    pub(crate) sigma_lookup: HashMap<(i64, u32), f64>,
    pub(crate) adv_lookup: HashMap<(i64, u32), f64>,
    pub(crate) rolling_cov: covariance::RollingCovariance,
    /// Pre-computed D1 aim weights at the preload config's l_target/ridge.
    /// Per-trial recompute happens in `run_trial` if those fields differ.
    pub(crate) aim_weights: Vec<aim::AimWeight>,
    /// Execution-bar timeline for `Market` mode (one ts per day, day-floor).
    pub(crate) bar_timestamps_market: Vec<i64>,
    /// Execution-bar timeline for `Twap` mode (all H1 timestamps with prices).
    pub(crate) bar_timestamps_twap: Vec<i64>,

    // ---- Per-trial rebuild support (added 2026-04-27) ----
    //
    // Pre-2026-04-27 the engine ignored trial-config divergence on
    // covariance/sigma/adv/alpha-l-target/alpha-units. The raw inputs and
    // the preload provenance are now retained so `run_trial` can detect
    // a divergent trial config and rebuild only the affected pieces
    // (~5s overhead per divergent trial).
    /// D1 close prices, day-major flat (n_days × n_syms).
    pub(crate) d1_close: Vec<f64>,
    /// D1 dollar volume, day-major flat.
    pub(crate) d1_dollar_vol: Vec<f64>,
    /// D1 raw (non-winsorized) log returns, day-major flat. Allows
    /// re-winsorize when `covariance.winsorize_sigma` differs per trial.
    pub(crate) d1_returns_raw: Vec<f64>,
    pub(crate) d1_ts: Vec<i64>,
    pub(crate) d1_syms: Vec<String>,
    /// Original alphas before unit conversion. Allows re-conversion when
    /// `alpha_units` differs (or when re-derived sigma changes the
    /// dollar-alpha values).
    pub(crate) alphas_raw: Vec<AlphaRow>,
    /// Snapshot of the config at preload time. Compared against the trial
    /// config in `run_trial` to detect which pieces need rebuilding.
    pub(crate) preload_provenance: crate::preproc::PreprocProvenance,
}

// Verify Send + Sync at compile time (needed for #[pyclass(frozen)])
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() { assert_send_sync::<PreloadedData>(); }
};

impl PreloadedData {
    /// Read current RSS in MB (for profiling).
    pub fn rss_mb_static() -> f64 {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(o) = Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output() {
                if let Ok(s) = std::str::from_utf8(&o.stdout) {
                    if let Ok(kb) = s.trim().parse::<u64>() { return kb as f64 / 1024.0; }
                }
            }
        }
        #[cfg(target_os = "linux")]
        {
            if let Ok(s) = std::fs::read_to_string(format!("/proc/{}/status", std::process::id())) {
                for line in s.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb) = line.split_whitespace().nth(1) {
                            if let Ok(v) = kb.parse::<u64>() { return v as f64 / 1024.0; }
                        }
                    }
                }
            }
        }
        0.0
    }

    /// Load parquet and precompute config-independent data at H1 resolution.
    ///
    /// Always loads full H1 data. D1 filtering happens per-trial in `run_trial`.
    /// Uses only fixed config fields: covariance.*.
    pub fn load(
        paths: &BacktestPaths,
        config: &ExecutionConfig,
    ) -> Result<Self> {
        let load_start = std::time::Instant::now();

        fn rss_mb() -> f64 {
            #[cfg(target_os = "macos")]
            {
                use std::process::Command;
                if let Ok(o) = Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output() {
                    if let Ok(s) = std::str::from_utf8(&o.stdout) {
                        if let Ok(kb) = s.trim().parse::<u64>() { return kb as f64 / 1024.0; }
                    }
                }
            }
            #[cfg(target_os = "linux")]
            {
                if let Ok(s) = std::fs::read_to_string(format!("/proc/{}/status", std::process::id())) {
                    for line in s.lines() {
                        if line.starts_with("VmRSS:") {
                            if let Some(kb) = line.split_whitespace().nth(1) {
                                if let Ok(v) = kb.parse::<u64>() { return v as f64 / 1024.0; }
                            }
                        }
                    }
                }
            }
            0.0
        }

        let baseline_mb = rss_mb();

        // 1. Load unified H1 execution parquet
        let t0 = std::time::Instant::now();
        let (rows, parquet_units) = load_execution_parquet(
            &paths.execution_h1,
            config.cost_model.spread_bps_calibration_multiplier,
            config.cost_model.kappa_calibration_multiplier,
        )?;
        if config.cost_model.spread_bps_calibration_multiplier != 1.0 || config.cost_model.kappa_calibration_multiplier != 1.0 {
            println!("  cost calibration: spread×{:.4} kappa×{:.4} (spec-pinned, applied at load)",
                config.cost_model.spread_bps_calibration_multiplier,
                config.cost_model.kappa_calibration_multiplier);
        }

        // Layer 2 (units contract): cross-check parquet metadata against
        // config.alpha_units. The parquet writer (build_execution) stamps
        // `prediction_units` if known. If absent → legacy parquet, trust
        // the config (with a warning). If present and disagrees → bail.
        match (&parquet_units, config.alpha_units) {
            (Some(p), c) => {
                let cfg_str = match c {
                    crate::config::AlphaUnits::DollarAlpha => "dollar_alpha",
                    crate::config::AlphaUnits::VolNormalizedExcess => "vol_normalized_excess",
                };
                if p != cfg_str {
                    anyhow::bail!(
                        "alpha_units mismatch: parquet declares prediction_units={p:?} but \
                         config.alpha_units={cfg_str:?}. Either rebuild the parquet with the \
                         correct target or set config.alpha_units to match. Refusing to load \
                         to prevent silent misweighting in the QP."
                    );
                }
                println!("  alpha_units contract: parquet={p:?} ✓ matches config");
            }
            (None, _) => {
                println!(
                    "  ⚠️  alpha_units contract: parquet has no prediction_units metadata \
                     (legacy build); trusting config.alpha_units = {:?}", config.alpha_units
                );
            }
        }
        let n_syms = {
            let mut s: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for r in &rows { s.insert(&r.symbol); }
            s.len()
        };
        println!("  execution H1: {} rows, {} symbols", rows.len(), n_syms);
        let load_elapsed = t0.elapsed().as_secs_f64();
        println!("  [mem] after parquet load: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);

        // 2. Intern symbols
        let mut symbols = SymbolTable::new();
        for r in &rows {
            symbols.intern(&r.symbol);
        }
        println!("  {} unique symbols interned", symbols.to_name.len());

        // 3. Build daily alphas: one row per (day, symbol). Alpha exists only
        //    at daily resolution; if the parquet has multiple H1 rows per day
        //    with the same daily alpha, take the first one.
        let alphas: Vec<AlphaRow> = {
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for r in &rows {
                if !r.alpha_future.is_finite() { continue; }
                let day = (r.timestamp / 86400) * 86400;
                if seen.insert((day, r.symbol.clone())) {
                    out.push(AlphaRow {
                        timestamp: day,
                        symbol: r.symbol.clone(),
                        alpha_future: r.alpha_future,
                    });
                }
            }
            out
        };
        println!("  {} D1 alpha rows", alphas.len());
        println!("  [mem] after alphas: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);

        // 4. Build H1-keyed lookups (needed for Twap intraday execution).
        let n_rows = rows.len();
        let mut spread_lookup: HashMap<(i64, u32), f64> = HashMap::with_capacity(n_rows);
        for r in &rows { spread_lookup.insert((r.timestamp, symbols.to_id[&r.symbol]), r.spread_bps); }
        println!("  [mem] after spread: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);
        let mut kappa_lookup: HashMap<(i64, u32), f64> = HashMap::with_capacity(n_rows);
        for r in &rows { kappa_lookup.insert((r.timestamp, symbols.to_id[&r.symbol]), r.kappa); }
        println!("  [mem] after kappa: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);
        let mut price_lookup: HashMap<(i64, u32), f64> = HashMap::with_capacity(n_rows);
        for r in &rows { price_lookup.insert((r.timestamp, symbols.to_id[&r.symbol]), r.close); }
        println!("  [mem] after price: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);
        // H1-keyed funding (Twap mode): direct copy of the parquet's hourly rates.
        let mut funding_lookup_twap: HashMap<(i64, u32), f64> = HashMap::with_capacity(n_rows);
        for r in &rows { funding_lookup_twap.insert((r.timestamp, symbols.to_id[&r.symbol]), r.funding_rate); }
        println!("  [mem] after funding (Twap): {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);
        // D1-keyed funding (Market mode): sum of all hourly rates per (day, sym).
        // Linear sum is correct because rates × 24 ≪ 1% (cross-terms negligible).
        // Missing hourly entries are written as 0.0 by the parquet builder, so
        // gappy days under-count. HL fundingHistory is reliable in practice;
        // if data quality degrades we'd want a coverage warning here.
        let mut funding_lookup_market: HashMap<(i64, u32), f64> = HashMap::new();
        for r in &rows {
            let day = (r.timestamp / 86400) * 86400;
            let sid = symbols.to_id[&r.symbol];
            *funding_lookup_market.entry((day, sid)).or_default() += r.funding_rate;
        }
        println!("  [mem] after funding (Market): {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);

        // 5. D1 covariance + D1 sigma/ADV/forward-returns (all keyed by day-floor).
        let t0 = std::time::Instant::now();
        let (d1_close, d1_dollar_vol, d1_ts, d1_syms) = extract_d1_data(&rows, config.cost_model.volume_is_usd);
        println!("  D1 grid: {} days × {} assets", d1_ts.len(), d1_syms.len());

        let d1_returns_raw = covariance::compute_log_returns(&d1_close, d1_ts.len(), d1_syms.len());
        // Winsorized copy is what the cov computation consumes. Raw is kept on
        // PreloadedData for per-trial re-winsorize when winsorize_sigma differs.
        let mut d1_returns = d1_returns_raw.clone();
        covariance::winsorize_per_asset(&mut d1_returns, d1_ts.len(), d1_syms.len(), config.covariance.winsorize_sigma);

        // Preprocessing sidecar: cov + aim + sigma + adv + forward_returns
        // are computed once per (parquet, config) tuple and persisted next to
        // the parquet as `<parquet>.preproc.bin`. On every subsequent load,
        // we read the sidecar instead of recomputing — eliminates cross-
        // platform FP non-determinism (faer SIMD, libm `ln()`) by definition,
        // since both machines load bit-identical bytes.
        // Set RUMPY_PREPROC_REBUILD=1 to force recomputation.
        // See docs/plans/cross-platform-divergence.md.
        let parquet_n_rows = rows.len() as u64;
        let parquet_first_ts = d1_ts.first().copied().unwrap_or(0);
        let parquet_last_ts = d1_ts.last().copied().unwrap_or(0);
        let parquet_n_symbols = d1_syms.len() as u64;
        // Content hash detects same-shape but different-value parquets (e.g.
        // alpha rebuild without changing row count or ts range). Hash D1
        // close + dollar_vol + alpha — all three independently change when
        // underlying data changes. Including alpha is critical: aim_weights
        // are derived from alpha and cached in the sidecar, so an alpha
        // rebuild without an OHLCV change must invalidate the sidecar.
        let mut content_hash = crate::preproc::fnv_f64s(&d1_close);
        content_hash = content_hash
            .wrapping_mul(0x100000001b3)
            .wrapping_add(crate::preproc::fnv_f64s(&d1_dollar_vol));
        let alpha_values: Vec<f64> = alphas.iter().map(|a| a.alpha_future).collect();
        content_hash = content_hash
            .wrapping_mul(0x100000001b3)
            .wrapping_add(crate::preproc::fnv_f64s(&alpha_values));
        let provenance = crate::preproc::PreprocProvenance::from_config(
            config,
            parquet_n_rows,
            parquet_first_ts,
            parquet_last_ts,
            parquet_n_symbols,
            content_hash,
        );
        let sidecar = crate::preproc::Preproc::sidecar_path(&paths.execution_h1);
        let force_rebuild = std::env::var("RUMPY_PREPROC_REBUILD").is_ok();

        let preproc = if !force_rebuild && sidecar.exists() {
            match crate::preproc::Preproc::load(&sidecar, &provenance) {
                Ok(p) => {
                    println!("  loaded preproc sidecar: {}", sidecar.display());
                    Some(p)
                }
                Err(e) => {
                    // Hard fail on arch mismatch — don't silently rebuild
                    // a cross-arch sidecar locally. User must explicitly
                    // choose: rsync canonical, override, or force rebuild.
                    let msg = e.to_string();
                    if msg.contains("sidecar arch mismatch") {
                        anyhow::bail!(
                            "{}\n\nRefusing to silently rebuild the sidecar on this machine \
                             because the existing one was built on a different arch. To resolve:\n\
                             - rsync the canonical sidecar from the build machine, OR\n\
                             - set RUMPY_ALLOW_CROSS_ARCH_SIDECAR=1 to load the file as-is, OR\n\
                             - set RUMPY_PREPROC_REBUILD=1 to rebuild locally (results will\n\
                               differ from the canonical machine due to faer/libm).",
                            e
                        );
                    }
                    println!("  preproc sidecar invalid ({e}); recomputing");
                    None
                }
            }
        } else {
            None
        };

        // Keep a copy of pre-conversion alphas so run_trial can re-convert when
        // sigma_halflife or alpha_units differs from preload.
        let alphas_raw = alphas.clone();

        let (rolling_cov, sigma_lookup, adv_lookup, forward_returns, aim_weights, alphas) =
            if let Some(p) = preproc {
                let alphas_converted =
                    convert_alpha_units(alphas, &p.sigma_lookup, &symbols, config.alpha_units);
                (
                    p.rolling_cov,
                    p.sigma_lookup,
                    p.adv_lookup,
                    p.forward_returns,
                    p.aim_weights,
                    alphas_converted,
                )
            } else {
                println!("  computing preproc artifacts (cov, aim, sigma, adv, fwd)...");
                let rolling_cov = covariance::compute_ewma_factor_covariance(
                    &d1_returns, d1_ts.len(), d1_syms.len(), &d1_ts, &d1_syms,
                    config.covariance.n_factors, config.covariance.ewma_halflife,
                    config.covariance.min_history, config.covariance.min_assets,
                );
                let sigma_lookup = build_sigma_lookup_d1(
                    &d1_close, &d1_ts, &d1_syms, &symbols,
                    config.cost_model.sigma_halflife_days,
                );
                let adv_lookup = build_adv_lookup_d1(
                    &d1_dollar_vol, &d1_ts, &d1_syms, &symbols,
                    config.cost_model.adv_halflife_days,
                );
                let forward_returns = build_forward_returns_d1(
                    &d1_close, &d1_ts, &d1_syms, &symbols,
                );
                // Convert alpha units BEFORE computing aim (aim depends on
                // dollar-scaled alpha).
                let alphas_converted =
                    convert_alpha_units(alphas, &sigma_lookup, &symbols, config.alpha_units);
                let aim_weights = aim::compute_aim_portfolio(
                    &alphas_converted, &rolling_cov,
                    config.alpha.l_target, config.covariance.ridge,
                );
                // Persist for next load.
                let p = crate::preproc::Preproc {
                    provenance: provenance.clone(),
                    rolling_cov,
                    aim_weights,
                    sigma_lookup,
                    adv_lookup,
                    forward_returns,
                };
                if let Err(e) = p.save(&sidecar) {
                    eprintln!("warn: preproc sidecar save failed: {e}");
                } else {
                    println!("  wrote preproc sidecar: {}", sidecar.display());
                }
                (
                    p.rolling_cov,
                    p.sigma_lookup,
                    p.adv_lookup,
                    p.forward_returns,
                    p.aim_weights,
                    alphas_converted,
                )
            };
        println!("  {} D1 timestamps with covariance", rolling_cov.len());
        let cov_elapsed = t0.elapsed().as_secs_f64();
        println!("  [mem] after preproc: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);
        println!("  {} D1 forward return pairs", forward_returns.len());
        println!("  {} D1 aim weights", aim_weights.len());
        println!("  [mem] after aim weights: {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);

        // 7. Build execution-bar timelines.
        //    Twap mode: every H1 timestamp present in price data (sorted, unique).
        //    Market mode: one timestamp per day (day-floor at 00:00 UTC).
        let bar_timestamps_twap: Vec<i64> = {
            let mut s: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for r in &rows { s.insert(r.timestamp); }
            s.into_iter().collect()
        };
        let bar_timestamps_market: Vec<i64> = {
            let mut s: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for &ts in &bar_timestamps_twap { s.insert((ts / 86400) * 86400); }
            s.into_iter().collect()
        };
        println!("  bar timelines: {} Market days, {} Twap H1 bars",
            bar_timestamps_market.len(), bar_timestamps_twap.len());

        // Drop rows — all lookups built, no longer needed
        drop(rows);
        println!("  [mem] after drop(rows): {:.0} MB (+{:.0})", rss_mb(), rss_mb() - baseline_mb);

        let total_elapsed = load_start.elapsed().as_secs_f64();
        println!("  preload: {:.1}s total (load {:.1}s, cov {:.1}s)",
            total_elapsed, load_elapsed, cov_elapsed);

        Ok(PreloadedData {
            symbols,
            alphas,
            spread_lookup,
            kappa_lookup,
            forward_returns,
            price_lookup,
            funding_lookup_twap,
            funding_lookup_market,
            sigma_lookup,
            adv_lookup,
            rolling_cov,
            aim_weights,
            bar_timestamps_market,
            bar_timestamps_twap,
            d1_close,
            d1_dollar_vol,
            d1_returns_raw,
            d1_ts,
            d1_syms,
            alphas_raw,
            preload_provenance: provenance,
        })
    }
}

// ---------------------------------------------------------------------------
// Trial runner (uses preloaded data)
// ---------------------------------------------------------------------------

/// Run one trial with preloaded data.
///
/// Strategic decisions (alpha, QP, vol target, dynamic gamma) are always daily.
/// `config.fill_mode` only changes the execution-bar timeline:
///   - `Market`: one fill per day at the day's price.
///   - `Twap`:   24 H1 fills walking toward the daily target.
///
/// Thread-safe: takes `&PreloadedData` (shared read-only reference).
pub fn run_trial(
    data: &PreloadedData,
    config: &ExecutionConfig,
    wf_config: &WalkForwardConfig,
    verbose: bool,
) -> Result<BacktestResult> {
    use crate::config::{AlphaUnits, FillMode};

    let trial_start = std::time::Instant::now();
    // Records are always at daily frequency in both fill modes.
    let bars_per_year = 365.0;

    // -----------------------------------------------------------------------
    // Per-trial preproc rebuild: detect which preproc-affecting fields differ
    // from the engine's preloaded values and rebuild only the affected pieces.
    // Pre-2026-04-27 these were silently ignored at trial time, so any sweep
    // that had cov.*, sigma_halflife, adv_halflife, alpha.l_target, or
    // alpha_units in its search space was treating them as fixed at the base
    // config's value. Cost: ~5s per divergent trial.
    // -----------------------------------------------------------------------
    let prov = &data.preload_provenance;
    let trial_units_str = match config.alpha_units {
        AlphaUnits::DollarAlpha => "dollar_alpha",
        AlphaUnits::VolNormalizedExcess => "vol_normalized_excess",
    };
    let cov_changed = config.covariance.n_factors != prov.cov_n_factors as usize
        || config.covariance.ewma_halflife != prov.cov_ewma_halflife as usize
        || config.covariance.min_history != prov.cov_min_history as usize
        || config.covariance.min_assets != prov.cov_min_assets as usize
        || (config.covariance.winsorize_sigma - prov.cov_winsorize_sigma).abs() > 1e-12
        || (config.covariance.ridge - prov.cov_ridge).abs() > 1e-15;
    let sigma_changed = config.cost_model.sigma_halflife_days != prov.sigma_halflife_days as usize;
    let adv_changed = config.cost_model.adv_halflife_days != prov.adv_halflife_days as usize;
    let units_changed = trial_units_str != prov.alpha_units;
    let l_target_changed = (config.alpha.l_target - prov.alpha_l_target).abs() > 1e-12;
    let needs_aim = cov_changed || sigma_changed || units_changed || l_target_changed;

    // Local rebuilds (Option-typed so we only pay the cost when needed; refs
    // fall back to the engine's preloaded values when the field matches).
    let mut local_cov: Option<covariance::RollingCovariance> = None;
    let mut local_sigma: Option<HashMap<(i64, u32), f64>> = None;
    let mut local_adv: Option<HashMap<(i64, u32), f64>> = None;
    let mut local_alphas: Option<Vec<AlphaRow>> = None;
    let mut local_aim: Option<Vec<aim::AimWeight>> = None;

    if cov_changed {
        let n_days = data.d1_ts.len();
        let n_syms = data.d1_syms.len();
        let mut returns = data.d1_returns_raw.clone();
        covariance::winsorize_per_asset(&mut returns, n_days, n_syms, config.covariance.winsorize_sigma);
        local_cov = Some(covariance::compute_ewma_factor_covariance(
            &returns, n_days, n_syms, &data.d1_ts, &data.d1_syms,
            config.covariance.n_factors, config.covariance.ewma_halflife,
            config.covariance.min_history, config.covariance.min_assets,
        ));
    }
    if sigma_changed {
        local_sigma = Some(build_sigma_lookup_d1(
            &data.d1_close, &data.d1_ts, &data.d1_syms, &data.symbols,
            config.cost_model.sigma_halflife_days,
        ));
    }
    if adv_changed {
        local_adv = Some(build_adv_lookup_d1(
            &data.d1_dollar_vol, &data.d1_ts, &data.d1_syms, &data.symbols,
            config.cost_model.adv_halflife_days,
        ));
    }
    // alpha conversion depends on (possibly rebuilt) sigma + alpha_units
    if sigma_changed || units_changed {
        let sigma_for_units = local_sigma.as_ref().unwrap_or(&data.sigma_lookup);
        local_alphas = Some(convert_alpha_units(
            data.alphas_raw.clone(), sigma_for_units, &data.symbols, config.alpha_units,
        ));
    }
    if needs_aim {
        let cov_for_aim = local_cov.as_ref().unwrap_or(&data.rolling_cov);
        let alphas_for_aim = local_alphas.as_ref().unwrap_or(&data.alphas);
        local_aim = Some(aim::compute_aim_portfolio(
            alphas_for_aim, cov_for_aim,
            config.alpha.l_target, config.covariance.ridge,
        ));
    }

    let rolling_cov = local_cov.as_ref().unwrap_or(&data.rolling_cov);
    let sigma_lookup = local_sigma.as_ref().unwrap_or(&data.sigma_lookup);
    let adv_lookup = local_adv.as_ref().unwrap_or(&data.adv_lookup);
    let alphas: &[AlphaRow] = local_alphas.as_deref().unwrap_or(&data.alphas);
    let aim_weights = local_aim.as_ref().unwrap_or(&data.aim_weights);
    let bar_timestamps: &[i64] = match config.fill_mode {
        FillMode::Market => &data.bar_timestamps_market,
        FillMode::Twap => &data.bar_timestamps_twap,
    };
    // Funding cadence must match bar cadence — both are physically equivalent
    // (24 hourly accruals per day); just pre-aggregated to the bar interval.
    let funding_lookup = match config.fill_mode {
        FillMode::Market => &data.funding_lookup_market,
        FillMode::Twap => &data.funding_lookup_twap,
    };
    if verbose {
        println!("  {} D1 alphas, {} execution bars ({:?})",
            alphas.len(), bar_timestamps.len(), config.fill_mode);
    }

    // L2 (2026-05-13): dynamic_gamma schedule + regime_score lookup removed.
    // γ is static; per_name_cap is policy-constant. SOC constraint in the QP
    // controls vol via qp.sigma_target_daily.

    // Simulation engine.
    if verbose {
        println!("\n  === Simulation Engine (holdings-based) ===");
    }
    let t0 = std::time::Instant::now();
    let mut engine = crate::simulation::SimulationEngine::new(config);
    let sim_result = engine.run(
        alphas, bar_timestamps, aim_weights, rolling_cov,
        &data.spread_lookup, &data.kappa_lookup, &data.price_lookup, funding_lookup,
        sigma_lookup, adv_lookup,
        &data.symbols.to_id,
    );
    if verbose {
        println!("  Simulation: {:.1}s, {} bars", t0.elapsed().as_secs_f64(), sim_result.records.len());
    }

    // 3. Extract returns from simulation
    let sim_returns: Vec<(i64, f64)> = sim_result.records.iter()
        .map(|r| (r.timestamp, r.portfolio_return))
        .collect();
    let sim_return_vals: Vec<f64> = sim_returns.iter().map(|(_, r)| *r).collect();

    // A4 Layer 2: liquid returns. Computed as MARK return minus the
    // bar-wise growth in liquidation haircut, scaled by pre-trade NAV.
    //
    //   r_liq_t = r_mark_t − Δhaircut_t / nav_pre_t
    //
    // This is the Frazzini-Israel-Moskowitz 2018 / Novy-Marx-Velikov 2016
    // net-of-cost convention adapted for ENDOGENOUS liquidity:
    // gross PnL minus the day's increase in un-realizable trapped
    // capital. Numerically stable (denominator is mark NAV, always
    // large in compounding mode). Economically meaningful (each bar's
    // contribution = mark gain minus today's added exit-cost reserve).
    //
    // Multiplicative `nav_liq_t / nav_liq_{t-1}` explodes when nav_liquid
    // passes near zero. Pure additive-from-start loses risk
    // normalization. This formulation has neither pathology.
    //
    // Reference: Frazzini, Israel, Moskowitz 2018 "Trading Costs of
    // Asset Pricing Anomalies" — net Sharpe = (mean return − mean cost) /
    // vol. Bangia et al. 1999 endogenous-liquidity adjustment.
    let mut sim_returns_liquid: Vec<(i64, f64)> = Vec::with_capacity(sim_result.records.len());
    let mut prev_haircut: f64 = 0.0;
    for r in sim_result.records.iter() {
        // nav_pre_t = pre-trade NAV at start of this bar's day. Recovered
        // from r.nav (post) and r.portfolio_return (gross). When portfolio
        // return is small (no positions / warmup), defaults to r.nav.
        let nav_pre = if r.portfolio_return.abs() > 1e-12 {
            r.nav / (1.0 + r.portfolio_return)
        } else {
            r.nav
        };
        let nav_pre_safe = if nav_pre > 1e-6 { nav_pre } else { 1.0 };
        let dh = r.liquidation_haircut - prev_haircut;
        let liq_ret = if r.portfolio_return.is_finite() && dh.is_finite() {
            r.portfolio_return - dh / nav_pre_safe
        } else {
            r.portfolio_return
        };
        sim_returns_liquid.push((r.timestamp, liq_ret));
        prev_haircut = r.liquidation_haircut;
    }

    let weights: Vec<pipeline::WeightOutput> = sim_result.records.iter()
        .flat_map(|r| {
            r.weights.iter().map(move |(sid, w)| pipeline::WeightOutput {
                timestamp: r.timestamp,
                symbol: data.symbols.to_name.get(*sid as usize)
                    .cloned().unwrap_or_default(),
                weight_qp: *w,
                weight_final: *w,
            })
        })
        .collect();

    // 4. Print diagnostics (verbose only — CLI use)
    if verbose {
        print_simulation_summary(&sim_result);
        print_simulation_monthly(&sim_result.records);
    }

    // 5. Walk-forward evaluation
    let all_ts: Vec<i64> = sim_returns.iter().map(|(ts, _)| *ts).collect();
    let folds = walkforward::compute_fold_boundaries(&all_ts, wf_config);
    let fold_eval = walkforward::evaluate_folds(&sim_returns, &folds, bars_per_year, wf_config.min_fold_bars);
    if verbose {
        println!(
            "  Walk-forward: {}/{} folds ok, mean Sharpe={:.2}, std={:.2}",
            fold_eval.n_folds_ok, folds.len(), fold_eval.mean_sharpe, fold_eval.std_sharpe
        );
    }

    // 6. Metrics + objectives
    let trial_metrics = metrics::compute_trial_metrics(&sim_return_vals, &weights, bars_per_year, 1);

    // Path-level CDaR / MaxDD on the full concatenated daily return series.
    // Captures cross-fold drawdown chains that per-fold-mean misses. Same
    // α=0.12 as the per-fold version for direct comparability.
    let path_cdar = metrics::cdar(&sim_return_vals, 0.12);
    // Path-level MaxDD: largest peak-to-trough on the cumulative additive
    // return path (consistent with metrics::cdar's drawdown definition).
    let path_max_dd = {
        let mut cum = 0.0f64;
        let mut peak = 0.0f64;
        let mut max_dd = 0.0f64;
        for &r in &sim_return_vals {
            if r.is_finite() {
                cum += r;
                if cum > peak { peak = cum; }
                let dd = peak - cum;
                if dd > max_dd { max_dd = dd; }
            }
        }
        max_dd
    };

    if verbose {
        println!("  Trial: {:.1}s total", trial_start.elapsed().as_secs_f64());
    }

    let cost_breakdown: Vec<DailyCosts> = sim_result.records.iter().map(|r| DailyCosts {
        commission: r.commission_cost,
        spread: r.spread_cost,
        impact: r.impact_cost,
        funding: r.funding_pnl,
        turnover: r.turnover,
        nav: r.nav,
    }).collect();

    Ok(BacktestResult {
        metrics: trial_metrics,
        fold_eval,
        weights,
        per_bar_returns: sim_returns,
        per_bar_returns_liquid: sim_returns_liquid,
        solve_stats: pipeline::SolveStats::default(),
        benchmark_sharpe: score_weighted_benchmark(alphas, &data.forward_returns, &data.symbols.to_id, bars_per_year),
        min_margin_ratio: sim_result.records.iter()
            .map(|r| r.margin_ratio)
            .fold(f64::INFINITY, f64::min),
        cost_breakdown,
        path_cdar,
        path_max_dd,
        cumulative_skimmed_usd: sim_result.cumulative_skimmed_usd,
        per_bar_skimmed: sim_result.per_bar_skimmed.clone(),
        per_bar_haircut: sim_result.records.iter().map(|r| (r.timestamp, r.liquidation_haircut)).collect(),
        per_bar_haircut_current: sim_result.records.iter().map(|r| r.liquidation_haircut_current).collect(),
        per_bar_haircut_last_known: sim_result.records.iter().map(|r| r.liquidation_haircut_last_known).collect(),
        per_bar_haircut_dlom: sim_result.records.iter().map(|r| r.liquidation_haircut_dlom).collect(),
        per_bar_n_dlom: sim_result.records.iter().map(|r| r.n_positions_dlom).collect(),
        per_bar_nav_liquid: sim_result.records.iter().map(|r| r.nav_liquid).collect(),
        n_rejected_trades: sim_result.n_rejected_trades,
        rejected_notional: sim_result.rejected_notional,
    })
}

// ---------------------------------------------------------------------------
// Main backtest function (CLI entry point — loads + runs in one call)
// ---------------------------------------------------------------------------

/// Run a full backtest from the unified H1 execution parquet.
/// This is the CLI/standalone entry point. For Optuna tuning, use
/// `PreloadedData::load()` + `run_trial()` to share data across workers.
pub fn run_backtest(
    paths: &BacktestPaths,
    config: &ExecutionConfig,
    wf_config: &WalkForwardConfig,
    _bars_per_year: f64,
) -> Result<BacktestResult> {
    let data = PreloadedData::load(paths, config)?;
    run_trial(&data, config, wf_config, true)
}

/// Print a summary of backtest results to stdout.
pub fn print_summary(result: &BacktestResult) {
    let m = &result.metrics;
    println!("\n  === Backtest Results ===");
    println!("  Sharpe:       {:.3}", m.sharpe);
    println!("  Ann Return:   {:.1}%", m.ann_return * 100.0);
    println!("  MDD:          {:.1}%", m.mdd * 100.0);
    println!("  Hit Rate:     {:.1}%", m.hit_rate * 100.0);
    println!("  Avg Turnover: {:.4}", m.avg_turnover);
    println!("  Calmar:       {:.2}", m.calmar);
    println!("  Martin:       {:.2}", m.martin_ratio);
    println!("  Ulcer Index:  {:.4}", m.ulcer_index);
    println!("  CI95:         [{:.2}, {:.2}]", m.sharpe_ci95.0, m.sharpe_ci95.1);
    println!("  DSR:          {:.3}", m.deflated_sharpe);
    println!("  N bars:       {}", m.n_bars);

    println!("\n  === Path-level Risk (diagnostic) ===");
    println!("  Path CDaR(α=0.12):    {:.3}", result.path_cdar);
    println!("  Path MaxDD:           {:.3}", result.path_max_dd);

    println!("\n  === Benchmark ===");
    println!("  Score-weighted Sharpe: {:.3}", result.benchmark_sharpe);

    let s = &result.solve_stats;
    println!("\n  === QP Stats ===");
    println!("  Solved: {}/{}, cov_fail={}, solve_fail={}",
        s.n_solved, s.n_bars, s.n_failed_cov, s.n_failed_solve);
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

use arrow::array::{Array, Float64Array, Int64Array, StringArray, LargeStringArray};

fn get_str(col: &dyn Array, i: usize) -> String {
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return a.value(i).to_string();
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return a.value(i).to_string();
    }
    panic!("column is neither Utf8 nor LargeUtf8");
}

/// Load the unified execution H1 parquet.
fn load_execution_parquet(path: &std::path::Path, spread_mult: f64, kappa_mult: f64) -> Result<(Vec<ExecRow>, Option<String>)> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // Layer 2 (units contract): read `prediction_units` from parquet kv_metadata
    // if present. The writer (crates/features/src/execution.rs::build_execution)
    // is responsible for stamping this; legacy parquets predate the contract
    // and return None, in which case we trust the config.
    let prediction_units: Option<String> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .and_then(|kvs| {
            kvs.iter()
                .find(|kv| kv.key == "prediction_units")
                .and_then(|kv| kv.value.clone())
        });

    let reader = builder.with_batch_size(200_000).build()?;

    let idx = |name: &str| -> Option<usize> {
        schema.fields().iter().position(|f| f.name() == name)
    };
    let ts_i = idx("timestamp").context("missing timestamp")?;
    let sym_i = idx("symbol").context("missing symbol")?;
    let close_i = idx("close").context("missing close")?;
    let vol_i = idx("volume").context("missing volume")?;
    let mcap_i = idx("market_cap").context("missing market_cap")?;
    let alpha_i = idx("alpha_future").context("missing alpha_future")?;
    let spread_i = idx("spread_bps").context("missing spread_bps")?;
    let kappa_i = idx("kappa").context("missing kappa")?;
    let funding_i = idx("funding_rate").context("missing funding_rate")?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ts = batch.column(ts_i).as_any().downcast_ref::<Int64Array>().unwrap();
        let sym = batch.column(sym_i);
        let close = batch.column(close_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let vol = batch.column(vol_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let mcap = batch.column(mcap_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let alpha = batch.column(alpha_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let spread = batch.column(spread_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let kappa = batch.column(kappa_i).as_any().downcast_ref::<Float64Array>().unwrap();
        let funding = batch.column(funding_i).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..batch.num_rows() {
            rows.push(ExecRow {
                timestamp: ts.value(i),
                symbol: get_str(sym.as_ref(), i),
                close: close.value(i),
                volume: vol.value(i),
                market_cap: mcap.value(i),
                alpha_future: alpha.value(i),
                spread_bps: spread.value(i) * spread_mult,
                kappa: kappa.value(i) * kappa_mult,
                funding_rate: funding.value(i),
            });
        }
    }
    Ok((rows, prediction_units))
}

// L2 (2026-05-13): load_regime_score removed with the regime per_name_cap overlay.

/// Extract D1 closes from H1 rows: take the first H1 bar per (day, symbol).
/// Returns (close_matrix row-major, d1_timestamps, symbols) — same format as OhlcvData.close_matrix().
/// Extract D1-aggregated data from H1 rows.
///
/// Per (day, symbol):
///   close      = first H1 close of the day (matches the convention used by
///                the D1 covariance pipeline).
///   dollar_vol = Σ(volume_h1 × close_h1) when `volume_is_usd=false` (LEGACY
///                — treats execution_h1's `volume` column as coin units),
///                OR Σ(volume_h1) when `volume_is_usd=true` (CORRECTED —
///                treats the `volume` column as USD directly, matching the
///                actual CG-sourced data). Choose via the spec's
///                `cost_model.volume_is_usd` flag. MUST match the flag used
///                when training the spec's cost-scores parquet.
///
/// Returns (close_matrix, dollar_vol_matrix, days, syms) where matrices are
/// row-major `n_days × n_syms`.
fn extract_d1_data(rows: &[ExecRow], volume_is_usd: bool) -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<String>) {
    let mut day_close: HashMap<(i64, String), f64> = HashMap::new();
    let mut day_dvol: HashMap<(i64, String), f64> = HashMap::new();
    for r in rows {
        let day = (r.timestamp / 86400) * 86400;
        day_close.entry((day, r.symbol.clone())).or_insert(r.close);
        let dv = if volume_is_usd {
            if r.volume.is_finite() { r.volume } else { 0.0 }
        } else {
            if r.volume.is_finite() && r.close.is_finite() { r.volume * r.close } else { 0.0 }
        };
        *day_dvol.entry((day, r.symbol.clone())).or_insert(0.0) += dv;
    }

    let mut days: Vec<i64> = day_close.keys().map(|(d, _)| *d).collect();
    days.sort();
    days.dedup();
    let mut syms: Vec<String> = day_close.keys().map(|(_, s)| s.clone()).collect();
    syms.sort();
    syms.dedup();

    let n_days = days.len();
    let n_syms = syms.len();
    let day_idx: HashMap<i64, usize> = days.iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let sym_idx: HashMap<&str, usize> = syms.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let mut close_matrix = vec![f64::NAN; n_days * n_syms];
    for ((day, sym), close) in &day_close {
        if let (Some(&di), Some(&si)) = (day_idx.get(day), sym_idx.get(sym.as_str())) {
            close_matrix[di * n_syms + si] = *close;
        }
    }
    let mut dvol_matrix = vec![0.0f64; n_days * n_syms];
    for ((day, sym), dv) in &day_dvol {
        if let (Some(&di), Some(&si)) = (day_idx.get(day), sym_idx.get(sym.as_str())) {
            dvol_matrix[di * n_syms + si] = *dv;
        }
    }

    (close_matrix, dvol_matrix, days, syms)
}

/// Build sigma EWMA at D1, keyed by day-floor timestamp.
///
/// Inserts BEFORE updating with bar i's return so sigma(day t) uses returns
/// from days strictly before t (no lookahead). Mirrors the no-lookahead pattern
/// in the H1 helper and in `build_adv_lookup_d1`.
fn build_sigma_lookup_d1(
    d1_close: &[f64],
    d1_ts: &[i64],
    d1_syms: &[String],
    symbols: &SymbolTable,
    halflife_days: usize,
) -> HashMap<(i64, u32), f64> {
    let alpha = 1.0 - (-2.0_f64.ln() / halflife_days.max(1) as f64).exp();
    let n_days = d1_ts.len();
    let n_syms = d1_syms.len();
    let mut lookup = HashMap::with_capacity(n_days * n_syms);

    for (sj, sym) in d1_syms.iter().enumerate() {
        let sid = match symbols.to_id.get(sym) { Some(&i) => i, None => continue };
        let mut ewma_sq = 0.0f64;
        let mut ewma_mean = 0.0f64;
        let mut n_seen = 0usize;
        let mut prev_close = f64::NAN;
        for i in 0..n_days {
            let close = d1_close[i * n_syms + sj];
            // Insert BEFORE update — sigma at day i uses returns through i-1.
            let sigma = if n_seen >= 2 {
                (ewma_sq - ewma_mean * ewma_mean).max(0.0).sqrt()
            } else { 0.0 };
            lookup.insert((d1_ts[i], sid), sigma);
            // Update ewma with this day's return (for future bars).
            if prev_close.is_finite() && prev_close > 0.0 && close.is_finite() && close > 0.0 {
                let r = (close / prev_close).ln();
                if r.is_finite() {
                    ewma_sq = alpha * r * r + (1.0 - alpha) * ewma_sq;
                    ewma_mean = alpha * r + (1.0 - alpha) * ewma_mean;
                    n_seen += 1;
                }
            }
            if close.is_finite() && close > 0.0 { prev_close = close; }
        }
    }
    lookup
}

/// Build dollar-volume EWMA at D1, keyed by day-floor timestamp.
///
/// Inserts BEFORE updating with day i's volume so ADV(day t) uses volumes
/// strictly before t (no lookahead). Bar 0 seeds with day 0's volume — only
/// matters during warmup (gated by `has_qp_solution` in the simulation).
fn build_adv_lookup_d1(
    d1_dollar_vol: &[f64],
    d1_ts: &[i64],
    d1_syms: &[String],
    symbols: &SymbolTable,
    halflife_days: usize,
) -> HashMap<(i64, u32), f64> {
    let alpha = 1.0 - (-2.0_f64.ln() / halflife_days.max(1) as f64).exp();
    let n_days = d1_ts.len();
    let n_syms = d1_syms.len();
    let mut lookup = HashMap::with_capacity(n_days * n_syms);

    for (sj, sym) in d1_syms.iter().enumerate() {
        let sid = match symbols.to_id.get(sym) { Some(&i) => i, None => continue };
        let mut ewma = 0.0f64;
        let mut initialized = false;
        for i in 0..n_days {
            let dv = d1_dollar_vol[i * n_syms + sj];
            if i == 0 {
                // Seed bar 0 with bar 0's volume (minimal warmup leak; QP gate masks it).
                ewma = if dv.is_finite() { dv } else { 0.0 };
                lookup.insert((d1_ts[i], sid), ewma);
                initialized = ewma > 0.0;
                continue;
            }
            // Insert BEFORE update — ADV at day i uses volumes from days 0..i-1.
            lookup.insert((d1_ts[i], sid), ewma);
            if dv.is_finite() && dv > 0.0 {
                if !initialized {
                    ewma = dv;
                    initialized = true;
                } else {
                    ewma = alpha * dv + (1.0 - alpha) * ewma;
                }
            }
        }
    }
    lookup
}

/// Build forward-return lookup at D1, keyed by day-floor timestamp.
/// `fwd(day t, sym) = close(t+1) / close(t) - 1`. Last day has no entry.
fn build_forward_returns_d1(
    d1_close: &[f64],
    d1_ts: &[i64],
    d1_syms: &[String],
    symbols: &SymbolTable,
) -> HashMap<(i64, u32), f64> {
    let n_days = d1_ts.len();
    let n_syms = d1_syms.len();
    let mut fwd = HashMap::with_capacity(n_days * n_syms);
    for (sj, sym) in d1_syms.iter().enumerate() {
        let sid = match symbols.to_id.get(sym) { Some(&i) => i, None => continue };
        for i in 0..n_days.saturating_sub(1) {
            let curr = d1_close[i * n_syms + sj];
            let next = d1_close[(i + 1) * n_syms + sj];
            if curr.is_finite() && curr > 0.0 && next.is_finite() {
                fwd.insert((d1_ts[i], sid), next / curr - 1.0);
            }
        }
    }
    fwd
}

/// Print monthly diagnostics from simulation records.
fn print_simulation_monthly(records: &[crate::simulation::SimulationRecord]) {
    use std::collections::BTreeMap;

    if records.is_empty() { return; }

    // Group records by month
    let mut by_month: BTreeMap<String, Vec<&crate::simulation::SimulationRecord>> = BTreeMap::new();
    for r in records {
        let (y, m, _) = diagnostics::epoch_to_ymd(r.timestamp);
        let key = format!("{y:04}-{m:02}");
        by_month.entry(key).or_default().push(r);
    }

    println!("\n  === Monthly Diagnostics (Simulation) ===");
    println!("    {:>7} | {:>10} | {:>8} | {:>5} | {:>4}L {:>4}S | {:>7} | {:>7} | {:>7} | {:>6} | {:>6} | {:>6} | {:>5} | {:>5}",
        "Month", "NAV", "Add P&L", "Gross", "", "", "Long%", "Short%", "Fund$", "Comm$", "Sprd$", "Ret%", "MDD%", "MarR");
    println!("    {}", "-".repeat(140));

    for (month, recs) in &by_month {
        let n = recs.len() as f64;

        // Avg gross leverage
        let avg_gross_lev: f64 = recs.iter()
            .map(|r| if r.nav > 1e-6 { r.gross_exposure / r.nav } else { 0.0 })
            .sum::<f64>() / n;

        // Avg L/S counts
        let avg_long = recs.iter().map(|r| r.n_long as f64).sum::<f64>() / n;
        let avg_short = recs.iter().map(|r| r.n_short as f64).sum::<f64>() / n;

        // PnL attribution
        let long_pnl: f64 = recs.iter().map(|r| r.long_pnl).sum();
        let short_pnl: f64 = recs.iter().map(|r| r.short_pnl).sum();
        let funding_pnl: f64 = recs.iter().map(|r| r.funding_pnl).sum();
        let commission: f64 = recs.iter().map(|r| r.commission_cost).sum();
        let spread: f64 = recs.iter().map(|r| r.spread_cost).sum();

        // Normalize PnL by start-of-month NAV
        let month_start_nav = recs[0].nav / (1.0 + recs[0].portfolio_return);
        let long_pct = if month_start_nav > 1e-6 { long_pnl / month_start_nav * 100.0 } else { 0.0 };
        let short_pct = if month_start_nav > 1e-6 { short_pnl / month_start_nav * 100.0 } else { 0.0 };

        // Monthly return (compounded)
        let month_return: f64 = recs.iter()
            .map(|r| 1.0 + r.portfolio_return)
            .product::<f64>() - 1.0;

        // Monthly MDD
        let mut peak = month_start_nav;
        let mut worst_dd = 0.0f64;
        for r in recs {
            if r.nav > peak { peak = r.nav; }
            let dd = (r.nav - peak) / peak;
            if dd < worst_dd { worst_dd = dd; }
        }

        // Min margin ratio this month
        let min_margin: f64 = recs.iter()
            .map(|r| r.margin_ratio)
            .fold(f64::INFINITY, f64::min);
        let margin_str = if min_margin.is_infinite() { "  inf".to_string() } else { format!("{:.1}x", min_margin) };

        let end_nav = recs.last().map(|r| r.nav).unwrap_or(0.0);
        let add_pnl = recs.last().map(|r| r.additive_return).unwrap_or(0.0);
        let nav_start_capital = recs[0].nav / (1.0 + recs[0].cumulative_return);
        let add_pnl_dollars = add_pnl * nav_start_capital;
        println!("    {:>7} | {:>10.0} | {:>+7.0}$ | {:>5.2} | {:>4.0}L {:>4.0}S | {:>+6.1}% | {:>+6.1}% | {:>+6.0}$ | {:>5.0}$ | {:>5.0}$ | {:>+5.1}% | {:>5.1}% | {:>5}",
            month, end_nav, add_pnl_dollars, avg_gross_lev, avg_long, avg_short,
            long_pct, short_pct, funding_pnl,
            commission, spread,
            month_return * 100.0, worst_dd * 100.0, margin_str);
    }
}

/// Print simulation engine summary.
fn print_simulation_summary(result: &crate::simulation::SimulationResult) {
    let book = &result.final_book;
    let records = &result.records;
    if records.is_empty() {
        println!("  No simulation records.");
        return;
    }

    let start_nav = records[0].nav / (1.0 + records[0].portfolio_return);
    // Use last record's nav (in constant-NAV mode this is the effective compounded NAV)
    let final_nav = records.last().unwrap().nav;
    let total_return = (final_nav / start_nav - 1.0) * 100.0;

    let total_commission: f64 = records.iter().map(|r| r.commission_cost).sum();
    let total_spread: f64 = records.iter().map(|r| r.spread_cost).sum();
    let total_impact: f64 = records.iter().map(|r| r.impact_cost).sum();
    let total_funding: f64 = records.iter().map(|r| r.funding_pnl).sum();
    let total_costs = total_commission + total_spread + total_impact;

    let avg_gross_leverage: f64 = records.iter()
        .map(|r| if r.nav > 1e-6 { r.gross_exposure / r.nav } else { 0.0 })
        .sum::<f64>() / records.len() as f64;
    let max_gross: f64 = records.iter().map(|r| r.gross_exposure).fold(0.0f64, f64::max);
    let avg_turnover: f64 = records.iter().map(|r| r.turnover).sum::<f64>() / records.len() as f64;
    let avg_positions: f64 = records.iter().map(|r| (r.n_long + r.n_short) as f64).sum::<f64>() / records.len() as f64;
    let min_margin: f64 = records.iter().map(|r| r.margin_ratio).fold(f64::INFINITY, f64::min);

    // Drawdown (mark NAV)
    let mut peak = start_nav;
    let mut worst_dd = 0.0f64;
    for r in records {
        if r.nav > peak { peak = r.nav; }
        let dd = (r.nav - peak) / peak;
        if dd < worst_dd { worst_dd = dd; }
    }

    // Layer 2 (A4): liquidation-aware NAV + drawdown.
    // The liquid NAV deducts the predicted cost of fully exiting all open
    // positions at the current bar. Stuck positions (held in illiquid
    // names where exit would breach the gate) get heavily haircut here,
    // surfacing wealth that the strategy cannot actually realize. Phase 1
    // is REPORTING ONLY — these metrics are diagnostic; trial_metrics
    // (used by the tuner) still operates off mark-NAV returns. Once
    // empirically validated this becomes the headline series, per
    // Frazzini-Israel-Moskowitz 2018 / Novy-Marx-Velikov 2016 convention
    // for net-of-cost performance evaluation.
    let final_nav_liquid = records.last().unwrap().nav_liquid;
    let final_haircut = records.last().unwrap().liquidation_haircut;
    // We use mark-NAV as the start (start_nav above, computed from
    // records[0]), since at t=0 there are no open positions and no
    // haircut.
    let total_return_liquid = (final_nav_liquid / start_nav - 1.0) * 100.0;
    let mut peak_liquid = start_nav;
    let mut worst_dd_liquid = 0.0f64;
    for r in records {
        if r.nav_liquid > peak_liquid { peak_liquid = r.nav_liquid; }
        let dd = if peak_liquid > 1e-12 { (r.nav_liquid - peak_liquid) / peak_liquid } else { 0.0 };
        if dd < worst_dd_liquid { worst_dd_liquid = dd; }
    }

    // Liquid-NAV Sharpe: compute returns from nav_liquid series. Defensive
    // division (skip bars where prior nav_liquid ≤ 0). Mean / vol on the
    // resulting series; annualization same as the trial metrics path.
    let n_records = records.len();
    let bars_per_year_for_sharpe = if n_records > 0 {
        // Estimate: time span / n_bars × seconds-in-year
        let span_secs = (records.last().unwrap().timestamp - records[0].timestamp).max(1);
        let span_years = span_secs as f64 / 31_557_600.0;
        if span_years > 1e-6 { n_records as f64 / span_years } else { 365.0 }
    } else { 365.0 };
    let mut liquid_returns: Vec<f64> = Vec::with_capacity(n_records);
    let mut prev_liq = start_nav;
    for r in records {
        if prev_liq.abs() > 1e-12 && r.nav_liquid.is_finite() && prev_liq.is_finite() {
            liquid_returns.push(r.nav_liquid / prev_liq - 1.0);
        }
        prev_liq = r.nav_liquid;
    }
    let sharpe_liquid = if liquid_returns.len() > 1 {
        let mean = liquid_returns.iter().sum::<f64>() / liquid_returns.len() as f64;
        let var = liquid_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (liquid_returns.len() - 1) as f64;
        let std = var.sqrt();
        if std > 1e-12 {
            mean / std * bars_per_year_for_sharpe.sqrt()
        } else { 0.0 }
    } else { 0.0 };

    let last_record = records.last().unwrap();
    println!("  Start NAV:      ${:.2}", start_nav);
    println!("  Final NAV:      ${:.2} (mark, compounded)", final_nav);
    println!("  Final NAV liq:  ${:.2} (post-haircut, compounded)", final_nav_liquid);
    println!("  End haircut:    ${:.2} ({:.2}% of mark NAV)",
        final_haircut,
        if final_nav > 1e-6 { 100.0 * final_haircut / final_nav } else { 0.0 });
    let last = records.last().unwrap();
    let h_total = (last.liquidation_haircut_current + last.liquidation_haircut_last_known + last.liquidation_haircut_dlom).max(1e-12);
    println!("    by source:    current=${:.2} ({:.1}%)  last-known=${:.2} ({:.1}%)  DLOM=${:.2} ({:.1}%, n={})",
        last.liquidation_haircut_current, 100.0 * last.liquidation_haircut_current / h_total,
        last.liquidation_haircut_last_known, 100.0 * last.liquidation_haircut_last_known / h_total,
        last.liquidation_haircut_dlom, 100.0 * last.liquidation_haircut_dlom / h_total,
        last.n_positions_dlom);
    if result.cumulative_skimmed_usd > 0.0 {
        let n_events = result.per_bar_skimmed.len();
        let synth_total = final_nav + result.cumulative_skimmed_usd;
        println!("  ---");
        println!("  Cumulative skim:    ${:.2} ({} events)", result.cumulative_skimmed_usd, n_events);
        println!("  Synth total wealth: ${:.2}  (working NAV + skimmed; what an uncapped run would have)", synth_total);
    }
    println!("  Return:         {:.1}% (mark)", total_return);
    println!("  Return liq:     {:.1}% (post-haircut)", total_return_liquid);
    println!("  Additive P&L:   ${:.2} ({:.1}% of operating capital)",
        last_record.additive_return * start_nav,
        last_record.additive_return * 100.0);
    println!("  MDD:            {:.1}% (mark)", worst_dd * 100.0);
    println!("  MDD liq:        {:.1}% (post-haircut)", worst_dd_liquid * 100.0);
    println!("  Sharpe liq:     {:.3} (post-haircut, diagnostic)", sharpe_liquid);
    println!("  ---");
    println!("  Commission:     ${:.2}", total_commission);
    println!("  Spread cost:    ${:.2}", total_spread);
    println!("  Impact cost:    ${:.2}", total_impact);
    println!("  Total costs:    ${:.2}", total_costs);
    println!("  Funding PnL:    ${:.2}", total_funding);
    println!("  ---");
    println!("  Avg gross:      {:.3}x", avg_gross_leverage);
    println!("  Max gross:      ${:.0}", max_gross);
    println!("  Avg turnover:   {:.4}", avg_turnover);
    println!("  Avg positions:  {:.0}", avg_positions);
    let min_liq_dist: f64 = records.iter()
        .map(|r| r.liquidation_distance)
        .fold(f64::INFINITY, f64::min);

    println!("  Min margin:     {:.2}x", min_margin);
    println!("  Min liq dist:   {:.1}% (0%=liquidated)", min_liq_dist * 100.0);
    println!("  Liquidations:   {}", result.liquidation_events.len());
    if result.n_rejected_trades > 0 {
        println!("  Gate rejects:   {} trades, ${:.2}M total notional refused",
            result.n_rejected_trades,
            result.rejected_notional / 1e6);
    }
    let total_funding_bars = result.n_funding_actual + result.n_funding_model;
    if total_funding_bars > 0 {
        println!("  Funding source: {} actual / {} model ({:.1}% model-estimated)",
            result.n_funding_actual, result.n_funding_model,
            result.n_funding_model as f64 / total_funding_bars as f64 * 100.0);
    }
    println!("  Bars:           {}", records.len());

    // Bookkeeping consistency: each day's pre-reset
    //   discrepancy = (post_nav − day_start_nav) − (trading_pnl + funding − costs)
    // is computed independently (trading_pnl from book.realized + book.unrealized
    // deltas; not a residual of NAV change). Sum across days is the cumulative
    // accounting error. Should be ≤ float ULP × n_days if cash flows are
    // consistent; large nonzero values flag a real bug.
    let cum_discrepancy: f64 = records.iter().map(|r| r.bookkeeping_discrepancy).sum();
    let abs_cum_discrepancy: f64 = records.iter().map(|r| r.bookkeeping_discrepancy.abs()).sum();
    let max_daily_discrepancy: f64 = records.iter()
        .map(|r| r.bookkeeping_discrepancy.abs())
        .fold(0.0f64, f64::max);
    println!("  Bookkeeping:    cum=${:.4}  Σ|daily|=${:.4}  max|daily|=${:.4}",
        cum_discrepancy, abs_cum_discrepancy, max_daily_discrepancy);

    // Stuck positions: in book but last updated long ago (stale mark)
    if !records.is_empty() {
        let last_ts = records.last().unwrap().timestamp;
        let mut stuck: Vec<(&str, f64, i64)> = Vec::new();
        for pos in book.positions.values() {
            if !pos.is_flat() && pos.last_fill_at < last_ts - 30 * 86400 {
                stuck.push((&pos.symbol, pos.notional(), pos.last_fill_at));
            }
        }
        if !stuck.is_empty() {
            let stuck_gross: f64 = stuck.iter().map(|(_, n, _)| n).sum();
            println!("  Stuck positions: {} (${:.0} gross, last traded >30d ago)", stuck.len(), stuck_gross);
            stuck.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (sym, notional, last_fill) in &stuck {
                let days_ago = (last_ts - last_fill) / 86400;
                println!("    {:<12} ${:>10.2}  ({}d ago)", sym, notional, days_ago);
            }
        }
    }
}

/// Convert alpha rows to dollar-alpha based on the declared units.
///
/// The QP solves `min γw'Σw − α'w + ...` and treats α as **dollar-PnL per
/// unit weight**. If the alpha pipeline produced predictions in any other
/// units, this function applies the per-target adapter to recover dollar
/// alphas BEFORE the QP sees them.
///
/// Contract (Layer 1): every supported alpha target must declare its units
/// here. Adding a new target type means extending the `AlphaUnits` enum and
/// adding the conversion. Layer 2 (parquet metadata + config check) prevents
/// silent disagreement; Layer 3 (the unit tests below) prevents the
/// conversion from regressing.
///
/// Reference: `train/rumpy_train/targets/excess_volnorm.py` —
/// "At execution time, multiply predictions by sigma_i to recover dollar
///  alphas." was the contract; this function makes it enforceable.
pub fn convert_alpha_units(
    alphas: Vec<AlphaRow>,
    sigma_lookup: &HashMap<(i64, u32), f64>,
    symbols: &SymbolTable,
    units: crate::config::AlphaUnits,
) -> Vec<AlphaRow> {
    use crate::config::AlphaUnits;
    match units {
        AlphaUnits::DollarAlpha => alphas,
        AlphaUnits::VolNormalizedExcess => {
            let mut converted = Vec::with_capacity(alphas.len());
            let mut n_dropped = 0usize;
            for a in alphas {
                let sid = match symbols.to_id.get(&a.symbol) {
                    Some(&i) => i,
                    None => { n_dropped += 1; continue; }
                };
                match sigma_lookup.get(&(a.timestamp, sid)) {
                    Some(&sigma) if sigma > 0.0 => {
                        converted.push(AlphaRow {
                            timestamp: a.timestamp,
                            symbol: a.symbol,
                            alpha_future: a.alpha_future * sigma,
                        });
                    }
                    _ => { n_dropped += 1; }
                }
            }
            println!(
                "  alpha_units adapter: VolNormalizedExcess → DollarAlpha ({} converted, {} dropped no σ)",
                converted.len(), n_dropped
            );
            converted
        }
    }
}

/// Score-weighted unit-gross benchmark.
pub fn score_weighted_benchmark(
    alphas: &[AlphaRow],
    forward_returns: &HashMap<(i64, u32), f64>,
    symbol_ids: &HashMap<String, u32>,
    bars_per_year: f64,
) -> f64 {
    // BTreeMap so outer iteration is sorted by ts, making bar_returns Vec
    // ordering deterministic. Sharpe is technically order-insensitive value-
    // set-wise, but mean/std sums have ULP-level order dependence.
    // See docs/plans/cross-platform-divergence.md (Patch 3).
    use std::collections::BTreeMap;
    let mut by_ts: BTreeMap<i64, Vec<(&str, f64)>> = BTreeMap::new();
    for row in alphas {
        by_ts.entry(row.timestamp).or_default().push((&row.symbol, row.alpha_future));
    }

    let mut bar_returns: Vec<f64> = Vec::new();
    for (ts, assets) in &by_ts {
        if assets.len() < 2 { continue; }
        let mean_alpha = assets.iter().map(|(_, a)| a).sum::<f64>() / assets.len() as f64;
        let demeaned: Vec<(&str, f64)> = assets.iter().map(|(s, a)| (*s, a - mean_alpha)).collect();
        let gross: f64 = demeaned.iter().map(|(_, d)| d.abs()).sum();
        if gross < 1e-12 { continue; }
        let bar_ret: f64 = demeaned.iter().map(|(sym, d)| {
            let w = d / gross;
            let r = symbol_ids.get(*sym).and_then(|&sid| forward_returns.get(&(*ts, sid))).copied().unwrap_or(0.0);
            w * r
        }).sum();
        bar_returns.push(bar_ret);
    }
    metrics::sharpe_annualized(&bar_returns, bars_per_year)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Layer 3 tests: alpha-units contract regression suite.
    //
    // These tests would have caught the original bug (vol-normalized alpha
    // fed straight into a dollar-PnL QP) at the first commit. They MUST
    // stay green; if they go red on a future change, the units adapter is
    // broken and dollar-alpha is being misreported to the QP.
    // ---------------------------------------------------------------------

    fn make_test_symbols(syms: &[&str]) -> SymbolTable {
        let mut t = SymbolTable::new();
        for s in syms { t.intern(s); }
        t
    }

    #[test]
    fn test_alpha_units_dollar_alpha_is_identity() {
        // DollarAlpha mode must NOT transform alphas at all — passes through.
        let symbols = make_test_symbols(&["BTC", "ETH"]);
        let day = 86400_i64;
        let alphas = vec![
            AlphaRow { timestamp: day, symbol: "BTC".into(), alpha_future: 0.5 },
            AlphaRow { timestamp: day, symbol: "ETH".into(), alpha_future: -0.3 },
        ];
        // sigma_lookup populated but should be ignored
        let mut sigma = HashMap::new();
        sigma.insert((day, symbols.to_id["BTC"]), 0.10);
        sigma.insert((day, symbols.to_id["ETH"]), 0.05);
        let out = convert_alpha_units(alphas.clone(), &sigma, &symbols, crate::config::AlphaUnits::DollarAlpha);
        assert_eq!(out.len(), 2);
        assert!((out[0].alpha_future - 0.5).abs() < 1e-12, "BTC alpha was modified: {}", out[0].alpha_future);
        assert!((out[1].alpha_future - (-0.3)).abs() < 1e-12, "ETH alpha was modified: {}", out[1].alpha_future);
    }

    #[test]
    fn test_alpha_units_volnorm_multiplies_by_sigma() {
        // VolNormalizedExcess mode must multiply alpha by per-asset sigma.
        // This is the exact transformation that was missing from the old
        // pipeline and that recovered the dollar-alpha units expected by
        // the QP — see contract from train/rumpy_train/targets/excess_volnorm.py.
        let symbols = make_test_symbols(&["BTC", "ETH"]);
        let day = 86400_i64;
        let alphas = vec![
            AlphaRow { timestamp: day, symbol: "BTC".into(), alpha_future: 0.5 },
            AlphaRow { timestamp: day, symbol: "ETH".into(), alpha_future: -0.3 },
        ];
        let mut sigma = HashMap::new();
        sigma.insert((day, symbols.to_id["BTC"]), 0.10);
        sigma.insert((day, symbols.to_id["ETH"]), 0.05);
        let out = convert_alpha_units(alphas, &sigma, &symbols, crate::config::AlphaUnits::VolNormalizedExcess);
        assert_eq!(out.len(), 2);
        // BTC: 0.5 × 0.10 = 0.05
        assert!((out[0].alpha_future - 0.05).abs() < 1e-12, "BTC dollar alpha wrong: {}", out[0].alpha_future);
        // ETH: -0.3 × 0.05 = -0.015
        assert!((out[1].alpha_future - (-0.015)).abs() < 1e-12, "ETH dollar alpha wrong: {}", out[1].alpha_future);
    }

    #[test]
    fn test_alpha_units_volnorm_dollar_aware_ranking_reverses() {
        // Two assets with same z-score (0.1) but 10× different vol.
        //   A: vol=0.01 → dollar alpha = 0.001
        //   B: vol=0.10 → dollar alpha = 0.010
        // After conversion the ranking by alpha magnitude must put B above A,
        // even though they have identical raw z-scores. Under DollarAlpha (no
        // conversion) the QP would equally weight A and B, missing the fact
        // that B carries 10× more dollar PnL per unit weight.
        let symbols = make_test_symbols(&["A", "B"]);
        let day = 86400_i64;
        let alphas = vec![
            AlphaRow { timestamp: day, symbol: "A".into(), alpha_future: 0.1 },
            AlphaRow { timestamp: day, symbol: "B".into(), alpha_future: 0.1 },
        ];
        let mut sigma = HashMap::new();
        sigma.insert((day, symbols.to_id["A"]), 0.01);
        sigma.insert((day, symbols.to_id["B"]), 0.10);

        // DollarAlpha: identical alphas → equal weighting in QP.
        let dollar = convert_alpha_units(alphas.clone(), &sigma, &symbols, crate::config::AlphaUnits::DollarAlpha);
        assert!((dollar[0].alpha_future - dollar[1].alpha_future).abs() < 1e-12,
                "DollarAlpha incorrectly differentiated identical z-scores");

        // VolNormalizedExcess: B's dollar alpha must be 10× A's.
        let dollar_aware = convert_alpha_units(alphas, &sigma, &symbols, crate::config::AlphaUnits::VolNormalizedExcess);
        let ratio = dollar_aware[1].alpha_future / dollar_aware[0].alpha_future;
        assert!((ratio - 10.0).abs() < 1e-12,
                "B/A dollar-alpha ratio should be 10×, got {ratio}");
    }

    #[test]
    fn test_alpha_units_volnorm_drops_missing_sigma() {
        // Rows whose (day, sid) has no sigma must be dropped, not silently
        // assigned zero (which would hand the QP a wrong-magnitude alpha).
        let symbols = make_test_symbols(&["BTC", "ETH"]);
        let day = 86400_i64;
        let alphas = vec![
            AlphaRow { timestamp: day, symbol: "BTC".into(), alpha_future: 0.5 },
            AlphaRow { timestamp: day, symbol: "ETH".into(), alpha_future: 0.3 },
        ];
        let mut sigma = HashMap::new();
        sigma.insert((day, symbols.to_id["BTC"]), 0.10);
        // ETH sigma intentionally absent
        let out = convert_alpha_units(alphas, &sigma, &symbols, crate::config::AlphaUnits::VolNormalizedExcess);
        assert_eq!(out.len(), 1, "missing-sigma row should have been dropped");
        assert_eq!(out[0].symbol, "BTC");
    }

    /// Measure EXACT memory per data structure using HashMap::capacity() and size_of.
    /// No RSS noise. No ambiguity. Pure arithmetic from actual allocated sizes.
    #[test]
    fn measure_actual_allocation_bytes() {
        use std::mem::size_of;
        use std::collections::HashMap;

        let symbols: Vec<String> = (0..921).map(|i| format!("SYM{:04}", i)).collect();
        let avg_sym_len: usize = symbols.iter().map(|s| s.len()).sum::<usize>() / symbols.len();
        let n = 1_000_000usize;

        // --- HashMap<(i64, String), f64> ---
        let old_map: HashMap<(i64, String), f64> = (0..n)
            .map(|i| ((i as i64, symbols[i % 921].clone()), i as f64 * 0.01))
            .collect();
        let old_cap = old_map.capacity();
        // hashbrown stores (K,V) pairs in a flat array + 1 control byte per slot
        let old_bucket_array = old_cap * (size_of::<(i64, String)>() + size_of::<f64>()) + old_cap;
        // Each String value has a heap allocation: the actual string bytes
        // Rust's allocator rounds up to size classes. For 7-byte strings: typically 8 bytes.
        // But String also stores capacity which may differ from len.
        let old_heap_per_string = {
            // Measure actual: create a String, check its capacity
            let s = symbols[0].clone();
            s.capacity() // this is how many bytes are allocated on heap
        };
        let old_total_heap = n * old_heap_per_string;
        let old_total = old_bucket_array + old_total_heap;
        let old_bpe = old_total as f64 / n as f64;

        // --- HashMap<(i64, u32), f64> ---
        let new_map: HashMap<(i64, u32), f64> = (0..n)
            .map(|i| ((i as i64, (i % 921) as u32), i as f64 * 0.01))
            .collect();
        let new_cap = new_map.capacity();
        let new_bucket_array = new_cap * (size_of::<(i64, u32)>() + size_of::<f64>()) + new_cap;
        let new_bpe = new_bucket_array as f64 / n as f64;

        // --- AlphaRow / AimWeight ---
        let alpha_inline = size_of::<crate::alpha::AlphaRow>();
        let aim_inline = size_of::<crate::aim::AimWeight>();
        // Each has a String field → heap alloc per entry
        let alpha_bpe = alpha_inline as f64 + old_heap_per_string as f64;
        let aim_bpe = aim_inline as f64 + old_heap_per_string as f64;

        // Keep alive
        assert!(old_map.len() + new_map.len() > 0);

        println!("\n=== EXACT MEMORY BREAKDOWN ===");
        println!("  String '{}': len={}, capacity={}, heap_alloc={} bytes",
            &symbols[0], symbols[0].len(), old_heap_per_string, old_heap_per_string);
        println!();
        println!("  HashMap<(i64,String),f64>:");
        println!("    entries:       {}", n);
        println!("    capacity:      {} (load factor: {:.1}%)", old_cap, n as f64 / old_cap as f64 * 100.0);
        println!("    bucket array:  {} MB", old_bucket_array / 1_000_000);
        println!("    string heap:   {} MB ({} entries × {} bytes)", old_total_heap / 1_000_000, n, old_heap_per_string);
        println!("    TOTAL:         {} MB ({:.1} bytes/entry)", old_total / 1_000_000, old_bpe);
        println!();
        println!("  HashMap<(i64,u32),f64>:");
        println!("    entries:       {}", n);
        println!("    capacity:      {} (load factor: {:.1}%)", new_cap, n as f64 / new_cap as f64 * 100.0);
        println!("    bucket array:  {} MB", new_bucket_array / 1_000_000);
        println!("    string heap:   0 MB (interned)");
        println!("    TOTAL:         {} MB ({:.1} bytes/entry)", new_bucket_array / 1_000_000, new_bpe);
        println!();
        println!("  AlphaRow:  {}B inline + {}B heap = {:.0}B total/entry", alpha_inline, old_heap_per_string, alpha_bpe);
        println!("  AimWeight: {}B inline + {}B heap = {:.0}B total/entry", aim_inline, old_heap_per_string, aim_bpe);

        let nl = 14_900_000.0f64;
        let nh1 = 7_300_000.0f64;
        println!("\n=== PROJECTED AT SCALE ===");
        println!("  Old lookups (7 × {:.1}M): {:.2} GB", nl/1e6, old_bpe * nl * 7.0 / 1e9);
        println!("  New lookups (7 × {:.1}M): {:.2} GB", nl/1e6, new_bpe * nl * 7.0 / 1e9);
        println!("  Lookup savings:           {:.2} GB ({:.0}%)",
            (old_bpe - new_bpe) * nl * 7.0 / 1e9,
            (1.0 - new_bpe / old_bpe) * 100.0);
        println!();
        println!("  Per H1 trial (alpha+aim): {:.0} MB", (alpha_bpe + aim_bpe) * nh1 / 1e6);
        println!("  × 20 threads:             {:.2} GB", (alpha_bpe + aim_bpe) * nh1 * 20.0 / 1e9);
        println!();
        println!("  TOTAL w/ old lookups + 20 H1 trials: {:.2} GB",
            old_bpe * nl * 7.0 / 1e9 + (alpha_bpe + aim_bpe) * nh1 * 20.0 / 1e9);
        println!("  TOTAL w/ new lookups + 20 H1 trials: {:.2} GB",
            new_bpe * nl * 7.0 / 1e9 + (alpha_bpe + aim_bpe) * nh1 * 20.0 / 1e9);
        println!();
        // --- Per-trial simulation allocations (NOT measured above) ---
        println!("\n=== PER-TRIAL SIMULATION ALLOCATIONS ===");

        // SimulationRecord: stored per bar, each has weights: Vec<(String, f64)>
        let record_inline = size_of::<crate::simulation::SimulationRecord>();
        println!("  SimulationRecord inline: {} bytes", record_inline);
        // weights Vec: ~150 positions per bar, each (String, f64)
        let weight_entry = size_of::<(String, f64)>() as f64 + old_heap_per_string as f64; // String inline + heap
        let avg_positions = 150.0f64;
        let weights_per_bar = weight_entry * avg_positions;
        println!("  Per weight entry: {:.0}B ({}B struct + {}B heap)", weight_entry, size_of::<(String, f64)>(), old_heap_per_string);
        println!("  Per bar weights ({:.0} positions): {:.0} bytes", avg_positions, weights_per_bar);

        let h1_bars = 48_000.0f64; // ~2000 days × 24 hours
        let d1_bars = 2_210.0f64;
        let h1_records = (record_inline as f64 + weights_per_bar) * h1_bars;
        let d1_records = (record_inline as f64 + weights_per_bar) * d1_bars;
        println!("  H1 records ({:.0}K bars): {:.0} MB", h1_bars/1000.0, h1_records / 1e6);
        println!("  D1 records ({:.0}K bars):  {:.0} MB", d1_bars/1000.0, d1_records / 1e6);

        // PortfolioBook: BTreeMap<String, PositionState>
        let pos_state = size_of::<crate::position::PositionState>();
        println!("  PositionState: {} bytes", pos_state);
        println!("  PortfolioBook ({:.0} positions): ~{:.0} KB", avg_positions, avg_positions * (pos_state as f64 + 40.0) / 1024.0);

        // bar_prices: HashMap<String, f64> rebuilt every bar (temporary but concurrent)
        let bar_prices_per_bar = avg_positions * (size_of::<(String, f64)>() as f64 + old_heap_per_string as f64 + 40.0); // HashMap overhead
        println!("  bar_prices per bar: {:.0} bytes ({:.0} entries)", bar_prices_per_bar, avg_positions);

        // gamma_schedule: HashMap<i64, f64>
        let gamma_entries = if true { h1_bars } else { d1_bars }; // depends on alpha count
        let gamma_size = gamma_entries * (size_of::<(i64, f64)>() as f64 + 1.0) / 0.875;
        println!("  gamma_schedule H1: {:.0} KB", gamma_size / 1024.0);

        // ExecRow Vec during parquet load (peak, before drop)
        let rows_peak = 14_900_000.0 * 88.0;
        println!("  Parquet rows (peak during load): {:.0} MB", rows_peak / 1e6);

        // --- COMPLETE MEMORY BUDGET ---
        println!("\n=== COMPLETE MEMORY BUDGET (20 H1 threads) ===");
        let lookups_old = old_bpe * nl * 7.0;
        let lookups_new = new_bpe * nl * 7.0;
        let clones = (alpha_bpe + aim_bpe) * nh1;
        let sim_records = h1_records;
        let sim_overhead = bar_prices_per_bar + gamma_size + avg_positions * (pos_state as f64 + 40.0);
        let per_trial = clones + sim_records + sim_overhead;
        let parquet_peak = rows_peak;
        let python_os = 500.0 * 1e6; // ~500 MB

        println!("  Lookups (old String):     {:.2} GB", lookups_old / 1e9);
        println!("  Lookups (interned u32):   {:.2} GB", lookups_new / 1e9);
        println!("  Per trial breakdown:");
        println!("    alpha+aim clones:       {:.0} MB", clones / 1e6);
        println!("    simulation records:     {:.0} MB", sim_records / 1e6);
        println!("    sim overhead:           {:.0} MB", sim_overhead / 1e6);
        println!("    TOTAL per trial:        {:.0} MB", per_trial / 1e6);
        println!("  × 20 trials:              {:.2} GB", per_trial * 20.0 / 1e9);
        println!("  Parquet peak (load only):  {:.2} GB", parquet_peak / 1e9);
        println!("  Python/OS:                {:.2} GB", python_os / 1e9);
        println!("  ---");
        println!("  GRAND TOTAL (old lookups): {:.2} GB",
            (lookups_old + per_trial * 20.0 + parquet_peak + python_os) / 1e9);
        println!("  GRAND TOTAL (new lookups): {:.2} GB",
            (lookups_new + per_trial * 20.0 + parquet_peak + python_os) / 1e9);

        // With FULL interning (AlphaRow/AimWeight also use u32)
        let alpha_interned = (size_of::<i64>() + size_of::<u32>() + size_of::<f64>()) as f64;
        let aim_interned = alpha_interned;
        let clones_interned = (alpha_interned + aim_interned) * nh1;
        // SimulationRecord.weights would also use u32 → saves heap allocs
        let weight_interned = (size_of::<(u32, f64)>() as f64) + 0.0; // no heap
        let h1_records_interned = (record_inline as f64 + weight_interned * avg_positions) * h1_bars;
        let per_trial_interned = clones_interned + h1_records_interned + sim_overhead;
        println!();
        println!("  WITH FULL INTERNING:");
        println!("    Per trial:  {:.0} MB (was {:.0})", per_trial_interned / 1e6, per_trial / 1e6);
        println!("    × 20:       {:.2} GB (was {:.2})", per_trial_interned * 20.0 / 1e9, per_trial * 20.0 / 1e9);
        println!("    GRAND TOTAL: {:.2} GB",
            (lookups_new + per_trial_interned * 20.0 + parquet_peak + python_os) / 1e9);
    }

    /// Profile heap allocations with dhat.
    /// Run: cargo test -p rumpy-execution --features heap-profile heap_profile -- --nocapture --test-threads=1
    /// Output: dhat-heap.json (open at https://nnethercote.github.io/dh_view/dh_view.html)
    #[test]
    #[cfg(feature = "heap-profile")]
    fn heap_profile_preload_and_trial() {
        #[global_allocator]
        static ALLOC: dhat::Alloc = dhat::Alloc;

        let _profiler = dhat::Profiler::new_heap();

        // Try relative (workspace root) then common absolute paths
        let candidates = [
            "crates/features/data/execution_h1.parquet".to_string(),
            format!("{}/crates/features/data/execution_h1.parquet", env!("CARGO_MANIFEST_DIR").replace("/crates/execution", "")),
        ];
        let parquet = candidates.iter()
            .map(std::path::PathBuf::from)
            .find(|p| p.exists());
        let parquet = match parquet {
            Some(p) => { println!("Using parquet: {:?}", p); p },
            None => { println!("Skipping: parquet not found"); return; }
        };

        // Use H1 config for worst-case profiling
        let config_str = r#"
qp: { gamma: 10.0, lambda_aim: 1.0, l_max: 1.5, per_name_cap: 0.05, ridge: 0.000001 }
covariance: { n_factors: 8, ewma_halflife: 45, min_history: 60, min_assets: 10, winsorize_sigma: 5.0, ridge: 0.000001 }
alpha: { horizons: [3, 6, 12, 24, 50], winsorize_sigma: 3.0, l_target: 1.0 }
funding: { enabled: true, ewma_halflife_hours: 24, long_run_mean_window_hours: 720, default_rho: 0.89, holding_hours: 24 }
cost_model: { c_lin_multiplier: 1.0, sigma_halflife_days: 30, adv_halflife_days: 30 }
nav_usd: 1000.0
fill_mode: twap
"#;
        let config: crate::config::ExecutionConfig = serde_yaml::from_str(config_str).unwrap();
        let paths = BacktestPaths { execution_h1: parquet };

        println!("=== Preloading H1 data ===");
        let data = PreloadedData::load(&paths, &config).unwrap();
        let stats = dhat::HeapStats::get();
        println!("After preload: curr={:.0} MB, peak={:.0} MB, total_allocs={}",
            stats.curr_bytes as f64 / 1e6, stats.max_bytes as f64 / 1e6, stats.total_blocks);

        println!("\n=== Running H1 trial ===");
        let wf = crate::config::WalkForwardConfig::default();
        let _result = run_trial(&data, &config, &wf, false).unwrap();
        let stats = dhat::HeapStats::get();
        println!("After H1 trial: curr={:.0} MB, peak={:.0} MB, total_allocs={}",
            stats.curr_bytes as f64 / 1e6, stats.max_bytes as f64 / 1e6, stats.total_blocks);

        // Now run 2 concurrent H1 trials to measure concurrent overhead
        println!("\n=== Running 2 CONCURRENT H1 trials ===");
        let data_ref = &data;
        let config_ref = &config;
        std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                let wf = crate::config::WalkForwardConfig::default();
                run_trial(data_ref, config_ref, &wf, false).unwrap();
            });
            let h2 = s.spawn(|| {
                let wf = crate::config::WalkForwardConfig::default();
                run_trial(data_ref, config_ref, &wf, false).unwrap();
            });
            h1.join().unwrap();
            h2.join().unwrap();
        });
        let stats = dhat::HeapStats::get();
        println!("After 2 concurrent H1: curr={:.0} MB, peak={:.0} MB, total_allocs={}",
            stats.curr_bytes as f64 / 1e6, stats.max_bytes as f64 / 1e6, stats.total_blocks);

        // dhat-heap.json is written on drop of _profiler
        println!("\n=== dhat-heap.json written. Open at https://nnethercote.github.io/dh_view/dh_view.html ===");
    }

    #[test]
    fn measure_memory_layout() {
        use std::mem::size_of;

        println!("\n=== Struct sizes (bytes) ===");
        println!("  ExecRow:    {}", size_of::<ExecRow>());
        println!("  AlphaRow:   {}", size_of::<AlphaRow>());
        println!("  AimWeight:  {}", size_of::<crate::aim::AimWeight>());
        println!("  String:     {}", size_of::<String>());

        println!("\n=== HashMap key sizes ===");
        println!("  (i64, String): {}", size_of::<(i64, String)>());
        println!("  (i64, u32):    {}", size_of::<(i64, u32)>());

        println!("\n=== HashMap entry sizes (K+V inline, no overhead) ===");
        println!("  ((i64,String), f64):  {}", size_of::<((i64, String), f64)>());
        println!("  ((i64,u32), f64):     {}", size_of::<((i64, u32), f64)>());

        let old_kv = size_of::<((i64, String), f64)>() as f64;
        let new_kv = size_of::<((i64, u32), f64)>() as f64;
        // hashbrown: 1 control byte per bucket, load factor 7/8
        let old_per = (old_kv + 1.0) / 0.875;
        let new_per = (new_kv + 1.0) / 0.875;

        println!("\n=== Per-entry with hashbrown overhead (7/8 load) ===");
        println!("  Old (String key):  {:.1} bytes/entry", old_per);
        println!("  New (u32 key):     {:.1} bytes/entry", new_per);
        println!("  + String heap alloc per old entry: ~8-16 bytes");

        let n: f64 = 14_900_000.0;
        let old_heap = 12.0; // avg symbol ~8 chars + allocator overhead
        println!("\n=== Projected at 14.9M entries x 7 lookups ===");
        println!("  Old lookups (inline):  {:.1} GB", old_per * n * 7.0 / 1e9);
        println!("  Old + heap allocs:     {:.1} GB", (old_per + old_heap) * n * 7.0 / 1e9);
        println!("  New lookups (interned): {:.1} GB", new_per * n * 7.0 / 1e9);
        println!("  Savings:               {:.1} GB ({:.0}%)",
            (old_per + old_heap - new_per) * n * 7.0 / 1e9,
            (1.0 - new_per / (old_per + old_heap)) * 100.0);

        let alpha = size_of::<AlphaRow>() as f64;
        let aim = size_of::<crate::aim::AimWeight>() as f64;
        let n_h1: f64 = 7_300_000.0;
        println!("\n=== Per H1 trial clone cost ===");
        println!("  AlphaRow:  {} + ~12 heap = {:.0} MB", alpha as usize, (alpha + 12.0) * n_h1 / 1e6);
        println!("  AimWeight: {} + ~12 heap = {:.0} MB", aim as usize, (aim + 12.0) * n_h1 / 1e6);
        println!("  Combined per trial: {:.0} MB", (alpha + aim + 24.0) * n_h1 / 1e6);
        println!("  x20 threads: {:.0} MB", (alpha + aim + 24.0) * n_h1 * 20.0 / 1e6);
    }
}
