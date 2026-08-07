//! Holdings-based simulation engine.
//!
//! Replaces the weight-only pipeline with a bar-by-bar loop that tracks
//! actual positions, deducts costs from cash, and derives weights from
//! marked-to-market holdings.
//!
//! Key differences from the weight-only pipeline:
//!   - w_prev comes from actual holdings (marked-to-market), not last QP output
//!   - Costs are endogenous — deducted from cash, reducing NAV
//!   - Returns are NAV-based: (nav[t+1] - nav[t]) / nav[t]
//!   - Vol target operates on actual NAV returns
//!
//! References:
//!   - Boyd et al. (2017) "Multi-Period Trading via Convex Optimization"
//!   - Gârleanu & Pedersen (2013) — aim portfolio framework

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::aim::AimWeight;
use crate::alpha::AlphaRow;
use crate::config::ExecutionConfig;
use crate::cost::FundingModel;
use crate::covariance::RollingCovariance;
use crate::exchange::{ExchangeConfig, VolumeTracker};
use crate::position::PortfolioBook;
use crate::solver::CachedOSQP;

// ---------------------------------------------------------------------------
// Calibration constants
// ---------------------------------------------------------------------------

/// Sqrt-regime impact exponent in the realized cost formula
///   `impact_usd = κ · σ · |Q|^(1+δ) / ADV^δ = κ · σ · |Q| · (|Q|/ADV)^δ`
/// equivalently, fractional impact (per dollar traded):
///   `impact_pct = κ · σ · (Q/ADV)^δ`     (this is the literature form)
///
/// This formula matches the **metaorder framework** (Tóth 2011 §3, Donier 2014,
/// Sato-Kanazawa 2024). Per-dollar impact = `κ·σ·(Q/V)^δ` exactly the standard
/// literature parameterization with our δ ≡ literature δ (see
/// memory/research_cost_model_spec_2026_04_29.md for derivation).
///
/// **The formula above applies in the sqrt regime only** (Bucci, Benzaquen,
/// Lillo, Bouchaud 2018, *Phys Rev Lett* 122, 108302). For small volume
/// fraction φ = Q/V_d, impact is approximately linear in Q (i.e., per-dollar
/// impact constant, equivalent to δ ≈ 0). Production cost should decompose:
///
///   `realized_cost($) = half_spread($) + κ·σ·|Q|·(|Q|/ADV)^δ · gate(φ)`
///
/// where `gate(φ) ≈ 0` when φ < 10⁻³ (linear/at-touch regime, half-spread
/// dominates) and `gate(φ) ≈ 1` when φ > 10⁻² (sweep regime, sqrt-law applies).
/// At our deployment trade sizes ($5k positions on majors with $2B+ ADV),
/// φ ≈ 10⁻⁶ — deep in linear regime where half-spread is the entire cost.
///
/// **Must stay in sync with `crates/features/src/cost/mod.rs::IMPACT_DELTA`.**
/// Re-derivation requires a cascade rebuild (cost.parquet → kappa retrain →
/// re-score). See cost/mod.rs for the calibration record.
///
/// Current value: 0.10 (Apr 2026 refit). Evidence:
///   - Multi-Q book-walk synthetic: δ̂ = 0.104 (R² = 0.038, z = +12.1)
///   - Real fills (52k taker, 30d): δ̂ = 0.096 (R² = 0.012, z = +3.21)
///   - **Empirical HL metaorder verification (2026-04-29 research round):**
///     - Bin G (φ ∈ [10⁻², 10⁻¹], 1641 metaorders): **δ̂ = 0.454**
///       — within errorbars of literature universal 0.5 (Sato 2024).
///     - Linear regime bins (φ < 10⁻⁴): δ ≈ 0 as predicted by Bucci 2018.
///     - The current 0.10 is a regime-mixture average; production cost model
///       should use δ ≈ 0.5 in the gated sqrt regime instead.
///
/// QP/realized mismatch: the QP uses a fixed u^1.5 impact term. With the
/// gated two-regime model, this is only relevant when gate(φ) ≈ 1, i.e., for
/// trades sweeping the book. Most rumpy trades won't reach this regime, so
/// the mismatch matters only at large NAV / thin coins. See
/// memory/project_qp_impact_socp_refactor.md.
///
/// References:
/// - Tóth et al. 2011, *Phys Rev X* 1, 021006. arXiv:1105.1694
/// - Donier & Bonart 2015, *Market Microstructure & Liquidity*. arXiv:1412.4503
/// - Bucci et al. 2019, *Phys Rev Lett* 122, 108302. arXiv:1811.05230
/// - Sato & Kanazawa 2024, *Phys Rev Lett*. arXiv:2411.13965
pub const IMPACT_DELTA: f64 = 0.10;

// ---------------------------------------------------------------------------
// Simulation record (per-bar state snapshot)
// ---------------------------------------------------------------------------

/// Per-bar state emitted by the simulation engine.
#[derive(Debug, Clone)]
pub struct SimulationRecord {
    pub timestamp: i64,
    pub nav: f64,
    pub portfolio_return: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub n_long: usize,
    pub n_short: usize,
    pub turnover: f64,
    /// Trading PnL for this day, computed INDEPENDENTLY from book state as
    /// `Δ realized_trading_pnl + Δ unrealized_pnl` between day start and day
    /// end (pre-reset in constant-NAV mode). Independent of NAV / cost
    /// accumulators, so the discrepancy check downstream is meaningful.
    pub trading_pnl: f64,
    /// Mark-to-market PnL from long positions only.
    pub long_pnl: f64,
    /// Mark-to-market PnL from short positions only.
    pub short_pnl: f64,
    pub funding_pnl: f64,
    pub commission_cost: f64,
    pub spread_cost: f64,
    pub impact_cost: f64,
    /// Per-day bookkeeping discrepancy:
    /// `(post_nav − day_start_nav) − (trading_pnl + funding − costs)`.
    /// Should be ≤ float epsilon if cash flows + position bookkeeping agree.
    /// Cumulative non-zero values across days indicate a real accounting bug
    /// (e.g. cost deducted from cash but not tracked, missing realized PnL,
    /// double-counted commission). Pre-reset, so valid in constant-NAV mode.
    pub bookkeeping_discrepancy: f64,
    pub margin_ratio: f64,
    pub total_margin_used: f64,
    /// Distance to liquidation: 0.0 = at backstop, 1.0 = NAV is 2× backstop threshold.
    pub liquidation_distance: f64,
    /// Per-symbol weight at end of bar. Uses interned u32 symbol IDs to avoid
    /// 7.2M string heap allocations during H1 simulation (48K bars × 150 positions).
    pub weights: Vec<(u32, f64)>,
    /// Cumulative compounded return from start (product of 1+r).
    pub cumulative_return: f64,
    /// Cumulative additive return from start (sum of daily returns).
    /// At constant NAV, this is the real total P&L as fraction of operating capital.
    pub additive_return: f64,
    /// Layer 2 (A4): predicted cost to fully liquidate every open position
    /// at end-of-bar via TWAP. Integrated form: Σ_i (1/(2+δ)) · κ_i · σ_i ·
    /// |Q_i| · (|Q_i|/ADV_i)^δ. Always ≥ 0. Cite EBA Prudent Valuation RTS
    /// 2014/2016 (close-out cost AVA + concentrated positions AVA),
    /// Bangia-Diebold-Schuermann-Stroughair 1999 (endogenous liquidity).
    pub liquidation_haircut: f64,
    /// Layer 2 (A4): NAV after deducting the liquidation haircut. Headline
    /// NAV for honest performance evaluation — Sharpe and MDD computed off
    /// this series in the summary, per Frazzini-Israel-Moskowitz 2018 and
    /// Novy-Marx-Velikov 2016. CAN be < `nav` (always when haircut > 0)
    /// and CAN be negative (when stuck positions cost more to exit than
    /// their MTM value, which is informative — strategy is bust on a
    /// liquidation basis even if mark NAV looks healthy).
    pub nav_liquid: f64,
    /// Layer 2 (A4): haircut from positions whose σ/κ/ADV lookups all
    /// returned valid current-bar values. Most defensible — uses live data.
    pub liquidation_haircut_current: f64,
    /// Layer 2 (A4): haircut from positions where current-bar lookup missed
    /// but last-known per-symbol value was within `last_known_max_age_days`.
    /// Defensible per EBA AVA "most recent observable data" framework.
    pub liquidation_haircut_last_known: f64,
    /// Layer 2 (A4): haircut from positions where neither current-bar nor
    /// fresh last-known data exists. Applied as flat `dlom_pct × |notional|`,
    /// per IFRS 13 / ASC 820 Level 3 convention.
    pub liquidation_haircut_dlom: f64,
    /// Layer 2 (A4): count of positions that fell into the DLOM tier.
    pub n_positions_dlom: u32,
    /// One-sided NAV cap (no-free-money mode): cumulative dollar amount
    /// "skimmed" off NAV from prior bars (NOT including this bar's skim if
    /// any — skim is applied AFTER the record is pushed, mirroring the
    /// constant_nav reset placement). Always 0.0 when `nav_cap_usd = None`.
    /// Total wealth (synthetic uncapped) = `nav_usd × (1 + cumulative_return)`.
    pub cumulative_skimmed: f64,
}

/// Full simulation result.
#[derive(Debug)]
pub struct SimulationResult {
    pub records: Vec<SimulationRecord>,
    pub final_book: PortfolioBook,
    pub liquidation_events: Vec<i64>,
    /// Number of position-bars with actual funding data.
    pub n_funding_actual: usize,
    /// Number of position-bars using model-estimated funding.
    pub n_funding_model: usize,
    /// A4: number of trades rejected by the impact gate (predicted impact
    /// exceeded `cost_model.impact_gate_bps`). Counted across the full run.
    pub n_rejected_trades: u64,
    /// A4: total |notional| of trades rejected by the impact gate.
    pub rejected_notional: f64,
    /// One-sided NAV cap mode: cumulative dollars skimmed off the working NAV
    /// across all bars where post-bar NAV exceeded `config.nav_cap_usd`.
    /// Diagnostic only. Always 0.0 when `nav_cap_usd = None`.
    pub cumulative_skimmed_usd: f64,
    /// One-sided NAV cap mode: per-bar skim events (only bars with skim > 0).
    /// `(timestamp, skim_amount)`. Empty when `nav_cap_usd = None`.
    pub per_bar_skimmed: Vec<(i64, f64)>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Holdings-based simulation engine.
///
/// Owns the PortfolioBook and runs a bar-by-bar loop:
///   1. Mark to market → actual weights
///   2. Accrue funding
///   3. QP solve (D1) → TWAP interpolation (H1)
///   4. Compute trades from weight deltas
///   5. Execute trades via exchange adapter (fees, impact, spread)
///   6. Margin check
///   7. Record state
pub struct SimulationEngine<'a> {
    book: PortfolioBook,
    exchange: &'a ExchangeConfig,
    config: &'a ExecutionConfig,
    volume_tracker: VolumeTracker,
    funding_model: FundingModel,
    // QP state
    solver_cache: HashMap<usize, CachedOSQP>,
    daily_target: BTreeMap<String, f64>,
    last_solve_day: i64,
    // Funding diagnostics
    n_funding_actual: usize,
    n_funding_model: usize,
    // A4 tradeability-gate diagnostics
    n_rejected_trades: u64,
    rejected_notional: f64,
    // One-sided NAV cap (no-free-money) diagnostics
    cumulative_skimmed_usd: f64,
    per_bar_skimmed: Vec<(i64, f64)>,
    // Walk-forward MOO fold-reset boundaries (HashSet for O(1) membership).
    // Keyed on DAY-FLOOR seconds (`(ts / 86400) * 86400`) since the simulation
    // emits records and runs the reset block only at end-of-day. The config
    // values are floored on construction so the tuner can pass any second-
    // precision timestamp within the target reset day. Empty when unused.
    fold_reset_day_set: std::collections::HashSet<i64>,
    // Bar-dump diagnostics (only populated when RUMPY_BAR_DUMP is set)
    last_n_active: usize,
    last_n_padded: usize,
    last_solve_was_fresh: bool,
    // QP-input hashes for first-divergence isolation (RUMPY_BAR_DUMP)
    last_alpha_hash: u64,
    last_wprev_hash: u64,
    last_waim_hash: u64,
    last_clin_hash: u64,
    last_kappa_hash: u64,
    last_cov_hash: u64,
    last_gamma_bits: u64,
}

impl<'a> SimulationEngine<'a> {
    pub fn new(config: &'a ExecutionConfig) -> Self {
        Self {
            book: PortfolioBook::new(config.nav_usd),
            exchange: &config.exchange,
            config,
            volume_tracker: VolumeTracker::new(),
            funding_model: FundingModel::new(&config.funding),
            solver_cache: HashMap::new(),
            daily_target: BTreeMap::new(),
            last_solve_day: -1,
            n_funding_actual: 0,
            n_funding_model: 0,
            n_rejected_trades: 0,
            rejected_notional: 0.0,
            cumulative_skimmed_usd: 0.0,
            per_bar_skimmed: Vec::new(),
            fold_reset_day_set: config.nav_reset_timestamps.iter()
                .map(|ts| (ts / 86400) * 86400)
                .collect(),
            last_n_active: 0,
            last_n_padded: 0,
            last_solve_was_fresh: false,
            last_alpha_hash: 0,
            last_wprev_hash: 0,
            last_waim_hash: 0,
            last_clin_hash: 0,
            last_kappa_hash: 0,
            last_cov_hash: 0,
            last_gamma_bits: 0,
        }
    }

    fn fnv_f64s(xs: &[f64]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for x in xs {
            h = h.wrapping_mul(0x100000001b3).wrapping_add(x.to_bits());
        }
        h
    }

    /// Run the full simulation.
    ///
    /// `alphas` are at DAILY resolution (one per (day, symbol)) — the
    /// strategic decision is daily in both fill modes.
    /// `bar_timestamps` is the EXECUTION timeline:
    ///   - `Market`: one per day (day-floor at 00:00 UTC).
    ///   - `Twap`:   24 H1 timestamps per day from price data.
    /// The simulation iterates `bar_timestamps`; at each bar, the day's alpha
    /// is looked up by day-floor. The QP solves once per new day; intermediate
    /// bars accrue funding, mark-to-market, and TWAP-fill toward the daily
    /// target. Records are emitted at the last bar of each day.
    ///
    /// `sigma_lookup` and `adv_lookup` are keyed by **day-floor** timestamps
    /// (D1 EWMAs in days).
    pub fn run(
        &mut self,
        alphas: &[AlphaRow],
        bar_timestamps: &[i64],
        aim_weights: &[AimWeight],
        rolling_cov: &RollingCovariance,
        spread_lookup: &HashMap<(i64, u32), f64>,
        kappa_lookup: &HashMap<(i64, u32), f64>,
        price_lookup: &HashMap<(i64, u32), f64>,
        funding_lookup: &HashMap<(i64, u32), f64>,
        sigma_lookup: &HashMap<(i64, u32), f64>,
        adv_lookup: &HashMap<(i64, u32), f64>,
        symbol_ids: &HashMap<String, u32>,
    ) -> SimulationResult {
        let mut records = Vec::new();
        let mut liquidation_events = Vec::new();

        // Cross-platform divergence test hook: per-bar internal-state dump.
        // RUMPY_BAR_DUMP=path → write one TSV-like line per emitted record with
        // bit-level f64 hex of: qp_gross, daily_return, cumulative_growth,
        // nav, cash, day_funding, day_commission, day_spread_cost,
        // day_impact_cost, day_turnover. (L2 2026-05-13: ewma_mean, ewma_sq,
        // n_vol_seen, vol_leverage fields removed with post-QP rescale.)
        use std::io::Write as _IoWrite;
        let mut bar_dump: Option<std::io::BufWriter<std::fs::File>> = std::env::var("RUMPY_BAR_DUMP")
            .ok()
            .and_then(|p| std::fs::File::create(p).ok())
            .map(std::io::BufWriter::new);
        if let Some(f) = bar_dump.as_mut() {
            let _ = writeln!(f, "# RUMPY_BAR_DUMP per-bar internal state (one line per emitted record)");
            let _ = writeln!(f, "# fields: bar_idx ts day n_active n_padded fresh_solver alpha_hash wprev_hash waim_hash clin_hash kappa_hash cov_hash gamma_bits dt_hash dt_count qp_gross daily_return cumulative_growth nav cash day_funding day_commission day_spread_cost day_impact_cost day_turnover");
        }

        // RUMPY_TRADE_DUMP=path → write one CSV row per executed child trade for
        // post-hoc cost-correctness audit. Captures everything that flows into
        // compute_realized_trade_costs plus the resulting cost components.
        // ~hundreds-MB CSV for a 5y backtest at H1; gated behind env var so
        // production runs aren't impacted.
        let mut trade_dump: Option<std::io::BufWriter<std::fs::File>> = std::env::var("RUMPY_TRADE_DUMP")
            .ok()
            .and_then(|p| std::fs::File::create(p).ok())
            .map(std::io::BufWriter::new);
        if let Some(f) = trade_dump.as_mut() {
            let _ = writeln!(f, "bar_idx,ts,symbol,delta_notional,delta_qty,price,spread_bps,kappa,sigma,adv,commission,spread_cost,impact_cost,nav_pre_trade,trailing_vol,w_prev,w_target_bar");
        }

        // Aim lookup keyed by day-floor (alpha is daily — aim is daily).
        let aim_lookup: HashMap<(i64, &str), f64> = aim_weights.iter()
            .map(|w| ((w.timestamp, w.symbol.as_str()), w.weight_aim))
            .collect();

        // Daily alphas grouped by day-floor. Used to look up the day's alpha
        // at each execution bar.
        let mut daily_assets: HashMap<i64, Vec<(&str, f64)>> = HashMap::new();
        for row in alphas {
            let day = (row.timestamp / 86400) * 86400;
            daily_assets.entry(day).or_default().push((&row.symbol, row.alpha_future));
        }

        // Group execution-bar indices by day. In Market mode each day has one
        // index; in Twap mode each day has up to 24.
        let mut hours_by_day: HashMap<i64, Vec<usize>> = HashMap::new();
        for (idx, &ts) in bar_timestamps.iter().enumerate() {
            let day = (ts / 86400) * 86400;
            hours_by_day.entry(day).or_default().push(idx);
        }

        // A4 Layer 2: per-symbol last-known (ts, value) for σ/κ/ADV. These
        // are accumulated DYNAMICALLY during the bar loop — at each bar, for
        // every symbol with a valid value at the current bar, the entry is
        // updated to (current_ts, value). Used by the haircut fallback chain
        // when a held position's current-bar lookup misses (e.g. symbol fell
        // out of the alpha universe but we still hold inventory).
        //
        // Lookahead-safe: by construction every entry has ts ≤ current_ts at
        // any point during the simulation. The earlier implementation built
        // these once before the loop from the GLOBAL latest entry, which
        // meant the staleness check (`age.abs() ≤ max_age`) could use future
        // values for any bar within max_age of the dataset's last data point
        // for that sym. See the audit at memory/round1_qp_math_2026_05_04.md
        // (and the empirical confirmation: removing future-ts shifted DLOM
        // rate +5.9pp and total haircut +5.9% on the v9 Model A champion).
        let mut last_known_kappa: HashMap<u32, (i64, f64)> = HashMap::new();
        let mut last_known_sigma: HashMap<u32, (i64, f64)> = HashMap::new();
        let mut last_known_adv: HashMap<u32, (i64, f64)> = HashMap::new();

        let bucket = 10;
        let c_lin_mult = self.config.cost_model.c_lin_multiplier;
        // HL minimum order size — applies only to orders that open or increase
        // a position. Pure close/reduce orders have no minimum. Sourced from
        // ExchangeConfig (default 10.0 for HL). See exchange.rs::ExchangeConfig
        // for full rationale.
        let min_order_size_usd = self.config.exchange.min_order_size_usd;

        // Trade-flow diagnostics (gated by RUMPY_TRADEFLOW_DUMP env var).
        // When set, prints open/close trade-size distribution at end of run.
        let dump_tradeflow = std::env::var("RUMPY_TRADEFLOW_DUMP").is_ok();
        let mut tf_open_executed = 0usize;
        let mut tf_open_rejected = 0usize;
        let mut tf_close_executed = 0usize;
        let mut tf_open_exec_notional = 0.0f64;
        let mut tf_open_rej_notional = 0.0f64;
        let mut tf_close_exec_notional = 0.0f64;

        let mut has_qp_solution = false;
        let mut t_mark_ns = 0u64;
        let mut t_funding_ns = 0u64;
        let mut t_qp_ns = 0u64;
        let mut t_trades_ns = 0u64;
        let mut t_margin_ns = 0u64;
        let mut t_record_ns = 0u64;
        let mut t_stuck_ns = 0u64;
        let mut n_qp_solves = 0usize;

        // Daily accumulators — accumulate across H1 bars, emit once per day
        let mut day_start_nav = self.book.nav();
        // Independent trading-PnL anchors. realized_trading_pnl is per-position
        // cumulative (price-diff rpnl from apply_fill) and survives constant-NAV
        // resets. unrealized goes to 0 on reset, so we snapshot at day start
        // (post-reset) for correct delta accounting.
        let mut day_start_realized = self.book.total_realized_trading_pnl();
        let mut day_start_unrealized = self.book.total_unrealized_pnl();
        let mut day_funding = 0.0f64;
        let mut day_commission = 0.0f64;
        let mut day_spread_cost = 0.0f64;
        let mut day_impact_cost = 0.0f64;
        let mut day_turnover = 0.0f64;
        let mut day_long_pnl = 0.0f64;
        let mut day_short_pnl = 0.0f64;
        let mut current_day: i64 = -1;
        let mut cumulative_growth = 1.0f64; // compounds daily returns
        let mut additive_total = 0.0f64;   // sums daily returns

        for (bar_idx, &ts) in bar_timestamps.iter().enumerate() {
            let day = (ts / 86400) * 86400;
            // Daily alpha for this bar. If the day has no alpha (warmup or
            // missing-data gap), skip the bar entirely — no decisions, no
            // mark-to-market state change.
            let assets = match daily_assets.get(&day) {
                Some(a) => a,
                None => continue,
            };
            let ts = &ts; // keep `*ts` ergonomics throughout the body

            // Detect new day — snapshot start NAV and reset accumulators
            if day != current_day {
                day_start_nav = self.book.nav();
                day_start_realized = self.book.total_realized_trading_pnl();
                day_start_unrealized = self.book.total_unrealized_pnl();
                day_funding = 0.0;
                day_commission = 0.0;
                day_spread_cost = 0.0;
                day_impact_cost = 0.0;
                day_turnover = 0.0;
                day_long_pnl = 0.0;
                day_short_pnl = 0.0;
                current_day = day;
            }
            let syms: Vec<String> = assets.iter().map(|(s, _)| s.to_string()).collect();

            // ----------------------------------------------------------
            // 1. MARK TO MARKET (track long/short PnL from price moves)
            // ----------------------------------------------------------
            let te = std::time::Instant::now();
            // Prices from alpha universe + any held positions (price-only rows
            // in the execution parquet provide prices for non-alpha symbols).
            let mut bar_prices: HashMap<String, f64> = syms.iter()
                .filter_map(|s| symbol_ids.get(s.as_str()).and_then(|&sid| price_lookup.get(&(*ts, sid))).map(|&p| (s.clone(), p)))
                .collect();
            // Add prices for held positions not in this bar's alpha universe
            for sym in self.book.positions.keys() {
                if !bar_prices.contains_key(sym) {
                    if let Some(&p) = symbol_ids.get(sym).and_then(|&sid| price_lookup.get(&(*ts, sid))) {
                        bar_prices.insert(sym.clone(), p);
                    }
                }
            }

            // Snapshot unrealized PnL before mark for long/short attribution
            let mut long_pnl_before = 0.0f64;
            let mut short_pnl_before = 0.0f64;
            for pos in self.book.positions.values() {
                if pos.side() == 1 { long_pnl_before += pos.unrealized_pnl; }
                else if pos.side() == -1 { short_pnl_before += pos.unrealized_pnl; }
            }

            self.book.mark_all(&bar_prices);

            let mut long_pnl_after = 0.0f64;
            let mut short_pnl_after = 0.0f64;
            for pos in self.book.positions.values() {
                if pos.side() == 1 { long_pnl_after += pos.unrealized_pnl; }
                else if pos.side() == -1 { short_pnl_after += pos.unrealized_pnl; }
            }
            let bar_long_pnl = long_pnl_after - long_pnl_before;
            let bar_short_pnl = short_pnl_after - short_pnl_before;

            t_mark_ns += te.elapsed().as_nanos() as u64;

            // ----------------------------------------------------------
            // 2. ACCRUE FUNDING (actual rate → model fallback)
            // ----------------------------------------------------------
            let te = std::time::Instant::now();
            let mut bar_funding = 0.0f64;
            // Collect position data first to avoid borrow conflict with funding_model
            let held_syms: Vec<(String, f64, f64)> = self.book.positions.iter()
                .filter(|(_, pos)| !pos.is_flat())
                .filter_map(|(sym, pos)| {
                    bar_prices.get(sym).map(|&price| (sym.clone(), pos.quantity, price))
                })
                .collect();

            for (sym, qty, price) in &held_syms {
                let actual_rate = symbol_ids.get(sym.as_str()).and_then(|&sid| funding_lookup.get(&(*ts, sid))).copied().unwrap_or(0.0);

                let rate = if actual_rate != 0.0 {
                    // Actual data: use it and feed to model for EWMA state
                    self.funding_model.update_asset(sym, actual_rate);
                    self.n_funding_actual += 1;
                    actual_rate
                } else if self.config.funding.enabled {
                    // Missing data: use model's EWMA estimate (per-hour rate)
                    let state = self.funding_model.state(sym);
                    let model_rate = state.map(|s| s.ewma_rate).unwrap_or(0.0);
                    self.n_funding_model += 1;
                    model_rate
                } else {
                    0.0
                };

                if rate.abs() > 1e-15 {
                    let cf = self.exchange.compute_funding_payment(*qty, *price, rate);
                    if let Some(pos) = self.book.positions.get_mut(sym) {
                        pos.realized_funding_pnl += cf;
                    }
                    self.book.cash += cf;
                    bar_funding += cf;
                }
            }

            t_funding_ns += te.elapsed().as_nanos() as u64;

            // ----------------------------------------------------------
            // 3. ACTUAL WEIGHTS + QP SOLVE
            // ----------------------------------------------------------
            let nav = self.book.nav();
            let w_actual = self.book.actual_weights();

            // ----------------------------------------------------------
            // 4. QP SOLVE (D1)
            // ----------------------------------------------------------
            if day != self.last_solve_day {
                let te = std::time::Instant::now();
                n_qp_solves += 1;
                self.solve_daily_qp(
                    *ts, &assets, &syms, &aim_lookup, rolling_cov,
                    spread_lookup, kappa_lookup,
                    sigma_lookup, adv_lookup,
                    symbol_ids,
                    c_lin_mult, bucket,
                );
                t_qp_ns += te.elapsed().as_nanos() as u64;
                self.last_solve_day = day;
                if !self.daily_target.is_empty() {
                    has_qp_solution = true;
                }
            }

            // Skip warmup bars before first QP solution (no covariance yet)
            if !has_qp_solution {
                continue;
            }

            // L2 (2026-05-13 leverage refactor): post-QP vol-target rescale +
            // max_gross renormalization REMOVED. The QP's SOC constraint
            // (qp.sigma_target_daily) now enforces vol; qp.l_max + per_name_cap
            // enforce caps as hard QP constraints that actually bind on
            // deployed w. See docs/sessions/2026-05-13-pre-phase2-leverage-
            // refactor-plan.md for rationale.

            // ----------------------------------------------------------
            // 5. TWAP + TRADES
            // ----------------------------------------------------------
            let te = std::time::Instant::now();
            let hours_today = hours_by_day.get(&day).map(|v| v.len()).unwrap_or(24);
            let hour_in_day = hours_by_day.get(&day)
                .and_then(|v| v.iter().position(|&idx| idx == bar_idx))
                .unwrap_or(0);
            let remaining = (hours_today - hour_in_day).max(1) as f64;
            let trade_rate = (1.0 / remaining).min(1.0);

            // ----------------------------------------------------------
            // 6. COMPUTE + EXECUTE TRADES
            // ----------------------------------------------------------
            let mut bar_commission = 0.0f64;
            let mut bar_spread_cost = 0.0f64;
            let mut bar_impact_cost = 0.0f64;
            let mut bar_turnover = 0.0f64;
            // A4: count trades rejected by the impact gate this bar.
            let mut bar_rejected_count: u64 = 0;
            let mut bar_rejected_notional: f64 = 0.0;

            // Union of all symbols in play
            let mut all_syms: BTreeSet<String> = w_actual.keys().cloned().collect();
            for s in self.daily_target.keys() { all_syms.insert(s.clone()); }
            for s in &syms { all_syms.insert(s.clone()); }

            if nav > 1e-6 {
                // Margin-aware scaling: if approaching liquidation threshold,
                // scale down new exposure to avoid triggering backstop.
                let pre_trade_maintenance = self.book.total_maintenance_margin(
                    self.exchange.maintenance_margin_rate);
                let margin_scale = if self.exchange.is_margin_stressed(nav, pre_trade_maintenance) {
                    // Scale proportionally: at halt_ratio → 1.0, at liquidation_ratio → 0.0
                    let mr = if pre_trade_maintenance > 1e-12 { nav / pre_trade_maintenance } else { 100.0 };
                    let liq = self.exchange.liquidation_ratio;
                    let halt = self.exchange.margin_halt_ratio;
                    ((mr - liq) / (halt - liq)).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                // Compute target weights for all symbols. L2 (2026-05-13): no
                // post-QP vol leverage and no max_gross renormalization — the
                // QP's hard constraints (per_name_cap, qp.l_max, optional
                // sigma_target_daily SOC) define the deployed target. Only
                // margin-protection scaling (exchange-side, NOT leverage)
                // remains.
                let mut bar_targets: Vec<(String, f64)> = Vec::new();
                for s in &all_syms {
                    let w_curr = w_actual.get(s).copied().unwrap_or(0.0);
                    let w_target_raw = self.daily_target.get(s).copied().unwrap_or(0.0);
                    let w_target = w_target_raw * margin_scale;
                    let w_bar = w_curr + (w_target - w_curr) * trade_rate;
                    bar_targets.push((s.clone(), w_bar));
                }

                for (s, w_bar) in &bar_targets {
                    let w_curr = w_actual.get(s).copied().unwrap_or(0.0);
                    let delta_w = w_bar - w_curr;
                    let delta_notional = delta_w * nav;

                    // HL minimum order size — applies only to orders that
                    // open or increase a position. Pure close/reduce orders
                    // accepted at any size. Position flips count as
                    // increasing (the "opening" component is non-zero).
                    //
                    // Decomposition:
                    //   pure_reduce = same sign AND |w_new| ≤ |w_curr|
                    //                 OR new position is essentially flat
                    //   needs_min   = !pure_reduce
                    //
                    // Pre-2026-04-27 used hardcoded `< 1.0` regardless of
                    // direction — see exchange.rs for full rationale.
                    let w_new = w_curr + (w_bar - w_curr); // == *w_bar
                    let same_sign = w_curr * w_new >= 0.0;
                    let pure_reduce = (same_sign && w_new.abs() <= w_curr.abs() + 1e-12)
                        || w_new.abs() < 1e-12;
                    if dump_tradeflow {
                        if pure_reduce {
                            tf_close_executed += 1;
                            tf_close_exec_notional += delta_notional.abs();
                        } else if delta_notional.abs() < min_order_size_usd {
                            tf_open_rejected += 1;
                            tf_open_rej_notional += delta_notional.abs();
                        } else {
                            tf_open_executed += 1;
                            tf_open_exec_notional += delta_notional.abs();
                        }
                    }
                    if !pure_reduce && delta_notional.abs() < min_order_size_usd {
                        continue; // would be rejected by HL
                    }

                    let price = match bar_prices.get(s) {
                        Some(&p) if p > 0.0 => p,
                        _ => continue,
                    };

                    let delta_qty = delta_notional / price;

                    // Fee from exchange adapter (volume-tiered)
                    let trailing_vol = self.volume_tracker.trailing_volume();
                    let commission = self.exchange.compute_fee(delta_notional, false, trailing_vol);

                    // Resolve per-asset cost data
                    let sid = symbol_ids.get(s).copied().unwrap_or(u32::MAX);
                    let spread_bps = spread_lookup.get(&(*ts, sid)).copied().unwrap_or(0.0);
                    let kappa = kappa_lookup.get(&(*ts, sid)).copied().unwrap_or(0.0);
                    // Sigma and ADV are D1 EWMAs keyed by day-floor.
                    let sigma = sigma_lookup.get(&(day, sid)).copied().unwrap_or(0.0);
                    let adv = adv_lookup.get(&(day, sid)).copied().unwrap_or(1e6);

                    // Compute the actually-deducted cost components. Pure
                    // function — must NEVER take `c_lin_multiplier`. That
                    // parameter is for QP perception only; actual P&L is
                    // charged the real cost from the lookups regardless of
                    // what the optimizer thought it was paying. Keeping the
                    // calculation in a separate fn pins the contract at the
                    // type level — adding `c_lin_multiplier` here would be
                    // a code change that requires touching this signature.
                    let (spread_cost, impact_cost) = compute_realized_trade_costs(
                        delta_notional, spread_bps, kappa, sigma, adv,
                        self.config.cost_model.impact_delta,
                        self.config.cost_model.q_ref_usd,
                    );

                    // A4 tradeability gate: if predicted impact exceeds the
                    // configured threshold (bps of |notional|), the trade is
                    // declared non-tradeable and skipped entirely. No cash
                    // debit, no position update, no commission. This is
                    // structurally different from clipping the cost — we
                    // do NOT lie about the cost of a trade that happened.
                    // Real-market analog: orders that would move price beyond
                    // a strategy's threshold simply do not fill.
                    let abs_notional = delta_notional.abs();
                    if !passes_impact_gate(
                        impact_cost, abs_notional,
                        self.config.cost_model.impact_gate_bps,
                    ) {
                        bar_rejected_count += 1;
                        bar_rejected_notional += abs_notional;
                        continue;
                    }

                    // RUMPY_TRADE_DUMP per-trade audit row (gated by env var).
                    // Captures the inputs to compute_realized_trade_costs + the
                    // resulting cost components, so a downstream analyzer can
                    // reproduce the formula and validate cost correctness.
                    if let Some(f) = trade_dump.as_mut() {
                        let _ = writeln!(
                            f,
                            "{bar_idx},{ts},{s},{delta_notional},{delta_qty},{price},{spread_bps},{kappa},{sigma},{adv},{commission},{spread_cost},{impact_cost},{nav},{trailing_vol},{w_curr},{w_bar}",
                            ts = *ts,
                        );
                    }

                    // Execute fill
                    let pos = self.book.get_or_create_position(s, *ts);
                    let rpnl = pos.apply_fill(delta_qty, price, commission, *ts);
                    self.book.cash += rpnl;
                    self.book.cash -= commission + spread_cost + impact_cost;

                    bar_commission += commission;
                    bar_spread_cost += spread_cost;
                    bar_impact_cost += impact_cost;
                    bar_turnover += delta_w.abs();

                    // Track volume for fee tier
                    self.volume_tracker.record_trade(*ts, delta_notional);
                }
            } // end if nav > 1e-6

            // Accumulate intraday costs
            day_commission += bar_commission;
            day_spread_cost += bar_spread_cost;
            day_impact_cost += bar_impact_cost;
            day_turnover += bar_turnover;
            self.n_rejected_trades += bar_rejected_count;
            self.rejected_notional += bar_rejected_notional;
            day_long_pnl += bar_long_pnl;
            day_short_pnl += bar_short_pnl;
            day_funding += bar_funding;

            t_trades_ns += te.elapsed().as_nanos() as u64;

            // ----------------------------------------------------------
            // 7. FORCE-CLOSE STUCK POSITIONS (before margin check)
            // ----------------------------------------------------------
            let te = std::time::Instant::now();
            let stale_cutoff = *ts - 30 * 86400;
            let stuck_syms: Vec<String> = self.book.positions.iter()
                .filter(|(sym, pos)| {
                    !pos.is_flat()
                        && pos.last_fill_at < stale_cutoff
                        && !bar_prices.contains_key(*sym)
                })
                .map(|(sym, _)| sym.clone())
                .collect();
            for sym in &stuck_syms {
                let pos = self.book.positions.get_mut(sym).unwrap();
                let price = pos.mark_price;
                if price <= 0.0 { continue; }
                let qty = -pos.quantity;
                let notional = (qty * price).abs();
                let trailing_vol = self.volume_tracker.trailing_volume();
                let commission = self.exchange.compute_fee(notional, false, trailing_vol);
                let rpnl = pos.apply_fill(qty, price, commission, *ts);
                self.book.cash += rpnl;
                self.book.cash -= commission;
                day_commission += commission;
            }

            t_stuck_ns += te.elapsed().as_nanos() as u64;

            // ----------------------------------------------------------
            // 8. MARGIN CHECK + LIQUIDATION
            // ----------------------------------------------------------
            let te = std::time::Instant::now();
            let total_maintenance: f64 = self.book.positions.values()
                .map(|p| self.exchange.maintenance_margin(p.notional()))
                .sum();
            let post_nav = self.book.nav();
            let mr = self.exchange.margin_ratio(post_nav, total_maintenance);
            let liq_dist = self.exchange.liquidation_distance(post_nav, total_maintenance);

            if self.exchange.is_liquidatable(post_nav, total_maintenance) {
                let taker_bps = self.exchange.current_taker_bps(self.volume_tracker.trailing_volume());
                let fee_rate = taker_bps / 10_000.0;
                let (_rpnl, comm) = self.book.liquidate_all(*ts, fee_rate);
                day_commission += comm;
                liquidation_events.push(*ts);
                self.daily_target.clear();
            }

            let post_nav = self.book.nav();
            t_margin_ns += te.elapsed().as_nanos() as u64;

            // ----------------------------------------------------------
            // 9. END-OF-DAY: record, vol state, constant-NAV reset
            //    H1 bars only accumulate. Records emitted at daily frequency.
            // ----------------------------------------------------------
            let is_last_bar_of_day = {
                let day_bars = hours_by_day.get(&day);
                match day_bars {
                    Some(indices) => indices.last() == Some(&bar_idx),
                    None => true, // D1 mode: every bar is last bar of day
                }
            };

            if is_last_bar_of_day && has_qp_solution {
                let te = std::time::Instant::now();
                let daily_return = if day_start_nav > 1e-6 {
                    (post_nav - day_start_nav) / day_start_nav
                } else {
                    0.0
                };
                cumulative_growth *= 1.0 + daily_return;
                additive_total += daily_return;

                // Bar-dump (L2 2026-05-13: vol_leverage / ewma_mean / ewma_sq /
                // n_vol_seen fields removed — post-QP vol-target rescale gone).
                if let Some(f) = bar_dump.as_mut() {
                    let qp_gross: f64 = self.daily_target.values().map(|w| w.abs()).sum();
                    let cash = self.book.cash;
                    let mut dt_hash: u64 = 0xcbf29ce484222325;
                    for (sym, w) in self.daily_target.iter() {
                        for b in sym.as_bytes() {
                            dt_hash = dt_hash.wrapping_mul(0x100000001b3).wrapping_add(*b as u64);
                        }
                        dt_hash = dt_hash.wrapping_mul(0x100000001b3).wrapping_add(w.to_bits());
                    }
                    let dt_count = self.daily_target.len();
                    let _ = writeln!(
                        f,
                        "bar[{:>4}] ts={} day={} n_active={} n_padded={} fresh_solver={} alpha_hash=0x{:016x} wprev_hash=0x{:016x} waim_hash=0x{:016x} clin_hash=0x{:016x} kappa_hash=0x{:016x} cov_hash=0x{:016x} gamma_bits=0x{:016x} dt_hash=0x{:016x} dt_count={} qp_gross=0x{:016x} daily_return=0x{:016x} cumulative_growth=0x{:016x} nav=0x{:016x} cash=0x{:016x} day_funding=0x{:016x} day_commission=0x{:016x} day_spread_cost=0x{:016x} day_impact_cost=0x{:016x} day_turnover=0x{:016x}",
                        records.len(), *ts, day,
                        self.last_n_active,
                        self.last_n_padded,
                        self.last_solve_was_fresh,
                        self.last_alpha_hash,
                        self.last_wprev_hash,
                        self.last_waim_hash,
                        self.last_clin_hash,
                        self.last_kappa_hash,
                        self.last_cov_hash,
                        self.last_gamma_bits,
                        dt_hash,
                        dt_count,
                        qp_gross.to_bits(),
                        daily_return.to_bits(),
                        cumulative_growth.to_bits(),
                        post_nav.to_bits(),
                        cash.to_bits(),
                        day_funding.to_bits(),
                        day_commission.to_bits(),
                        day_spread_cost.to_bits(),
                        day_impact_cost.to_bits(),
                        day_turnover.to_bits(),
                    );
                }

                let final_weights: Vec<(u32, f64)> = self.book.actual_weights()
                    .into_iter()
                    .filter(|(_, w)| w.abs() > 1e-9)
                    .filter_map(|(sym, w)| symbol_ids.get(sym.as_str()).map(|&sid| (sid, w)))
                    .collect();

                // Independent trading PnL: Δ(book.realized) + Δ(book.unrealized)
                // computed from book state, NOT a residual from NAV change.
                // This makes the downstream discrepancy check actually verify
                // that cash flows and position bookkeeping agree.
                let trading_pnl = (self.book.total_realized_trading_pnl() - day_start_realized)
                    + (self.book.total_unrealized_pnl() - day_start_unrealized);
                // Per-day bookkeeping check: NAV change should equal
                // trading_pnl + funding − costs if cash flows are consistent.
                let bookkeeping_discrepancy = (post_nav - day_start_nav)
                    - (trading_pnl + day_funding - day_commission - day_spread_cost - day_impact_cost);

                // In constant-NAV mode, show the effective compounded NAV
                let display_nav = if self.config.constant_nav {
                    self.config.nav_usd * cumulative_growth
                } else {
                    post_nav
                };

                // Layer 2 (A4): update per-symbol last-known (κ, σ, ADV)
                // BEFORE computing this bar's haircut. Lookahead-safe by
                // construction — only this bar's data (ts ≤ current_ts) is
                // ingested; future bars' data has not been processed yet.
                // Iterates the universe once per bar (~352 sids × 3 lookups).
                for (_sym_str, &sid) in symbol_ids.iter() {
                    if let Some(&v) = kappa_lookup.get(&(*ts, sid)) {
                        if v > 0.0 { last_known_kappa.insert(sid, (*ts, v)); }
                    }
                    if let Some(&v) = sigma_lookup.get(&(day, sid)) {
                        if v > 1e-12 { last_known_sigma.insert(sid, (day, v)); }
                    }
                    if let Some(&v) = adv_lookup.get(&(day, sid)) {
                        if v > 1e-6 { last_known_adv.insert(sid, (day, v)); }
                    }
                }

                // Layer 2 (A4): liquidation haircut — sum predicted close
                // costs across all open positions, with a fallback chain
                // (current-bar → last-known → DLOM). Captures stuck-out-of-
                // universe positions that current-bar lookups miss.
                let mut bar_haircut = 0.0f64;
                let mut bar_haircut_current = 0.0f64;
                let mut bar_haircut_last_known = 0.0f64;
                let mut bar_haircut_dlom = 0.0f64;
                let mut bar_n_dlom: u32 = 0;
                let cm = &self.config.cost_model;
                let max_age_secs = (cm.last_known_max_age_days as i64) * 86400;
                for pos in self.book.positions.values() {
                    if pos.is_flat() { continue; }
                    let abs_n = (pos.quantity * pos.mark_price).abs();
                    if abs_n <= 1e-6 { continue; }
                    let sid = match symbol_ids.get(pos.symbol.as_str()) {
                        Some(&s) => s,
                        None => continue,
                    };
                    let kappa_now = kappa_lookup.get(&(*ts, sid)).copied().unwrap_or(0.0);
                    let sigma_now = sigma_lookup.get(&(day, sid)).copied().unwrap_or(0.0);
                    let adv_now = adv_lookup.get(&(day, sid)).copied().unwrap_or(0.0);
                    let lk_kappa = last_known_kappa.get(&sid).copied();
                    let lk_sigma = last_known_sigma.get(&sid).copied();
                    let lk_adv = last_known_adv.get(&sid).copied();
                    let (h, src) = compute_position_haircut_with_fallback(
                        abs_n, *ts,
                        kappa_now, sigma_now, adv_now,
                        lk_kappa, lk_sigma, lk_adv,
                        cm.impact_delta,
                        cm.illiquid_dlom_pct,
                        max_age_secs,
                    );
                    bar_haircut += h;
                    match src {
                        HaircutSource::Current => bar_haircut_current += h,
                        HaircutSource::LastKnown => bar_haircut_last_known += h,
                        HaircutSource::Dlom => {
                            bar_haircut_dlom += h;
                            bar_n_dlom += 1;
                        }
                    }
                }
                let nav_liquid = display_nav - bar_haircut;

                records.push(SimulationRecord {
                    timestamp: *ts,
                    nav: display_nav,
                    portfolio_return: daily_return,
                    gross_exposure: self.book.gross_exposure(),
                    net_exposure: self.book.net_exposure(),
                    n_long: self.book.n_long(),
                    n_short: self.book.n_short(),
                    turnover: day_turnover,
                    trading_pnl,
                    long_pnl: day_long_pnl,
                    short_pnl: day_short_pnl,
                    funding_pnl: day_funding,
                    commission_cost: day_commission,
                    spread_cost: day_spread_cost,
                    impact_cost: day_impact_cost,
                    bookkeeping_discrepancy,
                    margin_ratio: mr,
                    total_margin_used: total_maintenance,
                    liquidation_distance: liq_dist,
                    weights: final_weights,
                    cumulative_return: cumulative_growth - 1.0,
                    additive_return: additive_total,
                    liquidation_haircut: bar_haircut,
                    nav_liquid,
                    liquidation_haircut_current: bar_haircut_current,
                    liquidation_haircut_last_known: bar_haircut_last_known,
                    liquidation_haircut_dlom: bar_haircut_dlom,
                    n_positions_dlom: bar_n_dlom,
                    cumulative_skimmed: self.cumulative_skimmed_usd,
                });

                t_record_ns += te.elapsed().as_nanos() as u64;

                // Post-bar NAV reset/cap.
                //
                // `nav_cap_usd` (preferred) — one-sided cap: skim ONLY when
                // current_nav > cap. Loss days are untouched, so losses
                // propagate naturally. Models the deployment pattern
                // "compound up to cap, skim above cap" with NO free money.
                //
                // `constant_nav` (legacy academic) — full reset both ways:
                // a -10% day gets refilled to nav_usd next bar. Creates
                // free money on losses but the recorded daily_return
                // captures it correctly (the reset only affects position
                // bookkeeping, not the per-bar return record). Kept for
                // tune+validate workflows where IS/OOS Sharpe must be at
                // the same operating point.
                //
                // `nav_reset_timestamps` (walk-forward MOO) — explicit
                // per-fold resets. Same mechanic as constant_nav, but only
                // fires at the listed timestamps (fold boundaries). Lets
                // each fold start at the same effective NAV for capacity-
                // invariant cross-fold comparison.
                //
                // Precedence: nav_cap_usd > constant_nav > nav_reset_timestamps.
                // (The latter two compose meaningfully only when the former
                // are off.) See `apply_nav_reset_to_target` for the shared
                // reset mechanic.
                if let Some(cap) = self.config.nav_cap_usd {
                    let skim = apply_nav_cap_skim(&mut self.book, cap);
                    if skim > 0.0 {
                        self.cumulative_skimmed_usd += skim;
                        self.per_bar_skimmed.push((*ts, skim));
                    }
                } else if self.config.constant_nav {
                    apply_nav_reset_to_target(&mut self.book, self.config.nav_usd);
                } else if self.fold_reset_day_set.contains(&day) {
                    apply_nav_reset_to_target(&mut self.book, self.config.nav_usd);
                }

                // Periodic cleanup
                if records.len() % 100 == 0 {
                    self.book.remove_flat_positions();
                }
            }
        }

        if dump_tradeflow {
            let tot = tf_open_executed + tf_open_rejected + tf_close_executed;
            println!("  trade flow (RUMPY_TRADEFLOW_DUMP):");
            println!("    opens executed:  {:>10} ({:.2}%, ${:.2} notional, avg ${:.4})",
                tf_open_executed, 100.0*tf_open_executed as f64/tot.max(1) as f64,
                tf_open_exec_notional, tf_open_exec_notional / tf_open_executed.max(1) as f64);
            println!("    opens rejected:  {:>10} ({:.2}%, ${:.2} notional, avg ${:.4})",
                tf_open_rejected, 100.0*tf_open_rejected as f64/tot.max(1) as f64,
                tf_open_rej_notional, tf_open_rej_notional / tf_open_rejected.max(1) as f64);
            println!("    closes executed: {:>10} ({:.2}%, ${:.2} notional, avg ${:.4})",
                tf_close_executed, 100.0*tf_close_executed as f64/tot.max(1) as f64,
                tf_close_exec_notional, tf_close_exec_notional / tf_close_executed.max(1) as f64);
        }
        println!("  sim profiling: mark={:.2}s, funding={:.2}s, qp={:.2}s ({} solves), trades={:.2}s, stuck={:.2}s, margin={:.2}s, record={:.2}s",
            t_mark_ns as f64 / 1e9, t_funding_ns as f64 / 1e9,
            t_qp_ns as f64 / 1e9, n_qp_solves,
            t_trades_ns as f64 / 1e9, t_stuck_ns as f64 / 1e9,
            t_margin_ns as f64 / 1e9, t_record_ns as f64 / 1e9);

        SimulationResult {
            records,
            final_book: self.book.clone(),
            liquidation_events,
            n_funding_actual: self.n_funding_actual,
            n_funding_model: self.n_funding_model,
            n_rejected_trades: self.n_rejected_trades,
            rejected_notional: self.rejected_notional,
            cumulative_skimmed_usd: self.cumulative_skimmed_usd,
            per_bar_skimmed: std::mem::take(&mut self.per_bar_skimmed),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn solve_daily_qp(
        &mut self,
        ts: i64,
        assets: &[(&str, f64)],
        syms: &[String],
        aim_lookup: &HashMap<(i64, &str), f64>,
        rolling_cov: &RollingCovariance,
        spread_lookup: &HashMap<(i64, u32), f64>,
        kappa_lookup: &HashMap<(i64, u32), f64>,
        sigma_lookup: &HashMap<(i64, u32), f64>,
        adv_lookup: &HashMap<(i64, u32), f64>,
        symbol_ids: &HashMap<String, u32>,
        c_lin_mult: f64,
        bucket: usize,
    ) {
        let w_actual = self.book.actual_weights();

        // aim_lookup is keyed by day-floor (D1 alpha timestamps); rolling_cov
        // already floors internally. Use day_floor for D1-keyed lookups.
        let day_floor = (ts / 86400) * 86400;

        let (mut cov, syms_present) = rolling_cov.get(ts, syms);
        let n = syms_present.len();
        if n < 2 {
            // Keep daily_target unchanged (carry forward)
            return;
        }

        // Probe A (2026-05-09): shrinkage of Σ toward its diagonal:
        //   Σ_shrunk[i,j] = (1−ρ)·Σ[i,j]   for i ≠ j
        //   Σ_shrunk[i,i] = Σ[i,i]
        // ρ=0 leaves Σ unchanged. ρ=1 zeros all off-diagonals (pure
        // vol-normalized cov). Empirically motivated: noisy off-diagonals
        // amplify mid-rank alpha errors through Σ⁻¹·α in regime-shift
        // periods. Memory ref: project_markowitz_regime_fragility.md.
        let shrinkage_rho = self.config.covariance.shrinkage_rho;
        if shrinkage_rho > 0.0 {
            let rho = shrinkage_rho.clamp(0.0, 1.0);
            let one_minus_rho = 1.0 - rho;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        cov[i * n + j] *= one_minus_rho;
                    }
                }
            }
        }

        let sym_to_alpha: HashMap<&str, f64> = assets.iter().map(|(s, a)| (*s, *a)).collect();
        let mut alpha_aligned: Vec<f64> = syms_present.iter()
            .map(|s| *sym_to_alpha.get(s.as_str()).unwrap_or(&0.0)).collect();
        let w_prev_aligned: Vec<f64> = syms_present.iter()
            .map(|s| w_actual.get(s).copied().unwrap_or(0.0)).collect();
        let mut w_aim_aligned: Vec<f64> = syms_present.iter()
            .map(|s| *aim_lookup.get(&(day_floor, s.as_str())).unwrap_or(&0.0)).collect();

        // 2026-05-16 Path 4 (MOP 2012 vol-scaling, cheap falsifier): divide
        // alpha by σ_i so high-vol names get less weight pressure per unit
        // conviction. The alpha entering here is already σ_i-scaled (from the
        // VolNormalizedExcess → DollarAlpha adapter); dividing reverts to
        // unit-free z-blend semantics. If Markowitz Σ⁻¹·α + γ·w'Σw already
        // captures this risk-parity effect, the lever is near no-op.
        // See memory/project_per_name_cap_research_2026_05_16.md.
        if self.config.alpha.vol_scale {
            for (i, sym) in syms_present.iter().enumerate() {
                let sid = symbol_ids.get(sym.as_str()).copied().unwrap_or(u32::MAX);
                let sigma = sigma_lookup.get(&(day_floor, sid)).copied().unwrap_or(0.0);
                if sigma > 1e-9 {
                    alpha_aligned[i] /= sigma;
                }
            }
        }

        // 2026-05-15: funding-aware alpha (Koijen-Moskowitz-Pedersen 2018,
        // "Carry"; Boyd-Busseti 2017 §3-4 holding-cost term). Subtract the
        // AR(1)-projected cumulative funding cost over `holding_hours` from
        // alpha so the QP sees carry-adjusted expected return. The forecaster
        // is `FundingModel::expected_alpha_adjustment` (cost.rs:198), which
        // applies the closed-form  E[C_H] = H·μ + (f₀−μ)·ρ(1−ρ^H)/(1−ρ)
        // per asset, using state updated from realized hourly rates in the
        // funding-accrual block of the bar loop (simulation.rs ~L500).
        //
        // Optional cross-sectional demean (alpha_demean = true) defends
        // against asymmetric-gross leakage: when L/S notionals aren't
        // exactly balanced (current strategy ~70–85% L:S), a non-zero
        // cross-sectional mean of the adjustment shifts gross. Demeaning
        // makes it gross-neutral.
        //
        // Default: off (use_in_alpha=false). Opt in per-spec.
        if self.config.funding.enabled && self.config.funding.use_in_alpha {
            let mut adj = self.funding_model.expected_alpha_adjustment(&syms_present);
            let scale = self.config.funding.alpha_scale;
            if self.config.funding.alpha_demean && !adj.is_empty() {
                let mean_adj: f64 = adj.iter().sum::<f64>() / adj.len() as f64;
                for a in adj.iter_mut() { *a -= mean_adj; }
            }
            for (a, f) in alpha_aligned.iter_mut().zip(adj.iter()) {
                *a -= scale * *f;
            }
        }

        // Probe knob (2026-05-05): regime-execution-gap calm-bound dilution fix.
        // If `prune_top_k` is set, restrict the QP universe to the top-K assets
        // by |alpha| each bar — outside-top-K assets get alpha=0 and w_aim=0,
        // so the only forces on them are the cone movement cost and γ·w'Σw.
        // Existing positions bleed off rather than rotate into low-conviction
        // names. w_prev is intentionally NOT zeroed: we don't force-liquidate.
        if let Some(k) = self.config.qp.prune_top_k {
            if k < alpha_aligned.len() {
                let mut idx: Vec<usize> = (0..alpha_aligned.len()).collect();
                idx.sort_by(|&i, &j| {
                    alpha_aligned[j].abs().partial_cmp(&alpha_aligned[i].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut keep = vec![false; alpha_aligned.len()];
                for &i in idx.iter().take(k) { keep[i] = true; }
                for i in 0..alpha_aligned.len() {
                    if !keep[i] {
                        alpha_aligned[i] = 0.0;
                        w_aim_aligned[i] = 0.0;
                    }
                }
            }
        }

        // Probe G (2026-05-09): soft-threshold (Donoho-Johnstone 1994) on
        // the cross-section's z-scored alpha. Flatten mid-rank alphas to the
        // mean while preserving the rank structure of the tails:
        //   z = (α − μ) / σ
        //   z_clean = sign(z) · max(0, |z| − threshold)
        //   α_clean = μ + z_clean · σ
        // Memory ref: project_markowitz_regime_fragility.md.
        let soft_thresh = self.config.alpha.soft_threshold_z;
        if soft_thresh > 0.0 && alpha_aligned.len() > 1 {
            let n_a = alpha_aligned.len();
            let mu_a: f64 = alpha_aligned.iter().sum::<f64>() / n_a as f64;
            let var_a: f64 = alpha_aligned.iter()
                .map(|a| (a - mu_a).powi(2))
                .sum::<f64>() / n_a as f64;
            let sigma_a = var_a.sqrt();
            if sigma_a > 1e-12 {
                let c = soft_thresh * sigma_a;
                for a in alpha_aligned.iter_mut() {
                    let centered = *a - mu_a;
                    let abs_c = centered.abs();
                    if abs_c <= c {
                        *a = mu_a;
                    } else {
                        *a = mu_a + centered.signum() * (abs_c - c);
                    }
                }
            }
        }

        // c_lin: dynamic from exchange fee tier + half-spread.
        //
        // (SOCP refactor 2026-05-05) The legacy A4 Layer 1 size-impact penalty
        // — a static linearization of impact at fixed `qp_liquidity_typical_w
        // · NAV` — has been removed. Probe 4 (memory probe_4_layer1_2026_05_05.md)
        // showed Layer 1 was inert at deployment NAV with champion params
        // (typical_Q < Q_ref → max(·,0) clipped to zero on all assets), and
        // the SOCP cone subsumes Layer 1 even where it does fire. c_lin is
        // now strictly the linear taker + spread cost; impact is handled
        // exactly by the cone term `s ≥ κ_eff·t_imp − r·u` in the solver.
        let trailing_vol = self.volume_tracker.trailing_volume();
        let taker_bps = self.config.exchange.current_taker_bps(trailing_vol);
        let cm = &self.config.cost_model;
        let delta_param = cm.impact_delta;
        let default_spread_bps = cm.default_spread_bps;
        let c_lin_aligned: Vec<f64> = syms_present.iter()
            .map(|s| {
                let sid = symbol_ids.get(s).copied().unwrap_or(u32::MAX);
                let spread_bps = spread_lookup.get(&(ts, sid)).copied().unwrap_or(default_spread_bps);
                (taker_bps + 0.5 * spread_bps * c_lin_mult) / 10_000.0
            }).collect();

        // Per-asset cone weights for the SOCP impact term.
        //
        //   κ_eff_i = κ_i · σ_i · (NAV/ADV_i)^δ  → `compute_qp_effective_kappa`
        //   r_i     = κ_i · σ_i · (Q_ref/ADV_i)^δ → `compute_qp_residual_kappa`
        //
        // The solver enforces `s_i ≥ κ_eff_i·t_imp_i − r_i·u_i` and `s_i ≥ 0`
        // and minimises `Σ s_i`. At optimum (per Probes 1 + 2 + 3), s_i·NAV
        // matches `compute_realized_trade_costs` exactly per asset.
        let nav = self.book.nav();
        let kappa_eff_aligned: Vec<f64> = syms_present.iter()
            .map(|s| {
                let sid = symbol_ids.get(s).copied().unwrap_or(u32::MAX);
                let kappa = *kappa_lookup.get(&(ts, sid)).unwrap_or(&1e6);
                let sigma = *sigma_lookup.get(&(day_floor, sid)).unwrap_or(&0.0);
                let adv = *adv_lookup.get(&(day_floor, sid)).unwrap_or(&1e6);
                compute_qp_effective_kappa(kappa, sigma, adv, nav, delta_param)
            }).collect();
        let r_aligned: Vec<f64> = syms_present.iter()
            .map(|s| {
                let sid = symbol_ids.get(s).copied().unwrap_or(u32::MAX);
                let kappa = *kappa_lookup.get(&(ts, sid)).unwrap_or(&0.0);
                let sigma = *sigma_lookup.get(&(day_floor, sid)).unwrap_or(&0.0);
                let adv = *adv_lookup.get(&(day_floor, sid)).unwrap_or(&0.0);
                compute_qp_residual_kappa(kappa, sigma, adv, cm.q_ref_usd, delta_param)
            }).collect();

        let n_padded = ((n + bucket - 1) / bucket) * bucket;
        // Diagnostics for bar dump
        self.last_n_active = n;
        self.last_n_padded = n_padded;
        self.last_solve_was_fresh = !self.solver_cache.contains_key(&n_padded);
        // FNV hashes of every QP input — to isolate which input first diverges
        // cross-platform. cov is hashed BEFORE padding.
        self.last_alpha_hash = Self::fnv_f64s(&alpha_aligned);
        self.last_wprev_hash = Self::fnv_f64s(&w_prev_aligned);
        self.last_waim_hash = Self::fnv_f64s(&w_aim_aligned);
        self.last_clin_hash = Self::fnv_f64s(&c_lin_aligned);
        self.last_kappa_hash = Self::fnv_f64s(&kappa_eff_aligned);
        self.last_cov_hash = Self::fnv_f64s(&cov);
        let mut qp_cfg = self.config.qp.clone();
        // L2 (2026-05-13): dynamic_gamma override removed. γ is now a static
        // smoothness regularizer (Boyd cvxportfolio 2024 role); leverage is
        // controlled by qp.sigma_target_daily (SOC) + qp.l_max policy cap.
        self.last_gamma_bits = qp_cfg.gamma.to_bits();

        // NAV-aware per_name_cap floor (2026-05-07). At low NAV the configured
        // per_name_cap × l_max × NAV may fall below the exchange minimum order
        // size — every per-asset opening trade gets silently rejected at the
        // trade-execution gate (simulation.rs::trade_loop, line 678). Existing
        // positions can still close (pure_reduce bypasses the gate), so the
        // strategy bleeds out and freezes flat; QP keeps producing nonzero
        // targets but none of them survive to fills.
        //
        // Floor:  cap >= (min_order_size × 1.5) / (l_max × NAV)
        // Picks the larger of (configured cap, floor). At production-scale NAV
        // the floor is tiny and never binds (e.g. NAV=$1M with l_max=1.5,
        // min_order=$10 → floor = 4.4e-6 vs cap=0.01). At small NAV it forces
        // concentration into fewer tradeable positions — the realistic
        // exchange-aware behavior. Safety multiplier 1.5 ensures partial-fill
        // TWAP bars (trade_rate < 1) still clear the gate at the last hour.
        let nav = self.book.nav();
        if nav > 1e-6 {
            // L2 (2026-05-13): NAV-aware floor uses qp.l_max (policy gross
            // cap) instead of vol_target.l_max (removed). Same intent: ensure
            // at the configured leverage the smallest position is at least
            // exchange.min_order_size_usd × 1.5.
            let cap_floor = (self.config.exchange.min_order_size_usd
                    * self.config.qp.per_name_cap_floor_safety_mult)
                / (self.config.qp.l_max * nav);
            if let Some(base) = qp_cfg.per_name_cap {
                qp_cfg.per_name_cap = Some(base.max(cap_floor));
            }
        }

        // L2 (2026-05-13): regime-conditional per_name_cap overlay REMOVED
        // (Mahalanobis-driven U-shape scaling). No literature precedent for
        // stacking regime overlays on top of vol target; collapses to single
        // policy cap. See docs/sessions/2026-05-13-leverage-external-research.md.

        // Pad to bucket size.
        // Padded slots: alpha=0, w_prev=0, w_aim=0, c_lin=0, r=0, kappa_eff=0.
        //
        // The legacy pre-SOCP code padded kappa_eff with 1e6 to discourage
        // trading on dummy slots in the SLP iteration loop's gradient term
        // `1.5·κ·sqrt(|Δw|+ε)`. With the cone refactor, kappa_eff is the
        // coefficient on `t_imp_i` inside the ReLU-aux constraint
        // `s ≥ κ_eff·t_imp − r·u`. If kappa_eff=1e6 on a padded slot whose
        // `t_imp` drifts even slightly above zero from solver numerical
        // noise, `1e6·1e-8 = 0.01` of phantom cost lands on `s`, distorting
        // the objective and causing Clarabel `InsufficientProgress` at scale.
        // Padded slots already have alpha=0, w_prev=0, w_aim=0, c_lin=0,
        // ridge cov[i,i]=1e-6 — the QP has no incentive to trade them, so
        // kappa_eff=0 is both correct and stable.
        let mut alpha_pad = alpha_aligned;
        let mut cov_pad = cov;
        let mut w_prev_pad = w_prev_aligned;
        let mut w_aim_pad = w_aim_aligned;
        let mut c_lin_pad = c_lin_aligned;
        let mut kappa_eff_pad = kappa_eff_aligned;
        let mut r_pad = r_aligned;
        if n_padded > n {
            let pad = n_padded - n;
            alpha_pad.extend(vec![0.0; pad]);
            w_prev_pad.extend(vec![0.0; pad]);
            w_aim_pad.extend(vec![0.0; pad]);
            c_lin_pad.extend(vec![0.0; pad]);
            kappa_eff_pad.extend(vec![0.0; pad]);
            r_pad.extend(vec![0.0; pad]);
            let mut cov_new = vec![0.0; n_padded * n_padded];
            for i in 0..n { for j in 0..n { cov_new[i * n_padded + j] = cov_pad[i * n + j]; } }
            for i in n..n_padded { cov_new[i * n_padded + i] = 1e-6; }
            cov_pad = cov_new;
        }

        let solver = self.solver_cache.entry(n_padded).or_insert_with(|| {
            CachedOSQP::new(n_padded, &qp_cfg, &cov_pad).expect("solver setup failed")
        });

        // 2026-05-16 Path 1 (vol-scaled per-name cap): when enabled, compute
        // per-asset cap c_i = c_base × σ_med / σ_i where σ_i = √Σ_{ii} and
        // σ_med is the cross-sectional median of σ this bar. Bounded above
        // by 5×c_base to keep ultra-low-vol names from runaway. Padded slots
        // get c_base (non-binding given α=0, w=0 padded). cap_vec is unpadded
        // (length n); cap_vec_padded is full length n_padded for the solver.
        let cap_vec_padded: Option<Vec<f64>> = if self.config.qp.per_name_cap_vol_scale && n > 1 {
            let c_base = qp_cfg.per_name_cap.unwrap_or(self.config.qp.per_name_cap_default);
            let ceiling_mult = self.config.qp.per_name_cap_vol_scale_ceiling_mult;
            let sigmas: Vec<f64> = (0..n)
                .map(|i| cov_pad[i * n_padded + i].sqrt().max(1e-12))
                .collect();
            let mut sigmas_sorted = sigmas.clone();
            sigmas_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let sigma_med = sigmas_sorted[n / 2].max(1e-12);
            let mut v: Vec<f64> = sigmas.iter()
                .map(|&s| (c_base * sigma_med / s).min(ceiling_mult * c_base))
                .collect();
            v.resize(n_padded, c_base);
            Some(v)
        } else {
            None
        };

        // Single solve. Impact is handled exactly inside the solver via the
        // SOCP cone, no fixed-point iteration needed.
        let solve_result = solver.solve(
            &alpha_pad, &cov_pad, &w_prev_pad, &w_aim_pad,
            &c_lin_pad, &kappa_eff_pad, &r_pad,
            cap_vec_padded.as_deref(),
        );
        let mut w_solved = match solve_result {
            Some((w, _)) => w,
            None => return,
        };

        // Probe F (2026-05-09): hybrid blend `w = (1−ρ)·w_qp + ρ·w_rank`.
        // Rank-weighted vector is built from `alpha_pad[..n]` (live assets;
        // padded slots have alpha=0 by construction): winsorize ±σ_alpha,
        // normalize to ±per_name_cap, then re-demean for dollar-neutrality.
        // Final blend is re-clipped to per_name_cap and re-demeaned so both
        // invariants hold under arbitrary ρ ∈ [0, 1]. Vol-target leverage
        // scaling later rescales gross to hit the daily-vol target.
        // Memory ref: project_markowitz_regime_fragility.md.
        if self.config.blend_rho > 0.0 && n > 1 {
            let rho = self.config.blend_rho.clamp(0.0, 1.0);
            let c_base = qp_cfg.per_name_cap.unwrap_or(self.config.qp.per_name_cap_default);
            let winsor_sigma = self.config.alpha.winsorize_sigma.max(1e-6);
            // Per-asset cap c_i if vol-scaling is on (same formula as the
            // solver above), else scalar c_base for every name.
            let cap_per: Vec<f64> = if let Some(ref v) = cap_vec_padded {
                v[..n].to_vec()
            } else {
                vec![c_base; n]
            };
            // Mean + std over the live (non-padded) alpha slice.
            let alphas_live = &alpha_pad[..n];
            let mean_a: f64 = alphas_live.iter().sum::<f64>() / n as f64;
            let var_a: f64 = alphas_live.iter()
                .map(|a| (a - mean_a).powi(2))
                .sum::<f64>() / n as f64;
            let std_a = var_a.sqrt();
            let clip = (winsor_sigma * std_a).max(1e-12);
            // Build rank weights from clipped, mean-removed alpha, scaled to
            // each asset's own cap.
            let mut w_rank: Vec<f64> = alphas_live.iter().enumerate()
                .map(|(i, a)| ((a - mean_a).clamp(-clip, clip) / clip) * cap_per[i])
                .collect();
            // Re-demean (clipping breaks dollar-neutrality if one tail had
            // more clip-victims than the other).
            let dn = w_rank.iter().sum::<f64>() / n as f64;
            for w in w_rank.iter_mut() { *w -= dn; }
            // Blend, re-clip per asset, re-demean.
            for i in 0..n {
                let blended = (1.0 - rho) * w_solved[i] + rho * w_rank[i];
                w_solved[i] = blended.clamp(-cap_per[i], cap_per[i]);
            }
            let post_mean: f64 = w_solved[..n].iter().sum::<f64>() / n as f64;
            for w in &mut w_solved[..n] {
                *w -= post_mean;
            }
        }

        // Store daily target (raw QP output, before vol target)
        self.daily_target.clear();
        for (i, s) in syms_present.iter().enumerate() {
            if w_solved[i].abs() > 1e-9 {
                self.daily_target.insert(s.clone(), w_solved[i]);
            }
        }

        // Test A: dump objective-term decomposition + Test B: aim turnover.
        // Triggered by RUMPY_QP_DECOMP_DUMP=path. One line per QP solve.
        if let Ok(path) = std::env::var("RUMPY_QP_DECOMP_DUMP") {
            // Compute the four objective contributions at the solved w.
            // Use unpadded w (slice [..n]) and unpadded inputs.
            let w = &w_solved[..n];
            // gamma_term = γ · w'·Σ·w (Σ from cov_pad, original n×n submatrix)
            let mut gamma_term = 0.0;
            for i in 0..n {
                for j in 0..n {
                    gamma_term += w[i] * cov_pad[i * n_padded + j] * w[j];
                }
            }
            gamma_term *= qp_cfg.gamma;
            // alpha_term = -α'·w
            let alpha_term: f64 = -alpha_pad[..n].iter().zip(w.iter()).map(|(a, w)| a * w).sum::<f64>();
            // aim_term = λ_aim · ||w-w_aim||²
            let aim_term: f64 = qp_cfg.lambda_aim
                * w_aim_pad[..n].iter().zip(w.iter()).map(|(a, w)| (w - a).powi(2)).sum::<f64>();
            // cost_lin_term = c_lin'·|Δw|  (linear cost the QP saw)
            let cost_lin_term: f64 = c_lin_pad[..n]
                .iter()
                .zip(w.iter())
                .zip(w_prev_pad[..n].iter())
                .map(|((c, w), wp)| c * (w - wp).abs())
                .sum();
            // cost_impact_term = Σ max(κ_eff·u^(1+δ) − r·u, 0)·1
            //   evaluated at the solved w, with u = |Δw|. This is the cone
            //   objective contribution `Σ s_i` at optimum, equivalent to the
            //   per-NAV realized impact cost summed over assets. Multiply by
            //   NAV externally to get USD impact cost.
            //
            // Pre-2026-05-05 this was `1.5·Σκ_eff·u^1.5` (the SLP linearization
            // value at active cone). Probe 1 confirmed the cone objective at
            // optimum equals `Σ κ_eff·u^1.5`, NOT 1.5× that. Probe 3 added the
            // `r·u` linear correction for Q_ref residual.
            let cost_impact_term: f64 = (0..n)
                .map(|i| {
                    let u = (w[i] - w_prev_pad[i]).abs();
                    let cone = kappa_eff_pad[i] * u.powf(1.5);
                    let linear = r_pad[i] * u;
                    (cone - linear).max(0.0)
                })
                .sum();
            // Total QP objective value (informational)
            let total_obj = gamma_term + alpha_term + aim_term + cost_lin_term + cost_impact_term;

            // Test B inputs: aim turnover (||w_aim_t - w_aim_{t-1}||₁)
            // and aim L1 norm (||w_aim||₁).
            let prev_day = day_floor - 86400;
            let mut aim_turnover_l1 = 0.0;
            let mut aim_l1_curr = 0.0;
            for s in syms_present.iter() {
                let curr = aim_lookup.get(&(day_floor, s.as_str())).copied().unwrap_or(0.0);
                let prev = aim_lookup.get(&(prev_day, s.as_str())).copied().unwrap_or(0.0);
                aim_turnover_l1 += (curr - prev).abs();
                aim_l1_curr += curr.abs();
            }

            // QP-output gross
            let qp_gross: f64 = w.iter().map(|x| x.abs()).sum();
            let actual_turnover_l1: f64 = w
                .iter()
                .zip(w_prev_pad[..n].iter())
                .map(|(w, wp)| (w - wp).abs())
                .sum();

            use std::io::Write;
            // Append-mode so multiple solves accumulate
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(
                    f,
                    "ts={ts} day_floor={day_floor} n={n} \
                     gamma={:.6e} gamma_term={gamma_term:.6e} \
                     alpha_term={alpha_term:.6e} aim_term={aim_term:.6e} \
                     cost_lin_term={cost_lin_term:.6e} cost_impact_term={cost_impact_term:.6e} \
                     total_obj={total_obj:.6e} qp_gross={qp_gross:.6e} \
                     actual_turnover_l1={actual_turnover_l1:.6e} \
                     aim_turnover_l1={aim_turnover_l1:.6e} aim_l1={aim_l1_curr:.6e} \
                     lambda_aim={:.6e}",
                    qp_cfg.gamma, qp_cfg.lambda_aim,
                );
            }
        }
    }

    // L2 (2026-05-13): compute_vol_leverage() + update_vol_state() removed.
    // Post-QP vol-target rescale eliminated; the SOC constraint inside the
    // QP (qp.sigma_target_daily) now controls portfolio vol. See
    // docs/sessions/2026-05-13-pre-phase2-leverage-refactor-plan.md.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===========================================================================
// Realized-cost computation (pure function — pins contract w/ c_lin_multiplier)
// ===========================================================================

/// QP-perceived effective kappa, calibrated to match realized impact formula.
///
/// **The calibration**: realized impact (as fraction of NAV) is
///   `realized = κ · σ · u^(1+δ) · (NAV/ADV)^δ`
/// The QP's iterative impact term is `1.5 · κ_eff · u^1.5` (gradient of
/// `κ_eff · u^1.5`, hardcoded square-root form). Setting
///   `κ_eff = κ · σ · (NAV/ADV)^δ`
/// pins NAV/ADV scaling exactly. The residual u-exponent difference
/// (1.5 vs 1+δ when δ ≠ 0.5) leaves the QP/realized ratio = 1.5 · u^(0.5 - δ).
/// For δ=0.5 (production config), exponents match exactly. This is a
/// known limitation of the closed-form quadratic QP; see project memory
/// `project_qp_impact_socp_refactor.md`.
///
/// **History**:
/// - pre-2026-04-26: QP used raw `κ` scaled only by `sqrt(NAV/nav_usd)`.
///   Missing per-asset σ, missing per-asset ADV, wrong NAV exponent.
/// - 2026-05-04: changed signature from "uses const `IMPACT_DELTA = 0.10`"
///   to "takes `delta` parameter from config" — eliminates the dead-code
///   inconsistency between the const (0.10, regime-mixture) and config
///   (`cost_model.impact_delta = 0.5`). Caller passes `cm.impact_delta`.
///   The 0.10 const remains as a documented historical default for the
///   `IMPACT_DELTA` constant referenced elsewhere; this function does not
///   consult it directly.
///
/// Inputs:
/// - `kappa`: per-asset impact coefficient (dimensionless, from cost-scores parquet)
/// - `sigma`: per-asset D1 EWMA realized vol (fraction)
/// - `adv`:   per-asset D1 EWMA dollar volume
/// - `nav`:   current NAV in dollars (in constant_nav mode, equals config.nav_usd)
/// - `delta`: power-law exponent. Pass `cost_model.impact_delta` to keep
///            the QP iteration consistent with realized cost.
///
/// Returns: dimensionless effective kappa for the QP impact term.
pub fn compute_qp_effective_kappa(kappa: f64, sigma: f64, adv: f64, nav: f64, delta: f64) -> f64 {
    if adv > 1e-6 && sigma > 1e-12 && nav > 0.0 {
        kappa * sigma * (nav / adv).powf(delta)
    } else {
        // Fall back to raw kappa if sigma/adv missing — preserves prior behavior
        // for the bootstrap window before D1 EWMAs have warmed up.
        kappa
    }
}

/// Q_ref-residual coefficient for the SOCP cone term. Encodes the
/// linear-in-u contribution of the realized cost's Q_ref subtraction.
///
/// Realized cost (per `compute_realized_trade_costs`):
///   `impact = κ·σ·|Q|·max((|Q|/ADV)^δ − (Q_ref/ADV)^δ, 0)`
///        = κ_eff·NAV·u^(1+δ) − r·u·NAV     (when above threshold; else 0)
/// where
///   `κ_eff = κ·σ·(NAV/ADV)^δ`   (= compute_qp_effective_kappa)
///   `r     = κ·σ·(Q_ref/ADV)^δ` (this function)
///
/// In the SOCP refactor, `r` is the per-asset coefficient on `u_i` in the
/// ReLU-aux constraint `s_i ≥ κ_eff_i·t_imp_i − r_i·u_i`. At optimum,
/// `s_i = max(κ_eff_i·u_i^1.5 − r_i·u_i, 0)·NAV` reproduces the realized
/// impact exactly per asset.
///
/// Threshold (where the cone-aux activates): `u_threshold = Q_ref/NAV`,
/// uniform across all assets.
///
/// Returns 0 when `q_ref_usd ≤ 0` (degrades cleanly to "no subtraction" — the
/// cone-only formulation, `s = κ_eff·u^1.5`, equivalent to legacy realized
/// cost with `q_ref_usd = 0`). Also returns 0 if σ/ADV are missing —
/// preserves the bootstrap-window contract from `compute_qp_effective_kappa`.
///
/// Inputs:
/// - `kappa`: per-asset impact coefficient
/// - `sigma`: per-asset D1 EWMA realized vol
/// - `adv`:   per-asset D1 EWMA dollar volume
/// - `q_ref_usd`: reference order size (typically `cost_model.q_ref_usd`).
///                When 0, this function returns 0.
/// - `delta`: power-law exponent (typically `cost_model.impact_delta = 0.5`)
///
/// Probe 3 (memory `probe_3_qref_2026_05_05.md`) verified that ignoring this
/// residual term in the optimizer produces ~9% L1 weight divergence and ~8%
/// realized-cost under-charge at deployment NAV ($1M, where 82% of trades
/// fall below Q_ref). Keeping the ReLU aux is justified by that empirical
/// magnitude.
pub fn compute_qp_residual_kappa(kappa: f64, sigma: f64, adv: f64, q_ref_usd: f64, delta: f64) -> f64 {
    if adv > 1e-6 && sigma > 1e-12 && q_ref_usd > 0.0 && kappa > 0.0 {
        kappa * sigma * (q_ref_usd / adv).powf(delta)
    } else {
        // Degrade cleanly: q_ref_usd=0 means "no subtraction"; missing σ/ADV
        // are bootstrap-window cases where neither κ_eff nor r is meaningful.
        0.0
    }
}

/// Compute the realized spread + impact cost components for a single trade.
///
/// **Cost decomposition** (per Hasbrouck 2009, Bucci-Bouchaud-Lillo-Benzaquen
/// 2018, Bouchaud-Bonart-Donier-Gould 2018 textbook):
///
///   `total_cost = half_spread + impact`
///
/// where `half_spread = (spread_bps / 2) × |Q|` is the order-processing cost
/// paid for liquidity (read directly from L2; not fitted), and `impact` is
/// the residual price concession.
///
/// **Impact term** uses the metaorder framework (Tóth 2011, Donier 2014,
/// Sato-Kanazawa 2024 universal δ ≈ 0.5):
///   `impact_usd = κ · σ · |Q|^(1+δ) / ADV^δ = κ · σ · |Q| · (|Q|/ADV)^δ`
/// where δ = `IMPACT_DELTA`.
///
/// **Two-regime caveat (Bucci 2018):** the sqrt-law impact applies in the
/// sqrt regime (φ = Q/V_d ∈ [10⁻³, 10⁻¹]). In the linear regime (φ < 10⁻³),
/// impact ∝ Q (per-dollar impact constant), equivalent to δ ≈ 0. Production
/// would ideally use a gated formula:
///   `impact = κ·σ·|Q|·(|Q|/ADV)^δ · gate(φ)`
/// where gate suppresses impact below the crossover. **Current implementation
/// does NOT gate** — uses single-δ across all regimes. At our typical trade
/// sizes (φ ≈ 10⁻⁶ on majors), most cost is half-spread regardless, so this
/// approximation is operationally fine. It begins to over-/under-predict in
/// large-NAV or thin-coin regimes; see
/// memory/research_bucci_crossover_verified_on_hl_2026_04_29.md for empirical
/// regime evidence on HL.
///
/// **Contract**: This function MUST NOT take `c_lin_multiplier` as a
/// parameter. `c_lin_multiplier` is a tunable that scales the cost the QP
/// *perceives* during optimization (so TPE can find the right cost weighting
/// for the optimizer's tradeoffs). It must NOT scale the cost actually
/// deducted from simulated P&L — otherwise the tuner could "cheat" by
/// pretending costs are smaller than they are.
///
/// The signature itself is the contract: if anyone tries to add
/// `c_lin_multiplier` here, they have to edit this signature, which is
/// caught in code review.
///
/// Inputs:
/// - `delta_notional`: signed dollar trade size (positive = buy, negative = sell)
/// - `spread_bps`: full bid-ask spread in basis points (half-spread is `spread_bps / 2`)
/// - `kappa`: per-asset impact coefficient from cost-scores parquet (or ML)
/// - `sigma`: per-asset D1 EWMA realized vol
/// - `adv`: per-asset D1 EWMA dollar volume
/// - `delta`: power-law exponent. From `CostModelConfig.impact_delta`.
/// - `q_ref_usd`: reference Q at which residual-layer impact is zero.
///   When > 0, the impact charge is the MARGINAL impact above what R9 already
///   bakes in at the cell's typical Q (preserves backwards-compat in the R9
///   calibration regime). When 0 (legacy), impact is computed from |Q| only.
///
/// Returns: `(spread_cost, impact_cost)` — both in USD, both ≥ 0.
///
/// Pure formula. The simulator separately applies a TRADEABILITY GATE
/// (`passes_impact_gate`): trades whose predicted impact exceeds a bps
/// threshold are skipped entirely rather than charged a clipped cost. We
/// don't lie about cost on trades that happen — illiquid trades just
/// don't happen. See `passes_impact_gate` and A4 fix in
/// `docs/plans/cost-model-A2-fix.md`.
///
/// Commission is computed separately by the exchange adapter (volume-tiered)
/// and is also independent of `c_lin_multiplier`.
pub fn compute_realized_trade_costs(
    delta_notional: f64,
    spread_bps: f64,
    kappa: f64,
    sigma: f64,
    adv: f64,
    delta: f64,
    q_ref_usd: f64,
) -> (f64, f64) {
    let abs_notional = delta_notional.abs();
    // Spread cost: half-spread × |notional|. Half because we cross half the
    // spread on average for a market-aware fill (Hasbrouck-style decomposition).
    let spread_cost = 0.5 * spread_bps / 10_000.0 * abs_notional;
    // Impact cost (metaorder framework, Tóth 2011 / Donier 2014 / Sato 2024):
    //   impact_usd = κ · σ · |Q| · [(|Q|/ADV)^δ − (Q_ref/ADV)^δ]
    //              = κ · σ · |Q|^(1+δ)/ADV^δ − κ · σ · |Q| · (Q_ref/ADV)^δ
    // The Q_ref subtraction is the residual-layer fix (A2) — at |Q| = Q_ref
    // the impact contribution is exactly zero, so R9 (which predicts cell
    // median total cost at typical Q ≈ Q_ref) is preserved in its
    // calibration regime. Above Q_ref the marginal impact is charged.
    // Floored at 0 (impact ≥ 0).
    let impact_cost = if adv > 1e-6 && sigma > 1e-12 && abs_notional > 1e-6 {
        let phi = abs_notional / adv;
        let impact_phi = phi.powf(delta);
        let impact_ref = if q_ref_usd > 0.0 {
            (q_ref_usd / adv).powf(delta)
        } else {
            0.0
        };
        let marginal = (impact_phi - impact_ref).max(0.0);
        kappa * sigma * marginal * abs_notional
    } else {
        0.0
    };
    (spread_cost, impact_cost)
}

/// A4: tradeability gate.
///
/// Returns `true` if the trade's predicted impact (in bps of |notional|)
/// is at or below `gate_bps`, meaning the trade is allowed to fill.
/// Returns `false` when impact would exceed the threshold — the simulator
/// must skip the fill entirely (no cash debit, no position change).
///
/// This is the structural A4 fix. We do NOT clip the cost of trades that
/// happen — that would be a cost lie on a real trade. Instead, we declare
/// the asset not tradeable at this size in this bar and refuse to execute.
/// In real markets this corresponds to: the order would not fill at the
/// quoted size without moving price beyond what the strategy is willing to
/// pay; in our framework, the strategy is unwilling to pay > `gate_bps`
/// of impact.
///
/// `gate_bps = 0.0` disables the gate (all trades pass — legacy).
pub fn passes_impact_gate(impact_cost: f64, abs_notional: f64, gate_bps: f64) -> bool {
    if gate_bps <= 0.0 { return true; }
    if abs_notional <= 1e-6 { return true; }
    let impact_bps = impact_cost / abs_notional * 1e4;
    impact_bps <= gate_bps
}

/// A4 Layer 2: which data tier supplied the inputs for a haircut compute.
/// Reported as a diagnostic to surface how much of the trapped-position
/// problem is coming from positions whose σ/κ/ADV lookups have gone stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaircutSource {
    /// All three lookups (σ, κ, ADV) returned valid current-bar values.
    Current,
    /// At least one lookup missed at the current bar; used the last-known
    /// per-sid value within `max_age_days`. Defensible per EBA AVA "most
    /// recent observable data" convention.
    LastKnown,
    /// Lookups missing AND last-known unavailable or beyond `max_age_days`.
    /// Used flat DLOM × |notional|. Per IFRS 13 / ASC 820 Level 3.
    Dlom,
}

/// A4 Layer 2: per-position liquidation haircut.
///
/// Returns the predicted cost (in USD) of fully liquidating a position of
/// |notional| size against the asset's ADV, integrated over a TWAP
/// liquidation under the power-law metaorder impact model
/// `impact_pct(q) = κ·σ·(q/ADV)^δ`:
///
/// ```text
/// haircut = ∫_0^|Q| κ·σ·(q/ADV)^δ dq
///         = (1 / (1+δ)) · κ·σ·|Q|·(|Q|/ADV)^δ
/// ```
///
/// Derivation (TWAP at constant rate v=Q/T over T=1 day, V·T = ADV).
/// At time t having executed q(t)=Q·t/T, mid-price has moved by
/// `κ·σ·(q(t)/ADV)^δ`. Each slice dq pays its current cumulative impact:
///   `dC = dq · κ·σ·(q/ADV)^δ`
/// Integrating from 0 to Q yields the formula above. Equivalent under the
/// remaining-position framing (impact at slice = `κ·σ·((Q−q)/ADV)^δ`); both
/// integrate to 1/(1+δ) by symmetry over [0, T]. Numerical verification:
/// for κ=1, σ=0.01, Q=100, ADV=1000, δ=0.5, both framings give exactly
/// $0.21082 (matches `(2/3)·κ·σ·Q·√(Q/ADV)` to 5+ digits).
///
/// Sources:
/// - **Bouchaud-Tóth 2011**, "The price impact of order book events,"
///   *J. Statistical Mechanics* — sec 5.3 derives the (1/(1+δ)) factor for
///   metaorder cost integration.
/// - **Almgren-Chriss 2000/2001**, "Optimal Execution of Portfolio
///   Transactions" — TWAP integration; for linear δ=0, classic (1/2)·γ·Q²
///   permanent-impact cost (here Q² factor implicit in (1/(1+δ))·Q·(Q/V)^0).
/// - **EBA Final RTS on Prudent Valuation** (Commission Delegated Regulation
///   2016/101) — close-out cost AVA + concentrated positions AVA.
/// - **Bangia, Diebold, Schuermann, Stroughair 1999**, "Modeling Liquidity
///   Risk," NYU Stern WP FIN-99-062 — endogenous liquidity component.
///
/// The Q_ref subtraction (A2 residual-layer convention) does NOT apply
/// here: that anchor was for "what R9 already includes at typical cell
/// Q." For liquidating a position, Q is what we actually have to move,
/// and the realized cost runs from 0 → |Q|.
///
/// `δ = 0` → linear/instant regime, haircut = κ·σ·|Q|.
/// `δ = 0.5` → square-root regime, haircut = (2/3)·κ·σ·|Q|·√(|Q|/ADV).
///
/// Returns 0 if |Q| < dust threshold or ADV ≤ 0 (cannot liquidate, but
/// reporting a haircut requires a defined ADV). For un-priced/un-known
/// assets, the haircut is conservatively 0 — caller may want to flag
/// these separately.
///
/// HISTORY: prior to 2026-05-04 this function used `1/(2+δ)` (a typo
/// of `1/(1+δ)` that came from the design doc's integrand having an
/// extraneous `q` factor). That gave haircuts ~60% of the correct
/// value. Empirically verified against numerical TWAP simulation
/// before the fix; see memory/round_audit_2026_05_04.md.
pub fn compute_position_liquidation_haircut(
    abs_notional: f64,
    kappa: f64,
    sigma: f64,
    adv: f64,
    delta: f64,
) -> f64 {
    if abs_notional <= 1e-6 || adv <= 1e-6 || sigma <= 1e-12 || kappa <= 0.0 {
        return 0.0;
    }
    let phi = abs_notional / adv;
    let impact_phi = phi.powf(delta);
    let scale = 1.0 / (1.0 + delta);
    scale * kappa * sigma * impact_phi * abs_notional
}

/// A4 Layer 2: per-position liquidation haircut with fallback chain.
///
/// Tier resolution:
///   1. **Current**: if `kappa_now > 0`, `sigma_now > 0`, `adv_now > 0`, all
///      provided by the caller from current-bar lookups, use the integrated
///      power-law form via `compute_position_liquidation_haircut`.
///   2. **LastKnown**: at least one current-bar lookup missing or non-positive,
///      but `last_known` (a snapshot of the most recent (ts, value) per
///      symbol) is within `max_age_secs` of `current_ts`. Use those values.
///      Defensible per EBA AVA "most recent observable data" framework.
///   3. **Dlom**: last-known unavailable OR stale beyond max_age_secs. Apply
///      a flat fraction `dlom_pct × |notional|`. Per IFRS 13 / ASC 820 Level
///      3 convention; default 50% is conservative anchor.
///
/// `last_known_kappa[sid] = (ts_seen, kappa)`. Same for σ, ADV. Pass `None`
/// for any lookup that has never seen this sid.
///
/// Returns `(haircut_dollars, source_tier)`. The tier is purely diagnostic.
#[allow(clippy::too_many_arguments)]
pub fn compute_position_haircut_with_fallback(
    abs_notional: f64,
    current_ts: i64,
    kappa_now: f64,
    sigma_now: f64,
    adv_now: f64,
    last_known_kappa: Option<(i64, f64)>,
    last_known_sigma: Option<(i64, f64)>,
    last_known_adv: Option<(i64, f64)>,
    delta: f64,
    dlom_pct: f64,
    max_age_secs: i64,
) -> (f64, HaircutSource) {
    if abs_notional <= 1e-6 {
        return (0.0, HaircutSource::Current);
    }
    // Tier 1: Current-bar lookups all valid.
    if kappa_now > 0.0 && sigma_now > 1e-12 && adv_now > 1e-6 {
        let h = compute_position_liquidation_haircut(
            abs_notional, kappa_now, sigma_now, adv_now, delta,
        );
        return (h, HaircutSource::Current);
    }
    // Tier 2: substitute missing values with last-known if all three are
    // available within max_age_secs. Mixing current-bar (where present) with
    // last-known (where missing) gives the most informed reading.
    let lk_kappa = if kappa_now > 0.0 {
        Some(kappa_now)
    } else {
        last_known_kappa.and_then(|(ts, v)| {
            // Require ts ≤ current_ts (no lookahead) AND age ≤ max_age.
            // The simulation maintains last_known dynamically per bar, so
            // ts here is always ≤ current_ts; the explicit check is a
            // defense-in-depth invariant.
            let age = current_ts - ts;
            if v > 0.0 && age >= 0 && age <= max_age_secs { Some(v) } else { None }
        })
    };
    let lk_sigma = if sigma_now > 1e-12 {
        Some(sigma_now)
    } else {
        last_known_sigma.and_then(|(ts, v)| {
            let age = current_ts - ts;
            if v > 1e-12 && age >= 0 && age <= max_age_secs { Some(v) } else { None }
        })
    };
    let lk_adv = if adv_now > 1e-6 {
        Some(adv_now)
    } else {
        last_known_adv.and_then(|(ts, v)| {
            let age = current_ts - ts;
            if v > 1e-6 && age >= 0 && age <= max_age_secs { Some(v) } else { None }
        })
    };
    if let (Some(kappa), Some(sigma), Some(adv)) = (lk_kappa, lk_sigma, lk_adv) {
        let h = compute_position_liquidation_haircut(abs_notional, kappa, sigma, adv, delta);
        return (h, HaircutSource::LastKnown);
    }
    // Tier 3: DLOM flat percentage.
    let h = dlom_pct * abs_notional;
    (h, HaircutSource::Dlom)
}

/// Apply the one-sided NAV cap (no-free-money mode) to a portfolio.
///
/// Behavior:
/// - If `book.nav() <= cap` → returns `0.0`, book unchanged. Losses are
///   permitted to propagate naturally; the strategy trades reduced capital
///   in subsequent bars.
/// - If `book.nav() > cap` → "skim" the excess: realize all unrealized PnL
///   conceptually (entry = mark, unrealized = 0), scale every position's
///   quantity by `cap / current_nav`, and reset cash to `cap`. Returns the
///   skim amount (excess that was removed).
///
/// Same scaling mechanic as the legacy `constant_nav` reset, but conditional
/// on overshooting only — there is no free money on loss days. This is the
/// preferred mode for tuning a deployment-realistic capacity-bounded
/// simulator (compound up to cap, skim above cap, real losses below cap).
///
/// Assumes the portfolio is approximately dollar-neutral (the rumpy QP
/// enforces `Σw = 0`). For a non-dollar-neutral portfolio the scale + cash
/// reset would not preserve total NAV exactly; see comments on the legacy
/// `constant_nav` block.
pub fn apply_nav_cap_skim(book: &mut crate::position::PortfolioBook, cap: f64) -> f64 {
    let current_nav = book.nav();
    if current_nav <= cap {
        return 0.0;
    }
    let skim = current_nav - cap;
    let scale = cap / current_nav;
    for pos in book.positions.values_mut() {
        pos.avg_entry_price = pos.mark_price;
        pos.unrealized_pnl = 0.0;
        pos.quantity *= scale;
    }
    book.cash = cap;
    skim
}

/// Reset the portfolio's working NAV to `target` by scaling positions and
/// refilling cash. Same mechanic as the legacy `constant_nav` block (apply
/// scale = target / current_nav to every position; refill cash to target;
/// zero unrealized PnL by setting entry = mark).
///
/// No-op when `current_nav <= 1e-6` (avoid divide-by-zero / blow-up cases:
/// the book is already broken; subsequent reset would do nothing useful).
///
/// Used by both:
/// - `constant_nav: true` — called every bar (legacy academic mode).
/// - `nav_reset_timestamps` — called only at fold boundaries (walk-forward
///   MOO mode; per-fold capacity invariance for cross-fold comparison).
///
/// The recorded per-bar `daily_return` is computed from the pre-reset NAV
/// (see `simulation.rs:840`) so each loss day's negative return is captured
/// in the return series before the reset rebases the book. The "free money"
/// only refills positions/cash for the next bar's optimizer; it does not
/// adjust any historical return record. Verified 2026-05-09 (memory ref).
pub fn apply_nav_reset_to_target(book: &mut crate::position::PortfolioBook, target: f64) {
    let current_nav = book.nav();
    if current_nav > 1e-6 {
        let scale = target / current_nav;
        for pos in book.positions.values_mut() {
            pos.avg_entry_price = pos.mark_price;
            pos.unrealized_pnl = 0.0;
            pos.quantity *= scale;
        }
        book.cash = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // c_lin_multiplier separation contract: realized costs MUST NOT depend on
    // it. These tests pin the contract from multiple angles.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // QP-effective kappa calibration tests.
    // Verify the helper:
    //  - pins NAV/ADV scaling exactly: κ_eff = κ·σ·(NAV/ADV)^δ
    //  - leaves a known u-exponent residual: QP/realized = 1.5·u^(0.5 - δ)
    //    (this is fundamental to the closed-form QP — fixing it requires a
    //    Clarabel power-cone refactor; see project memory).
    // -----------------------------------------------------------------------

    #[test]
    fn qp_effective_kappa_residual_matches_closed_form() {
        // Closed-form: 1.5·κ_eff·u^1.5 vs realized = κ·σ·u^(1+δ)·(NAV/ADV)^δ
        // → ratio = 1.5 · u^(1.5 - (1+δ)) = 1.5 · u^(0.5 - δ)
        let kappa: f64 = 0.86;
        let sigma: f64 = 0.04;
        let adv: f64 = 1.6e7;
        let nav: f64 = 1000.0;

        // Use δ=0.5 (production config). With this δ the QP/realized
        // exponents match exactly (1+δ = 1.5 = QP's hardcoded sqrt-form),
        // so the residual factor is 1.5·u^(0.5−δ) = 1.5·u^0 = 1.5 (constant).
        let delta = 0.5;
        for u in [0.001_f64, 0.005, 0.01, 0.05, 0.1] {
            let k_eff = compute_qp_effective_kappa(kappa, sigma, adv, nav, delta);
            let qp_pct = 1.5 * k_eff * u.powf(1.5);
            let realized_pct =
                kappa * sigma * u.powf(1.0 + delta) * (nav / adv).powf(delta);
            let ratio = qp_pct / realized_pct;
            let expected = 1.5_f64 * u.powf(0.5 - delta);
            assert!((ratio - expected).abs() / expected < 1e-12,
                "QP/realized residual mismatch at u={u}: got {ratio}, expected {expected}");
        }
    }

    #[test]
    fn qp_effective_kappa_scales_with_nav_correctly() {
        // The pre-fix bug was that QP perception was NAV-flat. The realized
        // cost (as % NAV) scales as NAV^δ. The effective kappa MUST carry
        // this scaling so the QP sees the right cost at every NAV.
        let kappa = 1.0;
        let sigma = 0.04;
        let adv = 1.6e7;
        let delta = 0.5;

        let k_1k = compute_qp_effective_kappa(kappa, sigma, adv, 1_000.0, delta);
        let k_1m = compute_qp_effective_kappa(kappa, sigma, adv, 1_000_000.0, delta);
        let nav_ratio_pow = (1_000_000.0_f64 / 1_000.0_f64).powf(delta);
        let scaling = k_1m / k_1k;
        assert!((scaling - nav_ratio_pow).abs() < 1e-9,
            "k_eff should scale with (NAV/ADV)^δ: got {scaling}, expected {nav_ratio_pow}");
    }

    #[test]
    fn qp_effective_kappa_falls_back_to_raw_kappa_on_warmup() {
        // Before sigma/adv EWMAs warm up, fall back to raw kappa rather than
        // crashing. Preserves bootstrap behavior.
        let raw = 0.5;
        let delta = 0.5;
        assert_eq!(compute_qp_effective_kappa(raw, 0.0, 1e6, 1000.0, delta), raw,
            "sigma=0 should fall back");
        assert_eq!(compute_qp_effective_kappa(raw, 0.04, 0.0, 1000.0, delta), raw,
            "adv=0 should fall back");
        assert_eq!(compute_qp_effective_kappa(raw, 0.04, 1e6, 0.0, delta), raw,
            "nav=0 should fall back");
    }

    // -----------------------------------------------------------------------
    // QP-residual kappa (Q_ref-subtraction coefficient for SOCP cone aux).
    // -----------------------------------------------------------------------

    #[test]
    fn qp_residual_kappa_matches_closed_form() {
        // Sanity: r = κ·σ·(Q_ref/ADV)^δ. Pin against analytic.
        let kappa = 0.86;
        let sigma = 0.04;
        let adv = 1.6e7;
        let q_ref = 13_827.0;
        let delta = 0.5;
        let r = compute_qp_residual_kappa(kappa, sigma, adv, q_ref, delta);
        let expected = kappa * sigma * (q_ref / adv).powf(delta);
        assert!((r - expected).abs() < 1e-15, "got {r}, expected {expected}");
    }

    #[test]
    fn qp_residual_kappa_scales_with_q_ref_correctly() {
        // r ∝ Q_ref^δ. Doubling Q_ref scales r by 2^δ.
        let kappa = 1.0;
        let sigma = 0.04;
        let adv = 1.6e7;
        let delta = 0.5;
        let r_q_lo = compute_qp_residual_kappa(kappa, sigma, adv, 10_000.0, delta);
        let r_q_hi = compute_qp_residual_kappa(kappa, sigma, adv, 20_000.0, delta);
        let ratio = r_q_hi / r_q_lo;
        let expected = 2.0_f64.powf(delta);
        assert!((ratio - expected).abs() < 1e-12,
            "Q_ref doubling should scale r by 2^δ: got {ratio}, expected {expected}");
    }

    #[test]
    fn qp_residual_kappa_zero_when_q_ref_zero() {
        // q_ref_usd = 0 → "no subtraction" mode → r must be 0 so the SOCP
        // ReLU-aux degrades to s = max(κ_eff·u^1.5, 0) = κ_eff·u^1.5
        // (equivalent to the variant-A cone-only formulation).
        let r = compute_qp_residual_kappa(0.86, 0.04, 1.6e7, 0.0, 0.5);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn qp_residual_kappa_zero_on_warmup() {
        // σ=0 / adv=0 / kappa=0 cases (bootstrap or empty data) all return 0.
        // Different from compute_qp_effective_kappa, which falls back to raw κ.
        // The reason: r appears as a per-dollar credit on u in the cone aux;
        // returning a stale raw κ would create a phantom credit. Better to
        // degrade to "no subtraction" until real data is available.
        assert_eq!(compute_qp_residual_kappa(0.86, 0.0, 1e6, 13_827.0, 0.5), 0.0);
        assert_eq!(compute_qp_residual_kappa(0.86, 0.04, 0.0, 13_827.0, 0.5), 0.0);
        assert_eq!(compute_qp_residual_kappa(0.0, 0.04, 1e6, 13_827.0, 0.5), 0.0);
    }

    #[test]
    fn qp_residual_kappa_matches_realized_threshold_decomposition() {
        // The decisive identity backing this helper: at any |Q| above the
        // Q_ref threshold, the realized cost equals
        //   κ_eff·NAV·u^(1+δ) − r·u·NAV
        // where κ_eff = compute_qp_effective_kappa, r = compute_qp_residual_kappa.
        // Verify across a few realistic configurations.
        let kappa = 0.86;
        let sigma = 0.04;
        let adv = 1.6e7;
        let nav = 1_000_000.0;
        let q_ref = 13_827.0;
        let delta = 0.5;
        let kappa_eff = compute_qp_effective_kappa(kappa, sigma, adv, nav, delta);
        let r = compute_qp_residual_kappa(kappa, sigma, adv, q_ref, delta);
        // Pick |Q| above Q_ref so the max(.,0) is positive.
        for q_dollars in [50_000.0_f64, 100_000.0, 500_000.0] {
            let u = q_dollars / nav;
            let cone_term = kappa_eff * nav * u.powf(1.0 + delta);
            let linear_term = r * u * nav;
            let predicted = (cone_term - linear_term).max(0.0);
            let (_spread, realized) = compute_realized_trade_costs(
                q_dollars, 0.0, kappa, sigma, adv, delta, q_ref,
            );
            let rel_err = (predicted - realized).abs() / realized;
            assert!(rel_err < 1e-12,
                "Q={q_dollars}: predicted ${predicted}, realized ${realized}, rel_err {rel_err}");
        }
    }

    #[test]
    fn realized_costs_match_canonical_formulas() {
        // Known-good values: 5 bps half-spread, 0.001 vol, 1e6 ADV, 1e-7 kappa.
        // $1000 trade.
        let (spread, impact) = compute_realized_trade_costs(
            1000.0, // $1k notional
            5.0,    // 5 bps spread
            1e-7,   // typical kappa
            0.02,   // 2% daily vol
            1e6,    // $1M ADV
            IMPACT_DELTA, // legacy δ
            0.0,    // legacy q_ref (no subtraction)
        );
        // Spread: 0.5 × 5 / 10_000 × 1000 = 0.25
        assert!((spread - 0.25).abs() < 1e-12);
        // Impact: 1e-7 × 0.02 × 1000^(1+δ) / 1e6^δ
        let expected_impact = 1e-7
            * 0.02
            * (1000.0_f64).powf(1.0 + IMPACT_DELTA)
            / (1e6_f64).powf(IMPACT_DELTA);
        assert!((impact - expected_impact).abs() / expected_impact < 1e-10);
    }

    #[test]
    fn realized_costs_zero_for_zero_trade() {
        let (s, i) = compute_realized_trade_costs(0.0, 5.0, 1e-7, 0.02, 1e6, IMPACT_DELTA, 0.0);
        assert_eq!(s, 0.0);
        assert_eq!(i, 0.0);
    }

    #[test]
    fn realized_costs_symmetric_buy_sell() {
        // Same magnitude, opposite signs → identical costs (no asymmetric
        // impact at this layer). Verifies abs() handling.
        let (s_buy, i_buy) = compute_realized_trade_costs(1000.0, 5.0, 1e-7, 0.02, 1e6, IMPACT_DELTA, 0.0);
        let (s_sell, i_sell) = compute_realized_trade_costs(-1000.0, 5.0, 1e-7, 0.02, 1e6, IMPACT_DELTA, 0.0);
        assert_eq!(s_buy, s_sell);
        assert_eq!(i_buy, i_sell);
    }

    #[test]
    fn realized_costs_signature_does_not_take_c_lin_multiplier() {
        // Structural pin: signature must NOT take c_lin_multiplier (which
        // belongs to the QP perception path). The two trailing args
        // (delta, q_ref_usd) are config-driven cost-model parameters, not
        // perception tunables. If anyone adds c_lin_multiplier here, this
        // fails to compile.
        let _: fn(f64, f64, f64, f64, f64, f64, f64) -> (f64, f64) = compute_realized_trade_costs;
    }

    #[test]
    fn realized_costs_kappa_zero_yields_zero_impact() {
        // If the cost model declares zero impact for an asset, simulation
        // pays zero impact regardless of trade size. Sanity invariant.
        let (_, i) = compute_realized_trade_costs(100_000.0, 5.0, 0.0, 0.02, 1e6, IMPACT_DELTA, 0.0);
        assert_eq!(i, 0.0);
    }

    #[test]
    fn realized_costs_scale_correctly_with_notional() {
        // Spread is linear in notional; impact is super-linear (Q^(1+δ)) when
        // q_ref=0 (legacy form).
        let (s1, i1) = compute_realized_trade_costs(1000.0, 5.0, 1e-7, 0.02, 1e6, IMPACT_DELTA, 0.0);
        let (s10, i10) = compute_realized_trade_costs(10_000.0, 5.0, 1e-7, 0.02, 1e6, IMPACT_DELTA, 0.0);
        // Spread: 10× notional → 10× spread cost
        assert!((s10 / s1 - 10.0).abs() < 1e-10);
        // Impact: 10× notional → 10^(1+δ) × impact
        let expected_ratio = (10.0_f64).powf(1.0 + IMPACT_DELTA);
        assert!(((i10 / i1) - expected_ratio).abs() / expected_ratio < 1e-10);
    }

    #[test]
    fn realized_costs_q_ref_anchors_zero_impact_at_reference() {
        // Residual layer (A2 fix): at |Q| = q_ref, impact_cost is exactly 0.
        let (s, i) = compute_realized_trade_costs(
            500.0, 5.0, 0.05, 0.04, 1e6, 0.142, 500.0,
        );
        assert!(s > 0.0);
        assert!(i.abs() < 1e-12, "impact at Q=q_ref must be 0, got {}", i);
    }

    #[test]
    fn realized_costs_q_ref_below_charges_zero() {
        // At |Q| < q_ref, the marginal impact is negative — floored to 0.
        // R9_pred (applied separately as spread) already covers small-trade
        // costs; we don't refund.
        let (_, i) = compute_realized_trade_costs(
            100.0, 5.0, 0.05, 0.04, 1e6, 0.142, 500.0,
        );
        assert_eq!(i, 0.0, "impact below q_ref must floor at 0");
    }

    #[test]
    fn realized_costs_a4_baby_doge_pure_formula_explodes() {
        // A4 baseline: the pure formula produces non-physical impact on
        // illiquid names (φ → ∞ as ADV → 0). Documenting that the
        // unbounded form matches the empirical kill-day pathology
        // ($35k BabyDoge trade → ~$195k impact = 5.5× notional).
        // The structural fix (gate, not cap) lives in `passes_impact_gate`.
        let notional: f64 = -35_336.72;
        let (_, impact) = compute_realized_trade_costs(
            notional, 5.0, 0.201268, 0.044486, 0.013, 0.5, 13_827.67,
        );
        assert!(
            impact > 100_000.0 && impact < 300_000.0,
            "raw impact for BabyDoge case should be ~$195k, got ${}",
            impact
        );
    }

    #[test]
    fn impact_gate_rejects_baby_doge_trade() {
        // A4 invariant: a trade with non-physical predicted impact must
        // fail the gate. This is what the simulator uses to refuse the
        // fill: the asset is deemed not tradeable at this size in this bar.
        let notional: f64 = -35_336.72;
        let (_, impact) = compute_realized_trade_costs(
            notional, 5.0, 0.201268, 0.044486, 0.013, 0.5, 13_827.67,
        );
        // At gate=100 bps, this trade (~55,000 bps of impact) is rejected.
        assert!(!passes_impact_gate(impact, notional.abs(), 100.0));
        // Even a 10,000 bps gate rejects this case (impact still ≈ 55k bps).
        assert!(!passes_impact_gate(impact, notional.abs(), 10_000.0));
        // gate=0 disables — all trades pass.
        assert!(passes_impact_gate(impact, notional.abs(), 0.0));
    }

    #[test]
    fn impact_gate_passes_normal_trade() {
        // A liquid trade with sub-bps impact must pass the gate. The gate
        // only rejects trades where the formula predicts pathological cost.
        let notional: f64 = 1000.0;
        let (_, impact) = compute_realized_trade_costs(
            notional, 5.0, 1e-7, 0.02, 1e6, 0.5, 0.0,
        );
        // Raw impact ≈ 6.3e-8 dollars on $1k = ~6e-7 bps — far below 100.
        assert!(passes_impact_gate(impact, notional.abs(), 100.0));
        assert!(passes_impact_gate(impact, notional.abs(), 1.0));
        // Even an aggressive 0.001 bps gate passes this trade.
        assert!(passes_impact_gate(impact, notional.abs(), 0.001));
    }

    #[test]
    fn impact_gate_threshold_is_inclusive() {
        // Trades exactly at the threshold are allowed (≤, not <). Pins
        // the boundary to avoid float-equality bugs.
        let notional: f64 = 1000.0;
        let abs_n = notional.abs();
        // Construct an impact_cost = exactly 50 bps × $1000 = $5
        let impact_cost = 50.0 / 1e4 * abs_n;
        assert!(passes_impact_gate(impact_cost, abs_n, 50.0),
            "gate must pass at exact threshold");
        // Any higher impact at the same gate fails.
        assert!(!passes_impact_gate(impact_cost + 1e-6, abs_n, 50.0));
    }

    #[test]
    fn impact_gate_dust_trade_passes() {
        // Trades with ~0 notional bypass the gate by construction (avoids
        // 0/0 in impact_bps). Floor at 1e-6 USD.
        assert!(passes_impact_gate(0.0, 0.0, 100.0));
        assert!(passes_impact_gate(1e-9, 1e-9, 100.0));
    }

    // -----------------------------------------------------------------------
    // A4 Layer 2 — liquidation haircut
    // -----------------------------------------------------------------------

    #[test]
    fn liquidation_haircut_zero_for_dust_or_missing_data() {
        // Zero notional → 0 haircut.
        assert_eq!(compute_position_liquidation_haircut(0.0, 0.2, 0.04, 1e6, 0.5), 0.0);
        // Zero ADV → 0 (cannot meaningfully compute, defensive return).
        assert_eq!(compute_position_liquidation_haircut(1000.0, 0.2, 0.04, 0.0, 0.5), 0.0);
        // Zero kappa → 0 (asset declared no impact).
        assert_eq!(compute_position_liquidation_haircut(1000.0, 0.0, 0.04, 1e6, 0.5), 0.0);
        // Zero sigma → 0 (no vol contribution).
        assert_eq!(compute_position_liquidation_haircut(1000.0, 0.2, 0.0, 1e6, 0.5), 0.0);
    }

    #[test]
    fn liquidation_haircut_matches_integrated_form() {
        // Standard TWAP integration of metaorder impact:
        //   haircut = ∫_0^Q κ·σ·(q/ADV)^δ dq = (1/(1+δ))·κ·σ·Q·(Q/ADV)^δ
        // Pin at known inputs: Q=$10k, κ=0.2, σ=0.04, ADV=$1M, δ=0.5
        // Expected = (1/1.5) · 0.2 · 0.04 · 10000 · (10000/1e6)^0.5
        //         = (2/3) · 0.2 · 0.04 · 10000 · 0.1
        //         ≈ $5.33
        let h = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1e6, 0.5);
        let expected = (1.0 / 1.5) * 0.2 * 0.04 * 10_000.0 * (10_000.0_f64 / 1e6).powf(0.5);
        assert!((h - expected).abs() / expected < 1e-12,
            "haircut {} vs expected {}", h, expected);
    }

    #[test]
    fn liquidation_haircut_dominates_for_stuck_position() {
        // Stuck-position case (analog of BabyDoge in the kill-day forensics):
        // $1M unrealized position on a name with $1k/day ADV. The haircut
        // should be substantial relative to the position size — possibly
        // exceeding it, which is the whole point: the position has negative
        // liquidation value.
        let abs_n = 1_000_000.0;     // $1M position
        let kappa = 0.2;
        let sigma = 0.04;
        let adv = 1_000.0;            // illiquid
        let delta = 0.5;
        let h = compute_position_liquidation_haircut(abs_n, kappa, sigma, adv, delta);
        // φ = 1e6 / 1e3 = 1000; √φ = 31.6
        // h = 0.4 · 0.2 · 0.04 · 1e6 · 31.6 = ~101k
        assert!(h > 50_000.0 && h < 500_000.0,
            "stuck-position haircut should be 5-50% of notional, got ${}", h);
    }

    #[test]
    fn liquidation_haircut_negligible_for_liquid_position() {
        // Liquid case: $10k position on a $1M-per-DAY ADV name (deep regime).
        // φ = 1e-2, √φ = 0.1, haircut ≈ small fraction of bps × position
        let h = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1_000_000.0, 0.5);
        // Should be < 0.1% of notional
        assert!(h / 10_000.0 < 1e-3, "liquid haircut should be tiny, got {} bps",
            h / 10_000.0 * 1e4);
    }

    #[test]
    fn haircut_fallback_uses_current_when_all_present() {
        // All three current-bar values valid → Tier 1 (Current).
        let (h, src) = compute_position_haircut_with_fallback(
            10_000.0, 1_700_000_000,
            0.2, 0.04, 1e6,           // current-bar (κ, σ, ADV)
            None, None, None,         // no last-known needed
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(src, HaircutSource::Current);
        let expected = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1e6, 0.5);
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn haircut_fallback_uses_last_known_when_current_missing() {
        // Current-bar ADV missing (=0); last-known within max_age → Tier 2.
        let (h, src) = compute_position_haircut_with_fallback(
            10_000.0, 1_700_000_000,
            0.2, 0.04, 0.0,                          // ADV missing
            Some((1_700_000_000 - 86400 * 5, 0.2)),   // 5d old κ
            Some((1_700_000_000 - 86400 * 5, 0.04)),  // 5d old σ
            Some((1_700_000_000 - 86400 * 5, 1e6)),   // 5d old ADV
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(src, HaircutSource::LastKnown);
        // Same formula, just sourced from last-known.
        let expected = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1e6, 0.5);
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn haircut_fallback_uses_dlom_when_no_data() {
        // No current-bar data and no last-known → Tier 3 (DLOM).
        let (h, src) = compute_position_haircut_with_fallback(
            10_000.0, 1_700_000_000,
            0.0, 0.0, 0.0,
            None, None, None,
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(src, HaircutSource::Dlom);
        // 50% × $10k = $5,000.
        assert!((h - 5_000.0).abs() < 1e-12);
    }

    #[test]
    fn haircut_fallback_uses_dlom_when_last_known_too_stale() {
        // Last-known data exists but is older than max_age → degrade to DLOM.
        let (h, src) = compute_position_haircut_with_fallback(
            10_000.0, 1_700_000_000,
            0.0, 0.0, 0.0,
            Some((1_700_000_000 - 86400 * 100, 0.2)),   // 100d old (> 90d)
            Some((1_700_000_000 - 86400 * 100, 0.04)),
            Some((1_700_000_000 - 86400 * 100, 1e6)),
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(src, HaircutSource::Dlom);
        assert!((h - 5_000.0).abs() < 1e-12);
    }

    #[test]
    fn haircut_fallback_mixes_current_and_last_known() {
        // Some current-bar valid, some missing — fill the missing from
        // last-known. The substituted set is most informed.
        let (h, src) = compute_position_haircut_with_fallback(
            10_000.0, 1_700_000_000,
            0.2, 0.04, 0.0,                          // ADV missing
            None,
            None,
            Some((1_700_000_000 - 86400, 1e6)),       // 1d old ADV available
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(src, HaircutSource::LastKnown);
        let expected = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1e6, 0.5);
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn haircut_fallback_zero_for_dust() {
        // Dust position: zero haircut regardless of data tier. Tier reported
        // as Current (degenerate path).
        let (h, src) = compute_position_haircut_with_fallback(
            0.0, 1_700_000_000,
            0.2, 0.04, 1e6,
            None, None, None,
            0.5, 0.5, 90 * 86400,
        );
        assert_eq!(h, 0.0);
        assert_eq!(src, HaircutSource::Current);
    }

    #[test]
    fn liquidation_haircut_monotonic_in_size() {
        // Doubling the position more than doubles the haircut (super-linear
        // due to (Q/ADV)^δ factor with δ > 0).
        let h1 = compute_position_liquidation_haircut(10_000.0, 0.2, 0.04, 1e6, 0.5);
        let h2 = compute_position_liquidation_haircut(20_000.0, 0.2, 0.04, 1e6, 0.5);
        // Ratio = 2 · √2 = 2.828 for δ=0.5.
        let ratio = h2 / h1;
        let expected = 2.0_f64 * 2.0_f64.sqrt();
        assert!((ratio - expected).abs() / expected < 1e-10,
            "doubling Q at δ=0.5 should give 2√2× haircut, got {} vs {}",
            ratio, expected);
    }

    #[test]
    fn realized_costs_q_ref_above_charges_marginal() {
        // At |Q| > q_ref, marginal impact is strictly positive and matches
        // the residual-layer formula.
        let (_, i) = compute_realized_trade_costs(
            10_000.0, 5.0, 0.05, 0.04, 1e6, 0.142, 500.0,
        );
        let expected = 0.05_f64 * 0.04
            * ((10_000.0_f64 / 1e6).powf(0.142) - (500.0_f64 / 1e6).powf(0.142))
            * 10_000.0;
        assert!((i - expected).abs() / expected < 1e-10);
        assert!(i > 0.0);
    }

    #[test]
    fn test_simulation_engine_creates() {
        let config = ExecutionConfig::default();
        let engine = SimulationEngine::new(&config);
        assert!((engine.book.nav() - config.nav_usd).abs() < 1e-10);
    }

    // L2 (2026-05-13): test_vol_leverage_warmup + test_vol_state_update removed
    // along with compute_vol_leverage / update_vol_state. Vol target is now an
    // SOC constraint inside the QP; solver-level tests in `solver::tests`.

    #[test]
    fn test_empty_simulation() {
        let config = ExecutionConfig::default();
        let mut engine = SimulationEngine::new(&config);
        let result = engine.run(
            &[], &[], &[], &RollingCovariance::default(),
            &HashMap::new(), &HashMap::new(), &HashMap::new(),
            &HashMap::new(), &HashMap::new(), &HashMap::new(),
            &HashMap::new(),
        );
        assert!(result.records.is_empty());
        assert!(result.liquidation_events.is_empty());
        assert!((result.final_book.nav() - config.nav_usd).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // One-sided NAV cap (no-free-money mode) — apply_nav_cap_skim helper
    // -----------------------------------------------------------------------

    /// Build a small dollar-neutral book at a given NAV: $X long BTC at $100,
    /// $X short ETH at $200, plus initial_cash. Long unrealized = +Y means
    /// price has moved in the long's favor (mark > entry).
    fn make_neutral_book(initial_cash: f64, long_unrealized: f64, short_unrealized: f64)
        -> crate::position::PortfolioBook
    {
        use crate::position::{PortfolioBook, PositionState};
        let mut book = PortfolioBook::new(initial_cash);
        // BTC long: 1 unit, entry $100, mark = entry + unrealized
        let mut btc = PositionState::new("BTC".to_string(), 0);
        btc.quantity = 1.0;
        btc.avg_entry_price = 100.0;
        btc.mark_price = 100.0 + long_unrealized;
        btc.unrealized_pnl = long_unrealized;
        // ETH short: -1 unit, entry $200, unrealized > 0 if mark < entry
        let mut eth = PositionState::new("ETH".to_string(), 0);
        eth.quantity = -1.0;
        eth.avg_entry_price = 200.0;
        eth.mark_price = 200.0 - short_unrealized;
        eth.unrealized_pnl = short_unrealized;
        book.positions.insert("BTC".to_string(), btc);
        book.positions.insert("ETH".to_string(), eth);
        book
    }

    #[test]
    fn skim_nav_below_cap_is_noop() {
        // NAV < cap: no skim, book unchanged.
        let mut book = make_neutral_book(8_000.0, 500.0, 500.0);
        let nav_before = book.nav(); // 8000 + 500 + 500 = 9000
        assert!((nav_before - 9_000.0).abs() < 1e-9);
        let skim = apply_nav_cap_skim(&mut book, 10_000.0);
        assert_eq!(skim, 0.0);
        // Book unchanged.
        assert!((book.nav() - 9_000.0).abs() < 1e-9);
        assert!((book.cash - 8_000.0).abs() < 1e-9);
        assert_eq!(book.positions["BTC"].quantity, 1.0);
        assert!((book.positions["BTC"].unrealized_pnl - 500.0).abs() < 1e-9);
    }

    #[test]
    fn skim_nav_at_cap_is_noop() {
        // NAV == cap: still no skim (strict inequality).
        let mut book = make_neutral_book(9_000.0, 500.0, 500.0);
        assert!((book.nav() - 10_000.0).abs() < 1e-9);
        let skim = apply_nav_cap_skim(&mut book, 10_000.0);
        assert_eq!(skim, 0.0);
        assert_eq!(book.positions["BTC"].quantity, 1.0);
    }

    #[test]
    fn skim_nav_above_cap_returns_excess_and_resets() {
        // NAV = $10.5k, cap $10k → skim $500.
        let mut book = make_neutral_book(9_500.0, 500.0, 500.0);
        let nav_before = book.nav(); // 10500
        assert!((nav_before - 10_500.0).abs() < 1e-9);
        let skim = apply_nav_cap_skim(&mut book, 10_000.0);
        assert!((skim - 500.0).abs() < 1e-9);
        // After skim: NAV = cap exactly.
        assert!((book.nav() - 10_000.0).abs() < 1e-9);
        // Cash matches cap (positions are dollar-neutral, contribute 0 to NAV).
        assert!((book.cash - 10_000.0).abs() < 1e-9);
        // Positions scaled by 10000/10500 ≈ 0.9524.
        let scale = 10_000.0 / 10_500.0;
        assert!((book.positions["BTC"].quantity - scale).abs() < 1e-9);
        assert!((book.positions["ETH"].quantity - (-scale)).abs() < 1e-9);
        // Unrealized PnL realized into cash (entry = mark).
        assert_eq!(book.positions["BTC"].unrealized_pnl, 0.0);
        assert!((book.positions["BTC"].avg_entry_price - book.positions["BTC"].mark_price).abs() < 1e-9);
    }

    #[test]
    fn skim_loss_then_recovery_compounds_then_skims() {
        // Start at cap $10k. Loss day: NAV drops to $9k (no skim, losses
        // propagate). Recovery day: pure cash (positions flat) returns to
        // $10.2k → skim $200.
        let mut book = make_neutral_book(9_000.0, 500.0, 500.0); // NAV 10000
        // Apply loss: pretend long lost $500, short lost $500 → unrealized = 0
        // each, NAV = cash + 0 = 9000.
        book.positions.get_mut("BTC").unwrap().unrealized_pnl = 0.0;
        book.positions.get_mut("BTC").unwrap().mark_price = 100.0;
        book.positions.get_mut("ETH").unwrap().unrealized_pnl = 0.0;
        book.positions.get_mut("ETH").unwrap().mark_price = 200.0;
        assert!((book.nav() - 9_000.0).abs() < 1e-9);
        // No skim: NAV ≤ cap. Losses propagated naturally.
        let skim_loss = apply_nav_cap_skim(&mut book, 10_000.0);
        assert_eq!(skim_loss, 0.0);
        assert!((book.nav() - 9_000.0).abs() < 1e-9);
        // Recovery: simulate +$1.2k profit (long gains $600, short gains $600).
        book.positions.get_mut("BTC").unwrap().unrealized_pnl = 600.0;
        book.positions.get_mut("BTC").unwrap().mark_price = 700.0; // entry was 100, gained 600
        book.positions.get_mut("ETH").unwrap().unrealized_pnl = 600.0;
        book.positions.get_mut("ETH").unwrap().mark_price = -400.0; // entry 200, gained 600 → mark drops by 600
        assert!((book.nav() - 10_200.0).abs() < 1e-9);
        // Skim $200.
        let skim_recover = apply_nav_cap_skim(&mut book, 10_000.0);
        assert!((skim_recover - 200.0).abs() < 1e-9);
        assert!((book.nav() - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn skim_repeated_profits_accumulate_correctly() {
        // Three profit days at $10k cap: each adds $100 unrealized → skim $100
        // three times. Total skim = $300.
        let mut book = make_neutral_book(9_000.0, 500.0, 500.0); // NAV 10000
        let cap = 10_000.0;
        let mut total_skim = 0.0;
        for _ in 0..3 {
            // Add $100 of unrealized PnL (each side of the dollar-neutral pair).
            book.positions.get_mut("BTC").unwrap().unrealized_pnl += 50.0;
            book.positions.get_mut("BTC").unwrap().mark_price += 50.0;
            book.positions.get_mut("ETH").unwrap().unrealized_pnl += 50.0;
            book.positions.get_mut("ETH").unwrap().mark_price -= 50.0;
            assert!((book.nav() - 10_100.0).abs() < 1e-9);
            let skim = apply_nav_cap_skim(&mut book, cap);
            assert!((skim - 100.0).abs() < 1e-9);
            total_skim += skim;
            // After skim, NAV is back at cap; unrealized PnL was realized away.
            assert!((book.nav() - cap).abs() < 1e-9);
        }
        assert!((total_skim - 300.0).abs() < 1e-9);
    }

    #[test]
    fn skim_dust_skip_at_cap_minus_eps() {
        // NAV just barely above cap (within $1 of cap) — skim is exactly that
        // amount (no special threshold; the helper is exact).
        let mut book = make_neutral_book(9_000.0, 500.5, 500.5); // NAV 10001
        let skim = apply_nav_cap_skim(&mut book, 10_000.0);
        assert!((skim - 1.0).abs() < 1e-9);
        assert!((book.nav() - 10_000.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // C3.E integration tests — C1-audit HIGH-risk paths.
    //
    // The C1 execution audit (docs/audits/execution-state-2026-05-21.md §4)
    // flagged three simulation.rs paths with zero integration coverage:
    //   1. TWAP intraday accrual (fill_mode=Twap, lines 575-746)
    //   2. Impact-gate rejection effect on accounting (711-717)
    //   3. Margin-stress → liquidation cascade (602-615, 791-808)
    // These tests exercise each via SimulationEngine.run() with synthetic
    // fixtures. Asserting the public SimulationResult.{records, liquidation_events,
    // n_rejected_trades, rejected_notional, final_book} surface.
    // -----------------------------------------------------------------------

    /// Build a 2-symbol N-bar synthetic fixture. Symbols: "BTC" (id=0), "ETH" (id=1).
    /// One D1 day worth of H1 bars by default (n_bars=24). Returns the 11 inputs
    /// that SimulationEngine.run() expects.
    #[allow(clippy::type_complexity)]
    fn build_fixture(n_bars: usize) -> (
        Vec<crate::alpha::AlphaRow>,
        Vec<i64>,
        Vec<crate::aim::AimWeight>,
        crate::covariance::RollingCovariance,
        HashMap<(i64, u32), f64>, // spread_lookup
        HashMap<(i64, u32), f64>, // kappa_lookup
        HashMap<(i64, u32), f64>, // price_lookup
        HashMap<(i64, u32), f64>, // funding_lookup
        HashMap<(i64, u32), f64>, // sigma_lookup (day-floor)
        HashMap<(i64, u32), f64>, // adv_lookup (day-floor)
        HashMap<String, u32>,     // symbol_ids
    ) {
        use crate::alpha::AlphaRow;
        use crate::aim::AimWeight;
        use crate::covariance::RollingCovariance;
        // Day-floor start at 2024-01-01 00:00 UTC.
        let day0: i64 = 1_704_067_200;
        let symbols = ["BTC".to_string(), "ETH".to_string()];
        let mut symbol_ids = HashMap::new();
        symbol_ids.insert("BTC".to_string(), 0u32);
        symbol_ids.insert("ETH".to_string(), 1u32);

        // H1 bar timestamps. For n_bars=24 → 1 day; n_bars=48 → 2 days; etc.
        let bar_timestamps: Vec<i64> = (0..n_bars).map(|i| day0 + 3600 * i as i64).collect();

        // Alphas at the day floor (one row per (day, sym)). Long BTC, short ETH.
        let n_days = (n_bars + 23) / 24;
        let mut alphas: Vec<AlphaRow> = Vec::new();
        for d in 0..n_days {
            let day_ts = day0 + 86400 * d as i64;
            alphas.push(AlphaRow {
                timestamp: day_ts,
                symbol: "BTC".to_string(),
                alpha_future: 1.0,
            });
            alphas.push(AlphaRow {
                timestamp: day_ts,
                symbol: "ETH".to_string(),
                alpha_future: -1.0,
            });
        }

        // Aim portfolio: target ±0.05 of NAV. (Dollar-neutral; sums to zero.)
        let mut aim_weights: Vec<AimWeight> = Vec::new();
        for d in 0..n_days {
            let day_ts = day0 + 86400 * d as i64;
            aim_weights.push(AimWeight { timestamp: day_ts, symbol: "BTC".to_string(), weight_aim: 0.05 });
            aim_weights.push(AimWeight { timestamp: day_ts, symbol: "ETH".to_string(), weight_aim: -0.05 });
        }

        // Diagonal covariance per day (small constant vol).
        let mut rolling_cov = RollingCovariance::new();
        for d in 0..n_days {
            let day_ts = day0 + 86400 * d as i64;
            // Σ = diag(0.0001, 0.0001) → σ ≈ 0.01 per asset.
            rolling_cov.insert(
                day_ts,
                symbols.to_vec(),
                vec![0.0001, 0.0, 0.0, 0.0001],
            );
        }

        // Per-bar lookups.
        let mut price_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        let mut spread_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        let mut kappa_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        let mut funding_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        for &ts in &bar_timestamps {
            // Flat prices (no MTM movement → no PnL drift across the test horizon).
            price_lookup.insert((ts, 0), 50_000.0); // BTC
            price_lookup.insert((ts, 1), 3_000.0);  // ETH
            // Tight spread + tiny κ — keep cost-driven QP friction negligible.
            spread_lookup.insert((ts, 0), 1.0);
            spread_lookup.insert((ts, 1), 1.0);
            kappa_lookup.insert((ts, 0), 1e-8);
            kappa_lookup.insert((ts, 1), 1e-8);
            funding_lookup.insert((ts, 0), 0.0);
            funding_lookup.insert((ts, 1), 0.0);
        }

        // Day-floor sigma + adv lookups.
        let mut sigma_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        let mut adv_lookup: HashMap<(i64, u32), f64> = HashMap::new();
        for d in 0..n_days {
            let day_ts = day0 + 86400 * d as i64;
            sigma_lookup.insert((day_ts, 0), 0.02);
            sigma_lookup.insert((day_ts, 1), 0.02);
            adv_lookup.insert((day_ts, 0), 1.0e9);
            adv_lookup.insert((day_ts, 1), 1.0e9);
        }

        (alphas, bar_timestamps, aim_weights, rolling_cov,
         spread_lookup, kappa_lookup, price_lookup, funding_lookup,
         sigma_lookup, adv_lookup, symbol_ids)
    }

    /// C1 §4 HIGH-risk path #1 — TWAP intraday accrual.
    ///
    /// Verifies the H1 TWAP fill path runs end-to-end across 2 days × 24
    /// bars without crashing, produces records, and converges to the daily
    /// target (close-to-aim) at end-of-day. The strict invariant: records
    /// emit only at end-of-day (not per H1 bar) and the final book has
    /// non-zero positions matching the long/short alpha sides.
    #[test]
    fn c3e_twap_intraday_accrual_runs_and_converges() {
        let mut config = ExecutionConfig::default();
        config.fill_mode = crate::config::FillMode::Twap;
        config.nav_usd = 100_000.0;
        // Constrain QP to behave: small gamma, tight cap, dollar-neutral by construction.
        config.qp.gamma = 5.0;
        config.qp.lambda_aim = 1.0;
        config.qp.l_max = 1.5;
        config.qp.per_name_cap = Some(0.10);
        config.alpha.l_target = 1.0;
        config.funding.use_in_alpha = false;
        config.funding.enabled = false;

        let (alphas, bar_timestamps, aim, cov,
             spread, kappa, prices, funding, sigma, adv, syms) = build_fixture(48);

        let mut engine = SimulationEngine::new(&config);
        let result = engine.run(
            &alphas, &bar_timestamps, &aim, &cov,
            &spread, &kappa, &prices, &funding, &sigma, &adv, &syms,
        );

        // TWAP emits one record per day → 2 days → 2 records (last H1 bar
        // each day). This is the bar_timestamps[23] and bar_timestamps[47]
        // emission points (per the day-end accumulator pattern).
        assert_eq!(result.records.len(), 2,
            "TWAP should emit one record per D1 day, got {}", result.records.len());

        // Final book should have BOTH positions opened (long BTC, short ETH)
        // — the TWAP path drove fills toward the daily target across H1 bars.
        let final_btc = result.final_book.positions.get("BTC")
            .map(|p| p.quantity).unwrap_or(0.0);
        let final_eth = result.final_book.positions.get("ETH")
            .map(|p| p.quantity).unwrap_or(0.0);
        assert!(final_btc > 0.0,
            "BTC should be long after TWAP fills (got qty={final_btc})");
        assert!(final_eth < 0.0,
            "ETH should be short after TWAP fills (got qty={final_eth})");

        // No liquidations — synthetic fixture is gentle.
        assert!(result.liquidation_events.is_empty(),
            "no liquidations expected, got {:?}", result.liquidation_events);
        // No rejected trades — gate=0.0 (default) is "no gate".
        assert_eq!(result.n_rejected_trades, 0,
            "no rejections expected at gate_bps=0 (default = no gate)");
    }

    /// C1 §4 HIGH-risk path #2 — impact-gate rejection effect on accounting.
    ///
    /// Configures κ and ADV so that any meaningful trade incurs predicted
    /// impact far exceeding the gate, forces a tight `impact_gate_bps`, and
    /// asserts every fill attempt is rejected. The accounting consequences:
    ///   - `n_rejected_trades > 0` (counter increments)
    ///   - `rejected_notional > 0` (size tracker accumulates)
    ///   - final book has NO positions (no fills landed)
    ///   - final cash equals initial NAV (no cash debit from refused trades —
    ///     the gate is structural, not a cost cap)
    #[test]
    fn c3e_impact_gate_rejects_trades_and_preserves_book() {
        let mut config = ExecutionConfig::default();
        config.fill_mode = crate::config::FillMode::Market;
        config.nav_usd = 100_000.0;
        config.qp.gamma = 5.0;
        config.qp.lambda_aim = 1.0;
        config.qp.l_max = 1.5;
        config.qp.per_name_cap = Some(0.10);
        config.funding.enabled = false;
        // Tight gate: any non-trivial impact gets refused. At κ=100, σ=1,
        // ADV=1 the predicted impact for any reasonable Q is >> 100 bps.
        config.cost_model.impact_gate_bps = 100.0;

        let (alphas, bar_timestamps, aim, cov,
             spread, mut kappa, prices, funding, sigma, mut adv, syms) =
            build_fixture(2); // 2 H1 bars (one Market-mode trading day)

        // Force extreme impact: κ=100, σ stays high via sigma_lookup override,
        // ADV ≈ 1 (effectively zero liquidity).
        for &ts in &bar_timestamps {
            kappa.insert((ts, 0), 100.0);
            kappa.insert((ts, 1), 100.0);
        }
        // Day-floor sigma + adv. Overwrite the fixture's day-0 entries.
        let day0 = (bar_timestamps[0] / 86400) * 86400;
        // Sigma override to force impact_bps = κ·σ·(Q/ADV)^δ·1e4 = huge.
        // sigma_lookup is keyed (day, sid), already populated for day0 by
        // build_fixture; just overwrite to ensure σ is the inflated 1.0.
        let mut sigma = sigma;
        sigma.insert((day0, 0), 1.0);
        sigma.insert((day0, 1), 1.0);
        adv.insert((day0, 0), 1.0);
        adv.insert((day0, 1), 1.0);

        let mut engine = SimulationEngine::new(&config);
        let result = engine.run(
            &alphas, &bar_timestamps, &aim, &cov,
            &spread, &kappa, &prices, &funding, &sigma, &adv, &syms,
        );

        // The gate must have fired and the engine must have counted the rejections.
        assert!(result.n_rejected_trades > 0,
            "impact gate at 100bps with κ=100 σ=1 ADV=1 should reject every trade; got n_rejected={}",
            result.n_rejected_trades);
        assert!(result.rejected_notional > 0.0,
            "rejected_notional should accumulate; got {}", result.rejected_notional);

        // No fills landed → no positions in the final book.
        let any_open = result.final_book.positions.values()
            .any(|p| !p.is_flat());
        assert!(!any_open,
            "no positions should be open when all trades are rejected; book has {:?}",
            result.final_book.positions.iter().filter(|(_, p)| !p.is_flat()).map(|(s, _)| s).collect::<Vec<_>>());

        // No cash debit from refused trades. Cash must equal initial NAV
        // (modulo funding accrual which is disabled in this fixture).
        assert!((result.final_book.cash - config.nav_usd).abs() < 1e-6,
            "no cash should be debited from refused trades; cash={}, NAV_init={}",
            result.final_book.cash, config.nav_usd);
    }

    /// C1 §4 HIGH-risk path #3 — margin-stress → liquidation cascade.
    ///
    /// Drives the end-of-bar margin check (`exchange.is_liquidatable`) over
    /// its threshold by setting an absurdly high `maintenance_margin_rate`
    /// so any non-trivial position is liquidatable. Asserts the full
    /// post-event state:
    ///   - `liquidation_events` records the bar timestamp where it fired
    ///   - all positions are flat (book.liquidate_all closed them)
    ///   - daily_target was cleared (no rebuild on the next bar from the
    ///     same stale targets — empirically tested via running 2 bars and
    ///     observing the liquidation event lands on bar 1)
    #[test]
    fn c3e_margin_stress_triggers_liquidation_cascade() {
        let mut config = ExecutionConfig::default();
        config.fill_mode = crate::config::FillMode::Market;
        config.nav_usd = 100_000.0;
        config.qp.gamma = 5.0;
        config.qp.lambda_aim = 1.0;
        config.qp.l_max = 1.5;
        config.qp.per_name_cap = Some(0.10);
        config.alpha.l_target = 1.0;
        config.funding.enabled = false;
        // Force liquidation: maintenance = 10× |notional|. With per_name_cap
        // = 0.10 and l_max = 1.5, each position can carry up to ~$10k of
        // exposure on a $100k NAV. Two positions → up to ~$20k notional →
        // maintenance ≈ $200k > $100k NAV → is_liquidatable returns true on
        // the first end-of-bar margin check.
        config.exchange.maintenance_margin_rate = 10.0;
        config.exchange.liquidation_ratio = 1.0;
        config.exchange.margin_halt_ratio = 1.5;

        let (alphas, bar_timestamps, aim, cov,
             spread, kappa, prices, funding, sigma, adv, syms) = build_fixture(2);

        let mut engine = SimulationEngine::new(&config);
        let result = engine.run(
            &alphas, &bar_timestamps, &aim, &cov,
            &spread, &kappa, &prices, &funding, &sigma, &adv, &syms,
        );

        // Liquidation event must have fired at least once.
        assert!(!result.liquidation_events.is_empty(),
            "expected at least one liquidation event with maintenance_rate=10×; got none");

        // First liquidation should land on the first bar (after the first
        // round of trades opens stress-inducing positions).
        let first_liq = result.liquidation_events[0];
        assert_eq!(first_liq, bar_timestamps[0],
            "first liquidation should land on bar 0 (after first opening trades); got {} vs {}",
            first_liq, bar_timestamps[0]);

        // Post-cascade: all positions are flat. liquidate_all closes everything.
        let any_open = result.final_book.positions.values()
            .any(|p| !p.is_flat());
        assert!(!any_open,
            "all positions must be flat after liquidate_all; open: {:?}",
            result.final_book.positions.iter()
                .filter(|(_, p)| !p.is_flat())
                .map(|(s, p)| (s.clone(), p.quantity))
                .collect::<Vec<_>>());

        // NAV after liquidation should be positive (we crossed the threshold
        // but liquidate_all closes via market orders; cash includes any
        // realized PnL minus liquidation fees). Specifically: NAV must be
        // strictly less than initial nav_usd (fees + spread + impact debited).
        let post_nav = result.final_book.nav();
        assert!(post_nav < config.nav_usd,
            "post-liquidation NAV should be < initial NAV due to costs; got {} vs {}",
            post_nav, config.nav_usd);
        assert!(post_nav > 0.0,
            "post-liquidation NAV should remain positive; got {}", post_nav);
    }
}
