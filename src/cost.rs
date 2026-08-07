//! Cost model: fees, tiered spread, sqrt-impact (NAV-aware kappa), funding.
//!
//! Ports `cost_primitives.py` (EWMA ADV, realized vol, liquidity tiers)
//! and `optimize.py` (build_c_lin_per_symbol, build_kappa_per_symbol).
//!
//! Funding model: per-asset EWMA of hourly rates + AR(1) closed-form
//! expected cumulative cost over holding period. Subtracted from alpha
//! before QP optimization (Koijen et al. 2018 "Carry" framework).

use std::collections::{HashMap, VecDeque};

use crate::config::FundingConfig;

/// Per-(timestamp, symbol) cost primitives.
#[derive(Debug, Clone)]
pub struct CostPrimitives {
    pub timestamp: i64,
    pub symbol: String,
    pub ewma_adv_usd: f64,
    pub realized_vol_daily: f64,
    pub c_lin: f64,  // fee + 0.5 * spread, as fraction (from ML model)
    pub kappa: f64,  // sqrt-impact coefficient (from ML model)
}

/// Compute EWMA ADV (average daily volume in USD).
/// `volumes` and `closes` are per-bar values for one asset.
pub fn ewma_adv(volumes: &[f64], closes: &[f64], halflife: usize) -> Vec<f64> {
    let alpha = 1.0 - (-2.0_f64.ln() / halflife as f64).exp();
    let n = volumes.len();
    let mut adv = vec![0.0; n];
    if n == 0 {
        return adv;
    }
    let mut ewma = volumes[0] * closes[0];
    adv[0] = ewma;
    for i in 1..n {
        let dollar_vol = volumes[i] * closes[i];
        if dollar_vol.is_finite() {
            ewma = alpha * dollar_vol + (1.0 - alpha) * ewma;
        }
        adv[i] = ewma;
    }
    adv
}

/// Compute EWMA realized volatility (daily).
/// `log_returns` are per-bar log returns for one asset.
pub fn ewma_realized_vol(log_returns: &[f64], halflife: usize) -> Vec<f64> {
    let alpha = 1.0 - (-2.0_f64.ln() / halflife as f64).exp();
    let n = log_returns.len();
    let mut vol = vec![0.0; n];
    if n == 0 {
        return vol;
    }
    let mut ewma_var = 0.0;
    for i in 0..n {
        let r = log_returns[i];
        if r.is_finite() {
            ewma_var = alpha * r * r + (1.0 - alpha) * ewma_var;
        }
        vol[i] = ewma_var.sqrt();
    }
    vol
}

/// Aggregate hourly funding rates to daily.
/// Input: (timestamp_ms, symbol, funding_rate) tuples.
/// Output: (timestamp_seconds_day_start, symbol, daily_rate) tuples.
pub fn aggregate_funding_daily(
    hourly: &[(i64, String, f64)],
) -> Vec<(i64, String, f64)> {
    // Group by (day, symbol), sum rates
    let mut daily: HashMap<(i64, String), f64> = HashMap::new();
    for (ts_ms, sym, rate) in hourly {
        let day_start_s = (*ts_ms / 1000 / 86400) * 86400; // floor to day, seconds
        *daily.entry((day_start_s, sym.clone())).or_default() += rate;
    }
    let mut result: Vec<_> = daily.into_iter().map(|((ts, sym), rate)| (ts, sym, rate)).collect();
    result.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    result
}

// ---------------------------------------------------------------------------
// Funding cost model
// ---------------------------------------------------------------------------

/// Per-asset funding state tracked by the FundingModel.
#[derive(Debug, Clone)]
pub struct AssetFundingState {
    /// EWMA of hourly funding rates (current rate estimate, f₀).
    pub ewma_rate: f64,
    /// Rolling window of recent hourly rates for long-run mean (μ) estimation.
    history: VecDeque<f64>,
    /// Cached long-run mean (recomputed on each update).
    pub long_run_mean: f64,
}

/// Funding cost model: per-asset EWMA + AR(1) expected cumulative cost.
///
/// Usage:
///   1. Call `update()` each hour with observed funding rates per asset.
///   2. Call `expected_alpha_adjustment()` at D1 optimization time to get
///      the per-asset funding cost to subtract from alpha.
///
/// The AR(1) closed form for expected cumulative funding over H hours:
///   E[C_H] = H·μ + (f₀ - μ) · ρ(1 - ρ^H) / (1 - ρ)
///
/// where f₀ = EWMA current rate, μ = long-run mean, ρ = hourly autocorrelation.
/// Positive value = longs pay, subtract from long alpha / add to short alpha.
#[derive(Debug, Clone)]
pub struct FundingModel {
    config: FundingConfig,
    ewma_alpha: f64,
    states: HashMap<String, AssetFundingState>,
}

impl FundingModel {
    pub fn new(config: &FundingConfig) -> Self {
        let ewma_alpha = 1.0 - (-2.0_f64.ln() / config.ewma_halflife_hours as f64).exp();
        Self {
            config: config.clone(),
            ewma_alpha,
            states: HashMap::new(),
        }
    }

    /// Ingest one hourly funding observation for a single asset.
    pub fn update_asset(&mut self, symbol: &str, hourly_rate: f64) {
        if !hourly_rate.is_finite() {
            return;
        }
        let max_history = self.config.long_run_mean_window_hours;
        let state = self.states.entry(symbol.to_string()).or_insert_with(|| {
            AssetFundingState {
                ewma_rate: hourly_rate,
                history: VecDeque::with_capacity(max_history + 1),
                long_run_mean: hourly_rate,
            }
        });

        // Update EWMA
        state.ewma_rate = self.ewma_alpha * hourly_rate + (1.0 - self.ewma_alpha) * state.ewma_rate;

        // Update rolling history for long-run mean
        state.history.push_back(hourly_rate);
        if state.history.len() > max_history {
            state.history.pop_front();
        }

        // Recompute long-run mean
        if !state.history.is_empty() {
            let sum: f64 = state.history.iter().sum();
            state.long_run_mean = sum / state.history.len() as f64;
        }
    }

    /// Batch update: ingest hourly rates for multiple assets at once.
    pub fn update_batch(&mut self, rates: &[(String, f64)]) {
        for (symbol, rate) in rates {
            self.update_asset(symbol, *rate);
        }
    }

    /// Expected cumulative funding cost over the holding period for one asset.
    ///
    /// Returns a signed value: positive = longs pay, negative = longs receive.
    /// Subtract this from alpha before QP optimization.
    pub fn expected_cumulative_cost(&self, symbol: &str) -> f64 {
        let state = match self.states.get(symbol) {
            Some(s) => s,
            None => return 0.0,
        };
        Self::ar1_cumulative(
            state.ewma_rate,
            state.long_run_mean,
            self.config.default_rho,
            self.config.holding_hours,
        )
    }

    /// AR(1) closed-form expected cumulative funding:
    ///   E[C_H] = H·μ + (f₀ - μ) · ρ(1 - ρ^H) / (1 - ρ)
    fn ar1_cumulative(f0: f64, mu: f64, rho: f64, holding_hours: usize) -> f64 {
        let h = holding_hours as f64;
        if (rho - 1.0).abs() < 1e-12 {
            // No mean-reversion: cost = f₀ × H
            return f0 * h;
        }
        let rho_h = rho.powi(holding_hours as i32);
        let decay_sum = rho * (1.0 - rho_h) / (1.0 - rho);
        h * mu + (f0 - mu) * decay_sum
    }

    /// Compute alpha adjustment vector for a list of symbols.
    ///
    /// Returns a Vec aligned with the input symbols. Each value is the
    /// expected cumulative funding cost to subtract from that symbol's alpha.
    pub fn expected_alpha_adjustment(&self, symbols: &[String]) -> Vec<f64> {
        symbols
            .iter()
            .map(|s| self.expected_cumulative_cost(s))
            .collect()
    }

    /// Get the current state for an asset (for diagnostics/logging).
    pub fn state(&self, symbol: &str) -> Option<&AssetFundingState> {
        self.states.get(symbol)
    }

    /// Number of assets currently tracked.
    pub fn n_assets(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_adv() {
        let volumes = vec![100.0, 200.0, 150.0, 300.0, 250.0];
        let closes = vec![50.0, 55.0, 52.0, 60.0, 58.0];
        let adv = ewma_adv(&volumes, &closes, 3);
        assert_eq!(adv.len(), 5);
        assert!(adv[4] > 0.0);
    }

    #[test]
    fn test_funding_aggregation() {
        let hourly = vec![
            (1000 * 3600 * 0, "BTC".to_string(), 0.001),
            (1000 * 3600 * 1, "BTC".to_string(), 0.002),
            (1000 * 3600 * 24, "BTC".to_string(), 0.003),
        ];
        let daily = aggregate_funding_daily(&hourly);
        // Day 0: 0.001 + 0.002 = 0.003, Day 1: 0.003
        assert_eq!(daily.len(), 2);
        assert!((daily[0].2 - 0.003).abs() < 1e-10);
        assert!((daily[1].2 - 0.003).abs() < 1e-10);
    }

    // --- FundingModel tests ---

    fn test_funding_config() -> FundingConfig {
        FundingConfig {
            enabled: true,
            ewma_halflife_hours: 24,
            long_run_mean_window_hours: 720,
            default_rho: 0.89,
            holding_hours: 24,
            use_in_alpha: false,
            alpha_demean: true,
            alpha_scale: 1.0,
        }
    }

    #[test]
    fn test_funding_model_single_update() {
        let cfg = test_funding_config();
        let mut model = FundingModel::new(&cfg);
        model.update_asset("BTC", 0.0001);

        let state = model.state("BTC").unwrap();
        assert!((state.ewma_rate - 0.0001).abs() < 1e-12);
        assert!((state.long_run_mean - 0.0001).abs() < 1e-12);
        assert_eq!(model.n_assets(), 1);
    }

    #[test]
    fn test_funding_model_ewma_decay() {
        let cfg = test_funding_config();
        let mut model = FundingModel::new(&cfg);

        // Feed a spike then zeros
        model.update_asset("BTC", 0.001);
        for _ in 0..48 {
            model.update_asset("BTC", 0.0);
        }

        let state = model.state("BTC").unwrap();
        // After 48 hours of zero with halflife=24h, EWMA should be ~25% of spike
        // 0.001 * (1 - alpha)^48 where alpha = 1 - exp(-ln2/24) ≈ 0.0285
        // (1 - 0.0285)^48 ≈ 0.253
        assert!(state.ewma_rate < 0.0004, "EWMA should have decayed: {}", state.ewma_rate);
        assert!(state.ewma_rate > 0.0001, "EWMA should retain some memory: {}", state.ewma_rate);
    }

    #[test]
    fn test_funding_model_long_run_mean() {
        let cfg = test_funding_config();
        let mut model = FundingModel::new(&cfg);

        // Feed constant rate for 100 hours
        for _ in 0..100 {
            model.update_asset("BTC", 0.0001);
        }
        let state = model.state("BTC").unwrap();
        assert!((state.long_run_mean - 0.0001).abs() < 1e-12);

        // Now feed higher rate for 100 hours
        for _ in 0..100 {
            model.update_asset("BTC", 0.0003);
        }
        let state = model.state("BTC").unwrap();
        // Mean should be between 0.0001 and 0.0003 (200 obs total, 100 at each)
        assert!(state.long_run_mean > 0.00015);
        assert!(state.long_run_mean < 0.00025);
    }

    #[test]
    fn test_ar1_cumulative_at_mean() {
        // When f₀ = μ, the transient term vanishes: E[C_H] = H × μ
        let cost = FundingModel::ar1_cumulative(0.0001, 0.0001, 0.89, 24);
        let expected = 24.0 * 0.0001;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn test_ar1_cumulative_spike() {
        // f₀ = 0.001 (10x above mean), μ = 0.0001, ρ = 0.89, H = 24
        let cost = FundingModel::ar1_cumulative(0.001, 0.0001, 0.89, 24);
        let naive = 24.0 * 0.001; // = 0.024

        // AR(1) should be less than naive (rate mean-reverts toward μ)
        assert!(cost < naive, "AR(1) cost {} should be < naive {}", cost, naive);

        // But more than H × μ (current rate is above mean)
        let at_mean = 24.0 * 0.0001; // = 0.0024
        assert!(cost > at_mean, "AR(1) cost {} should be > at-mean {}", cost, at_mean);

        // Verify exact value:
        // E[C] = 24 * 0.0001 + (0.001 - 0.0001) * 0.89 * (1 - 0.89^24) / (1 - 0.89)
        //      = 0.0024 + 0.0009 * 0.89 * (1 - 0.0659) / 0.11
        //      = 0.0024 + 0.0009 * 0.89 * 8.4918...
        //      = 0.0024 + 0.006802...
        //      = 0.009202...
        assert!((cost - 0.0092).abs() < 0.001, "cost = {}", cost);
    }

    #[test]
    fn test_ar1_cumulative_rho_one() {
        // ρ = 1.0 (no mean-reversion): should equal f₀ × H
        let cost = FundingModel::ar1_cumulative(0.001, 0.0001, 1.0, 24);
        assert!((cost - 0.024).abs() < 1e-10);
    }

    #[test]
    fn test_ar1_cumulative_negative_funding() {
        // Negative funding (shorts pay). f₀ = -0.0002, μ = -0.0001
        let cost = FundingModel::ar1_cumulative(-0.0002, -0.0001, 0.89, 24);
        // Should be negative (longs receive)
        assert!(cost < 0.0, "cost should be negative for negative funding: {}", cost);
    }

    #[test]
    fn test_expected_alpha_adjustment() {
        let cfg = test_funding_config();
        let mut model = FundingModel::new(&cfg);

        // Feed steady rates
        for _ in 0..100 {
            model.update_asset("BTC", 0.0001);
            model.update_asset("ETH", 0.0003);
            model.update_asset("DOGE", -0.0001);
        }

        let symbols: Vec<String> = vec!["BTC".into(), "ETH".into(), "DOGE".into(), "UNKNOWN".into()];
        let adj = model.expected_alpha_adjustment(&symbols);

        assert_eq!(adj.len(), 4);
        // BTC: positive funding → positive adjustment (subtract from long alpha)
        assert!(adj[0] > 0.0);
        // ETH: higher funding → larger adjustment
        assert!(adj[1] > adj[0]);
        // DOGE: negative funding → negative adjustment (adds to long alpha)
        assert!(adj[2] < 0.0);
        // UNKNOWN: no data → 0
        assert!((adj[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_funding_model_nan_ignored() {
        let cfg = test_funding_config();
        let mut model = FundingModel::new(&cfg);
        model.update_asset("BTC", 0.0001);
        model.update_asset("BTC", f64::NAN);

        let state = model.state("BTC").unwrap();
        // NaN should be ignored, EWMA unchanged
        assert!((state.ewma_rate - 0.0001).abs() < 1e-12);
    }

    #[test]
    fn test_funding_model_history_window_cap() {
        let cfg = FundingConfig {
            long_run_mean_window_hours: 10,
            ..test_funding_config()
        };
        let mut model = FundingModel::new(&cfg);

        // Feed 20 observations
        for i in 0..20 {
            model.update_asset("BTC", i as f64 * 0.0001);
        }

        let state = model.state("BTC").unwrap();
        // History should only have the last 10
        assert_eq!(state.history.len(), 10);
        // Long-run mean should be average of last 10 values (10..19 * 0.0001)
        // = (10+11+12+13+14+15+16+17+18+19) * 0.0001 / 10 = 14.5 * 0.0001
        assert!((state.long_run_mean - 0.00145).abs() < 1e-8);
    }
}
