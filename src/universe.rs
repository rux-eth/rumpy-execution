//! Tradable universe filter with hysteresis.
//!
//! State machine per symbol: Excluded → Active → ExitOnly → Excluded.
//! Prevents churn by requiring consecutive days meeting/failing criteria.
//!
//! Entry: ALL criteria hold for `entry_days` consecutive days.
//! Exit: ANY criterion fails for `exit_days` consecutive days.
//!
//! Runs identically in backtest and live modes (same code, different data source).

use std::collections::HashMap;

use serde::Deserialize;

use crate::pipeline::WeightOutput;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Universe filter configuration. All thresholds are configurable.
#[derive(Debug, Clone, Deserialize)]
pub struct UniverseConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Consecutive days meeting ALL criteria to enter.
    #[serde(default = "default_entry_days")]
    pub entry_days: usize,
    /// Consecutive days failing ANY criterion to exit.
    #[serde(default = "default_exit_days")]
    pub exit_days: usize,
    /// Minimum EWMA ADV in USD.
    #[serde(default = "default_min_adv")]
    pub min_adv_usd: f64,
    /// Maximum spread in bps.
    #[serde(default = "default_max_spread_bps")]
    pub max_spread_bps: f64,
    /// Minimum book depth within 1% of mid, in USD.
    #[serde(default = "default_min_depth")]
    pub min_depth_1pct_usd: f64,
    /// Minimum open interest in USD.
    #[serde(default = "default_min_oi")]
    pub min_oi_usd: f64,
}

fn default_entry_days() -> usize { 5 }
fn default_exit_days() -> usize { 3 }
fn default_min_adv() -> f64 { 500_000.0 }
fn default_max_spread_bps() -> f64 { 30.0 }
fn default_min_depth() -> f64 { 100_000.0 }
fn default_min_oi() -> f64 { 2_000_000.0 }

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            entry_days: 5,
            exit_days: 3,
            min_adv_usd: 500_000.0,
            max_spread_bps: 30.0,
            min_depth_1pct_usd: 100_000.0,
            min_oi_usd: 2_000_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Universe membership state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniverseState {
    /// Not in tradable universe. Cannot open new positions.
    Excluded,
    /// In tradable universe. Full trading allowed.
    Active,
    /// Failing criteria, winding down. Reduce-only (no new/increased positions).
    ExitOnly,
}

/// Per-symbol universe tracking state.
#[derive(Debug, Clone)]
struct SymbolState {
    state: UniverseState,
    /// Consecutive days meeting ALL entry criteria (while Excluded).
    qualifying_streak: usize,
    /// Consecutive days failing ANY exit criterion (while Active).
    failing_streak: usize,
}

impl Default for SymbolState {
    fn default() -> Self {
        Self {
            state: UniverseState::Excluded,
            qualifying_streak: 0,
            failing_streak: 0,
        }
    }
}

/// Per-symbol, per-bar market metrics for universe evaluation.
#[derive(Debug, Clone)]
pub struct UniverseMetrics {
    pub adv_usd: f64,
    pub spread_bps: f64,
    pub depth_1pct_usd: f64,
    pub oi_usd: f64,
}

// ---------------------------------------------------------------------------
// Universe filter
// ---------------------------------------------------------------------------

/// Hysteresis-based universe filter.
pub struct UniverseFilter {
    config: UniverseConfig,
    states: HashMap<String, SymbolState>,
}

impl UniverseFilter {
    pub fn new(config: &UniverseConfig) -> Self {
        Self {
            config: config.clone(),
            states: HashMap::new(),
        }
    }

    /// Check if a symbol meets all entry criteria.
    fn meets_criteria(&self, m: &UniverseMetrics) -> bool {
        m.adv_usd >= self.config.min_adv_usd
            && m.spread_bps <= self.config.max_spread_bps
            && m.depth_1pct_usd >= self.config.min_depth_1pct_usd
            && m.oi_usd >= self.config.min_oi_usd
    }

    /// Update one symbol's state with today's metrics. Returns new state.
    pub fn update(&mut self, symbol: &str, metrics: &UniverseMetrics) -> UniverseState {
        let qualifies = self.meets_criteria(metrics);
        let entry = self.states.entry(symbol.to_string()).or_default();

        match entry.state {
            UniverseState::Excluded => {
                if qualifies {
                    entry.qualifying_streak += 1;
                    if entry.qualifying_streak >= self.config.entry_days {
                        entry.state = UniverseState::Active;
                        entry.qualifying_streak = 0;
                        entry.failing_streak = 0;
                    }
                } else {
                    entry.qualifying_streak = 0;
                }
            }
            UniverseState::Active => {
                if qualifies {
                    entry.failing_streak = 0;
                } else {
                    entry.failing_streak += 1;
                    if entry.failing_streak >= self.config.exit_days {
                        entry.state = UniverseState::ExitOnly;
                        entry.failing_streak = 0;
                    }
                }
            }
            UniverseState::ExitOnly => {
                // Stay in ExitOnly until position is fully closed.
                // Transition to Excluded happens in filter_weights when position is zero.
                // But if criteria are met again, can re-enter Active.
                if qualifies {
                    entry.qualifying_streak += 1;
                    if entry.qualifying_streak >= self.config.entry_days {
                        entry.state = UniverseState::Active;
                        entry.qualifying_streak = 0;
                    }
                } else {
                    entry.qualifying_streak = 0;
                }
            }
        }

        entry.state
    }

    /// Batch update for all symbols at one timestamp.
    pub fn update_batch(&mut self, metrics: &[(String, UniverseMetrics)]) {
        for (sym, m) in metrics {
            self.update(sym, m);
        }
    }

    /// Get current state for a symbol. Excluded if unknown.
    pub fn state(&self, symbol: &str) -> UniverseState {
        self.states
            .get(symbol)
            .map(|s| s.state)
            .unwrap_or(UniverseState::Excluded)
    }

    /// Number of symbols in each state.
    pub fn state_counts(&self) -> (usize, usize, usize) {
        let mut active = 0;
        let mut exit_only = 0;
        let mut excluded = 0;
        for s in self.states.values() {
            match s.state {
                UniverseState::Active => active += 1,
                UniverseState::ExitOnly => exit_only += 1,
                UniverseState::Excluded => excluded += 1,
            }
        }
        (active, exit_only, excluded)
    }

    /// Filter weights based on universe state.
    ///
    /// - Active: no change.
    /// - ExitOnly: only allow reducing position (toward zero).
    /// - Excluded: zero out weight.
    ///
    /// `prev_weights`: previous bar's weights for ExitOnly reduce-only logic.
    pub fn filter_weights(
        &mut self,
        weights: &mut [WeightOutput],
        prev_weights: &HashMap<String, f64>,
    ) {
        for w in weights.iter_mut() {
            match self.state(&w.symbol) {
                UniverseState::Active => {
                    // Full trading allowed
                }
                UniverseState::ExitOnly => {
                    let prev = prev_weights.get(&w.symbol).copied().unwrap_or(0.0);
                    if prev.abs() < 1e-12 {
                        // No existing position → zero out
                        w.weight_final = 0.0;
                    } else if prev > 0.0 {
                        // Long position: can only reduce (weight must be between 0 and prev)
                        w.weight_final = w.weight_final.clamp(0.0, prev);
                    } else {
                        // Short position: can only reduce (weight must be between prev and 0)
                        w.weight_final = w.weight_final.clamp(prev, 0.0);
                    }
                    // If position is now zero, transition to Excluded
                    if w.weight_final.abs() < 1e-12 {
                        if let Some(entry) = self.states.get_mut(&w.symbol) {
                            entry.state = UniverseState::Excluded;
                        }
                    }
                }
                UniverseState::Excluded => {
                    w.weight_final = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_metrics() -> UniverseMetrics {
        UniverseMetrics {
            adv_usd: 1_000_000.0,
            spread_bps: 10.0,
            depth_1pct_usd: 200_000.0,
            oi_usd: 5_000_000.0,
        }
    }

    fn bad_metrics() -> UniverseMetrics {
        UniverseMetrics {
            adv_usd: 100_000.0, // below 500K threshold
            spread_bps: 50.0,   // above 30 bps threshold
            depth_1pct_usd: 30_000.0,
            oi_usd: 500_000.0,
        }
    }

    #[test]
    fn test_excluded_to_active_after_entry_days() {
        let config = UniverseConfig { enabled: true, entry_days: 5, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // 4 qualifying days → still Excluded
        for _ in 0..4 {
            let state = filter.update("BTC", &good_metrics());
            assert_eq!(state, UniverseState::Excluded);
        }

        // 5th qualifying day → Active
        let state = filter.update("BTC", &good_metrics());
        assert_eq!(state, UniverseState::Active);
    }

    #[test]
    fn test_qualifying_streak_resets_on_fail() {
        let config = UniverseConfig { enabled: true, entry_days: 5, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // 3 qualifying days, then 1 fail → streak resets
        for _ in 0..3 {
            filter.update("BTC", &good_metrics());
        }
        filter.update("BTC", &bad_metrics());

        // Need 5 more qualifying days now
        for _ in 0..4 {
            let state = filter.update("BTC", &good_metrics());
            assert_eq!(state, UniverseState::Excluded);
        }
        let state = filter.update("BTC", &good_metrics());
        assert_eq!(state, UniverseState::Active);
    }

    #[test]
    fn test_active_to_exit_only_after_exit_days() {
        let config = UniverseConfig { enabled: true, entry_days: 3, exit_days: 3, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // Enter
        for _ in 0..3 {
            filter.update("BTC", &good_metrics());
        }
        assert_eq!(filter.state("BTC"), UniverseState::Active);

        // 2 failing days → still Active
        for _ in 0..2 {
            let state = filter.update("BTC", &bad_metrics());
            assert_eq!(state, UniverseState::Active);
        }

        // 3rd failing day → ExitOnly
        let state = filter.update("BTC", &bad_metrics());
        assert_eq!(state, UniverseState::ExitOnly);
    }

    #[test]
    fn test_hysteresis_no_churn() {
        let config = UniverseConfig { enabled: true, entry_days: 3, exit_days: 3, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // Enter
        for _ in 0..3 {
            filter.update("BTC", &good_metrics());
        }
        assert_eq!(filter.state("BTC"), UniverseState::Active);

        // Fail 2 days then recover → stays Active (no churn)
        filter.update("BTC", &bad_metrics());
        filter.update("BTC", &bad_metrics());
        let state = filter.update("BTC", &good_metrics()); // recovery
        assert_eq!(state, UniverseState::Active);
    }

    #[test]
    fn test_failing_streak_resets_on_qualify() {
        let config = UniverseConfig { enabled: true, entry_days: 3, exit_days: 3, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // Enter
        for _ in 0..3 {
            filter.update("BTC", &good_metrics());
        }

        // 2 fails, 1 good, 2 fails → should NOT trigger exit (streak was reset)
        filter.update("BTC", &bad_metrics());
        filter.update("BTC", &bad_metrics());
        filter.update("BTC", &good_metrics()); // resets failing_streak
        filter.update("BTC", &bad_metrics());
        let state = filter.update("BTC", &bad_metrics());
        assert_eq!(state, UniverseState::Active); // only 2 consecutive fails
    }

    #[test]
    fn test_filter_weights_excluded_zeroed() {
        let config = UniverseConfig { enabled: true, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);
        // BTC is Excluded (never qualified)

        let mut weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.5 },
        ];
        filter.filter_weights(&mut weights, &HashMap::new());
        assert!((weights[0].weight_final).abs() < 1e-12, "Excluded → zero weight");
    }

    #[test]
    fn test_filter_weights_exit_only_reduce() {
        let config = UniverseConfig { enabled: true, entry_days: 2, exit_days: 2, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // Enter
        filter.update("BTC", &good_metrics());
        filter.update("BTC", &good_metrics());
        assert_eq!(filter.state("BTC"), UniverseState::Active);

        // Exit
        filter.update("BTC", &bad_metrics());
        filter.update("BTC", &bad_metrics());
        assert_eq!(filter.state("BTC"), UniverseState::ExitOnly);

        // Previous weight was 0.5 (long). New weight wants 0.8 → clamped to 0.5 (reduce only).
        let mut weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.8, weight_final: 0.8 },
        ];
        let mut prev = HashMap::new();
        prev.insert("BTC".into(), 0.5);
        filter.filter_weights(&mut weights, &prev);
        assert!((weights[0].weight_final - 0.5).abs() < 1e-10, "ExitOnly: can't increase beyond prev");

        // Reducing is allowed: new weight 0.2 < prev 0.5
        let mut weights2 = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.2, weight_final: 0.2 },
        ];
        filter.filter_weights(&mut weights2, &prev);
        assert!((weights2[0].weight_final - 0.2).abs() < 1e-10, "ExitOnly: reducing is ok");
    }

    #[test]
    fn test_state_counts() {
        let config = UniverseConfig { enabled: true, entry_days: 2, exit_days: 2, ..Default::default() };
        let mut filter = UniverseFilter::new(&config);

        // Enter BTC and ETH
        for _ in 0..2 {
            filter.update("BTC", &good_metrics());
            filter.update("ETH", &good_metrics());
        }
        // Exit BTC
        filter.update("BTC", &bad_metrics());
        filter.update("BTC", &bad_metrics());
        // SOL never seen (Excluded)
        filter.update("SOL", &bad_metrics());

        let (active, exit_only, excluded) = filter.state_counts();
        assert_eq!(active, 1);     // ETH
        assert_eq!(exit_only, 1);  // BTC
        assert_eq!(excluded, 1);   // SOL
    }

    #[test]
    fn test_unknown_symbol_is_excluded() {
        let config = UniverseConfig::default();
        let filter = UniverseFilter::new(&config);
        assert_eq!(filter.state("UNKNOWN"), UniverseState::Excluded);
    }
}
