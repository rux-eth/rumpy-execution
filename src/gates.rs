//! Pre-trade gates: hard checks before every trade.
//!
//! Per asset, per hourly slice. Runs identically in backtest and live modes.
//! Gate failure defers the trade — it doesn't reject the signal permanently.

use std::collections::HashMap;

use serde::Deserialize;

use crate::pipeline::WeightOutput;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Pre-trade gate configuration. All thresholds are configurable.
#[derive(Debug, Clone, Deserialize)]
pub struct GateConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Max spread in bps for a trade to proceed.
    #[serde(default = "default_max_spread")]
    pub max_spread_bps: f64,
    /// Min book depth within 1% of mid, in USD. Trade size must fit.
    #[serde(default = "default_min_depth")]
    pub min_depth_1pct_usd: f64,
    /// Max participation rate (trade_notional / hourly_volume).
    #[serde(default = "default_max_participation")]
    pub max_participation_rate: f64,
    /// Max estimated exit cost as fraction of position value.
    #[serde(default = "default_max_exit_cost")]
    pub max_exit_cost_pct: f64,
    /// Min margin headroom as fraction of account equity.
    #[serde(default = "default_min_margin")]
    pub min_margin_headroom: f64,
}

fn default_max_spread() -> f64 { 50.0 }
fn default_min_depth() -> f64 { 50_000.0 }
fn default_max_participation() -> f64 { 0.15 }
fn default_max_exit_cost() -> f64 { 0.02 }
fn default_min_margin() -> f64 { 0.30 }

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_spread_bps: 50.0,
            min_depth_1pct_usd: 50_000.0,
            max_participation_rate: 0.15,
            max_exit_cost_pct: 0.02,
            min_margin_headroom: 0.30,
        }
    }
}

// ---------------------------------------------------------------------------
// Gate inputs and results
// ---------------------------------------------------------------------------

/// Market data needed for gate checks, per (timestamp, symbol).
#[derive(Debug, Clone)]
pub struct GateInput {
    pub spread_bps: f64,
    pub depth_1pct_usd: f64,
    pub hourly_volume_usd: f64,
    pub exit_cost_pct: f64,
    pub margin_headroom: f64,
}

/// Result of a single gate check.
#[derive(Debug, Clone)]
pub enum GateResult {
    Pass,
    Reject(GateReason),
}

/// Reason a gate rejected a trade.
#[derive(Debug, Clone)]
pub enum GateReason {
    SpreadTooWide { spread_bps: f64, max: f64 },
    InsufficientDepth { depth_usd: f64, required: f64 },
    ExcessiveParticipation { rate: f64, max: f64 },
    HighExitCost { cost_pct: f64, max: f64 },
    LowMargin { headroom: f64, min: f64 },
    MissingData,
}

impl std::fmt::Display for GateReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateReason::SpreadTooWide { spread_bps, max } =>
                write!(f, "spread {spread_bps:.1} bps > max {max:.1}"),
            GateReason::InsufficientDepth { depth_usd, required } =>
                write!(f, "depth ${depth_usd:.0} < required ${required:.0}"),
            GateReason::ExcessiveParticipation { rate, max } =>
                write!(f, "participation {:.1}% > max {:.1}%", rate * 100.0, max * 100.0),
            GateReason::HighExitCost { cost_pct, max } =>
                write!(f, "exit cost {:.2}% > max {:.2}%", cost_pct * 100.0, max * 100.0),
            GateReason::LowMargin { headroom, min } =>
                write!(f, "margin {:.1}% < min {:.1}%", headroom * 100.0, min * 100.0),
            GateReason::MissingData =>
                write!(f, "missing gate input data"),
        }
    }
}

/// Record of a gate rejection for diagnostics.
#[derive(Debug, Clone)]
pub struct GateRejection {
    pub timestamp: i64,
    pub symbol: String,
    pub reason: GateReason,
}

// ---------------------------------------------------------------------------
// Gate checker
// ---------------------------------------------------------------------------

/// Pre-trade gate checker.
pub struct PreTradeGate {
    config: GateConfig,
}

impl PreTradeGate {
    pub fn new(config: &GateConfig) -> Self {
        Self { config: config.clone() }
    }

    /// Check all gates for one (timestamp, symbol) trade.
    pub fn check(
        &self,
        input: &GateInput,
        trade_notional_usd: f64,
    ) -> GateResult {
        // 1. Spread check
        if input.spread_bps > self.config.max_spread_bps {
            return GateResult::Reject(GateReason::SpreadTooWide {
                spread_bps: input.spread_bps,
                max: self.config.max_spread_bps,
            });
        }

        // 2. Depth check: trade must fit within available depth
        if input.depth_1pct_usd < trade_notional_usd.min(self.config.min_depth_1pct_usd) {
            return GateResult::Reject(GateReason::InsufficientDepth {
                depth_usd: input.depth_1pct_usd,
                required: trade_notional_usd.min(self.config.min_depth_1pct_usd),
            });
        }

        // 3. Participation check
        if input.hourly_volume_usd > 0.0 {
            let participation = trade_notional_usd / input.hourly_volume_usd;
            if participation > self.config.max_participation_rate {
                return GateResult::Reject(GateReason::ExcessiveParticipation {
                    rate: participation,
                    max: self.config.max_participation_rate,
                });
            }
        }

        // 4. Exit cost check
        if input.exit_cost_pct > self.config.max_exit_cost_pct {
            return GateResult::Reject(GateReason::HighExitCost {
                cost_pct: input.exit_cost_pct,
                max: self.config.max_exit_cost_pct,
            });
        }

        // 5. Margin headroom check
        if input.margin_headroom < self.config.min_margin_headroom {
            return GateResult::Reject(GateReason::LowMargin {
                headroom: input.margin_headroom,
                min: self.config.min_margin_headroom,
            });
        }

        GateResult::Pass
    }

    /// Apply gates to proposed weights. Rejected trades get weight zeroed.
    ///
    /// Returns (filtered weights unchanged in-place, rejection log).
    pub fn apply(
        &self,
        weights: &mut [WeightOutput],
        gate_inputs: &HashMap<(i64, String), GateInput>,
        nav_usd: f64,
        prev_weights: &HashMap<String, f64>,
    ) -> Vec<GateRejection> {
        let mut rejections = Vec::new();

        for w in weights.iter_mut() {
            // Only check trades that change position
            let prev = prev_weights.get(&w.symbol).copied().unwrap_or(0.0);
            let delta = (w.weight_final - prev).abs();
            if delta < 1e-9 {
                continue; // no trade → no gate needed
            }
            let trade_notional = delta * nav_usd;

            let key = (w.timestamp, w.symbol.clone());
            let result = match gate_inputs.get(&key) {
                Some(input) => self.check(input, trade_notional),
                None => {
                    // Missing gate data — conservative: defer trade
                    GateResult::Reject(GateReason::MissingData)
                }
            };

            if let GateResult::Reject(reason) = result {
                rejections.push(GateRejection {
                    timestamp: w.timestamp,
                    symbol: w.symbol.clone(),
                    reason,
                });
                // Defer: keep previous weight (don't execute the trade)
                w.weight_final = prev;
            }
        }

        rejections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_gate_input() -> GateInput {
        GateInput {
            spread_bps: 5.0,
            depth_1pct_usd: 500_000.0,
            hourly_volume_usd: 10_000_000.0,
            exit_cost_pct: 0.005,
            margin_headroom: 0.50,
        }
    }

    #[test]
    fn test_all_gates_pass() {
        let gate = PreTradeGate::new(&GateConfig { enabled: true, ..Default::default() });
        let result = gate.check(&good_gate_input(), 100_000.0);
        assert!(matches!(result, GateResult::Pass));
    }

    #[test]
    fn test_spread_gate_rejects() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            max_spread_bps: 20.0,
            ..Default::default()
        });
        let mut input = good_gate_input();
        input.spread_bps = 25.0;
        let result = gate.check(&input, 100_000.0);
        assert!(matches!(result, GateResult::Reject(GateReason::SpreadTooWide { .. })));
    }

    #[test]
    fn test_depth_gate_rejects() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            min_depth_1pct_usd: 200_000.0,
            ..Default::default()
        });
        let mut input = good_gate_input();
        input.depth_1pct_usd = 50_000.0;
        let result = gate.check(&input, 100_000.0);
        assert!(matches!(result, GateResult::Reject(GateReason::InsufficientDepth { .. })));
    }

    #[test]
    fn test_participation_gate_rejects() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            max_participation_rate: 0.10,
            ..Default::default()
        });
        let mut input = good_gate_input();
        input.hourly_volume_usd = 500_000.0; // trade=100K, volume=500K → 20% > 10%
        let result = gate.check(&input, 100_000.0);
        assert!(matches!(result, GateResult::Reject(GateReason::ExcessiveParticipation { .. })));
    }

    #[test]
    fn test_exit_cost_gate_rejects() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            max_exit_cost_pct: 0.01,
            ..Default::default()
        });
        let mut input = good_gate_input();
        input.exit_cost_pct = 0.03;
        let result = gate.check(&input, 100_000.0);
        assert!(matches!(result, GateResult::Reject(GateReason::HighExitCost { .. })));
    }

    #[test]
    fn test_margin_gate_rejects() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            min_margin_headroom: 0.40,
            ..Default::default()
        });
        let mut input = good_gate_input();
        input.margin_headroom = 0.20;
        let result = gate.check(&input, 100_000.0);
        assert!(matches!(result, GateResult::Reject(GateReason::LowMargin { .. })));
    }

    #[test]
    fn test_apply_defers_rejected_trade() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            max_spread_bps: 20.0,
            ..Default::default()
        });

        let mut weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.5 },
            WeightOutput { timestamp: 100, symbol: "BAD".into(), weight_qp: 0.3, weight_final: 0.3 },
        ];

        let mut inputs = HashMap::new();
        inputs.insert((100, "BTC".into()), good_gate_input());
        inputs.insert((100, "BAD".into()), GateInput {
            spread_bps: 60.0, // too wide
            ..good_gate_input()
        });

        let mut prev = HashMap::new();
        prev.insert("BTC".into(), 0.0);
        prev.insert("BAD".into(), 0.1); // had position

        let rejections = gate.apply(&mut weights, &inputs, 10_000.0, &prev);

        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].symbol, "BAD");

        // BTC: passed → weight unchanged
        assert!((weights[0].weight_final - 0.5).abs() < 1e-10);
        // BAD: rejected → reverted to prev weight (0.1)
        assert!((weights[1].weight_final - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_apply_no_trade_no_gate() {
        let gate = PreTradeGate::new(&GateConfig {
            enabled: true,
            max_spread_bps: 20.0,
            ..Default::default()
        });

        // Weight same as prev → no trade → no gate check
        let mut weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.5 },
        ];
        let mut prev = HashMap::new();
        prev.insert("BTC".into(), 0.5);

        let rejections = gate.apply(&mut weights, &HashMap::new(), 10_000.0, &prev);
        assert!(rejections.is_empty(), "no trade → no gate needed");
        assert!((weights[0].weight_final - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_missing_gate_data_defers() {
        let gate = PreTradeGate::new(&GateConfig { enabled: true, ..Default::default() });

        let mut weights = vec![
            WeightOutput { timestamp: 100, symbol: "NEW".into(), weight_qp: 0.3, weight_final: 0.3 },
        ];

        // No gate data for NEW → should defer
        let rejections = gate.apply(&mut weights, &HashMap::new(), 10_000.0, &HashMap::new());
        assert_eq!(rejections.len(), 1);
        assert!(matches!(rejections[0].reason, GateReason::MissingData));
        assert!((weights[0].weight_final - 0.0).abs() < 1e-10); // reverted to prev=0
    }
}
