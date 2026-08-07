//! ML-predicted cost model: per-asset, per-hour spread and kappa from trained models.
//!
//! Resolution chain (per (timestamp, symbol)):
//!   1. L2 real data (book-walk spread + kappa) — when available
//!   2. ML model predictions (OHLCV-based spread/kappa scores from parquet)
//!   3. Error — missing prediction means data pipeline failure
//!
//! No tier-based costs. The ML model provides per-asset predictions directly.

use std::collections::HashMap;

use anyhow::{bail, Result};

/// Per-(timestamp, symbol) ML cost predictions.
#[derive(Debug, Clone)]
pub struct CostScore {
    /// Spread in bps (exp-transformed from log-space).
    pub spread_bps: f64,
    /// Impact kappa coefficient (exp-transformed from log-space).
    pub kappa: f64,
}

/// Cost provider: resolves spread and kappa per (timestamp, symbol).
///
/// In production, always uses ML predictions (with L2 override when available).
/// Missing ML prediction = error.
pub struct CostProvider {
    /// ML model predictions: (timestamp, symbol) -> CostScore
    ml_scores: HashMap<(i64, String), CostScore>,
    /// L2 real data: (timestamp, symbol) -> (spread_bps, kappa)
    l2_data: HashMap<(i64, String), (f64, f64)>,
    /// Taker fee in bps (always static — exchange-level, not model-predicted).
    taker_bps: f64,
    /// Linear cost multiplier (from Optuna champion config).
    c_lin_multiplier: f64,
    /// Whether to error on missing ML prediction (false = test mode with flat defaults).
    strict: bool,
    /// Flat default spread/kappa for test mode only.
    test_default_spread_bps: f64,
    test_default_kappa: f64,
}

impl CostProvider {
    /// Create from ML cost score parquet data.
    pub fn new(
        ml_scores: HashMap<(i64, String), CostScore>,
        l2_data: HashMap<(i64, String), (f64, f64)>,
        taker_bps: f64,
        c_lin_multiplier: f64,
    ) -> Self {
        Self {
            ml_scores,
            l2_data,
            taker_bps,
            c_lin_multiplier,
            strict: true,
            test_default_spread_bps: 0.0,
            test_default_kappa: 0.0,
        }
    }

    /// Create a test-only provider with flat default costs for all assets.
    /// Not for production — only for unit tests that don't need ML infrastructure.
    pub fn test_static(taker_bps: f64, default_spread_bps: f64, default_kappa: f64) -> Self {
        Self {
            ml_scores: HashMap::new(),
            l2_data: HashMap::new(),
            taker_bps,
            c_lin_multiplier: 1.0,
            strict: false,
            test_default_spread_bps: default_spread_bps,
            test_default_kappa: default_kappa,
        }
    }

    /// Linear cost for QP: c_lin = (taker_bps + 0.5 * spread_bps * multiplier) / 10000.
    pub fn c_lin(&self, ts: i64, symbol: &str) -> Result<f64> {
        let spread_bps = self.resolve_spread_bps(ts, symbol)?;
        Ok((self.taker_bps + 0.5 * spread_bps * self.c_lin_multiplier) / 10_000.0)
    }

    /// Impact kappa for sqrt-impact model (raw coefficient from ML).
    pub fn kappa(&self, ts: i64, symbol: &str) -> Result<f64> {
        self.resolve_kappa(ts, symbol)
    }

    /// Build spread and kappa lookup maps for the full universe.
    ///
    /// Returns (spread_bps_map, kappa_map). Pairs without ML coverage are
    /// omitted from the maps (the QP solver skips assets with missing costs).
    /// c_lin is computed dynamically inside the QP loop using exchange fee tiers.
    pub fn build_lookups(
        &self,
        timestamps_symbols: &[(i64, String)],
    ) -> Result<(HashMap<(i64, String), f64>, HashMap<(i64, String), f64>)> {
        let mut spread_map = HashMap::with_capacity(timestamps_symbols.len());
        let mut kappa_map = HashMap::with_capacity(timestamps_symbols.len());
        let mut n_missing = 0usize;

        for (ts, sym) in timestamps_symbols {
            match (self.resolve_spread_bps(*ts, sym), self.kappa(*ts, sym)) {
                (Ok(s), Ok(k)) => {
                    spread_map.insert((*ts, sym.clone()), s);
                    kappa_map.insert((*ts, sym.clone()), k);
                }
                _ => {
                    n_missing += 1;
                }
            }
        }

        if n_missing > 0 {
            let coverage_pct = 100.0 * spread_map.len() as f64 / timestamps_symbols.len() as f64;
            println!(
                "  cost coverage: {}/{} ({:.1}%), {} pairs missing (date range or universe mismatch)",
                spread_map.len(),
                timestamps_symbols.len(),
                coverage_pct,
                n_missing,
            );
        }

        Ok((spread_map, kappa_map))
    }

    /// Number of ML predictions loaded.
    pub fn n_ml_scores(&self) -> usize {
        self.ml_scores.len()
    }

    /// Number of L2 data points loaded.
    pub fn n_l2_data(&self) -> usize {
        self.l2_data.len()
    }

    // --- Internal resolution ---

    /// Look up ML score, trying exact timestamp first, then day-start (for D1 alpha vs H1 scores).
    fn lookup_ml(&self, ts: i64, symbol: &str) -> Option<&CostScore> {
        let key = (ts, symbol.to_string());
        if let Some(score) = self.ml_scores.get(&key) {
            return Some(score);
        }
        // D1 alpha timestamps may not match H1 score timestamps exactly.
        // Try the day-start (floor to 86400) which is typically the first H1 bar of the day.
        let day_start = (ts / 86400) * 86400;
        if day_start != ts {
            if let Some(score) = self.ml_scores.get(&(day_start, symbol.to_string())) {
                return Some(score);
            }
        }
        None
    }

    fn resolve_spread_bps(&self, ts: i64, symbol: &str) -> Result<f64> {
        // Priority 1: L2 real data
        if let Some(&(spread_bps, _)) = self.l2_data.get(&(ts, symbol.to_string())) {
            return Ok(spread_bps);
        }
        // Priority 2: ML model prediction (exact ts or day-start fallback)
        if let Some(score) = self.lookup_ml(ts, symbol) {
            return Ok(score.spread_bps);
        }
        // Priority 3: Error (or flat test default)
        if self.strict {
            bail!("missing spread prediction for ({}, {}) — data pipeline issue", ts, symbol);
        }
        Ok(self.test_default_spread_bps)
    }

    fn resolve_kappa(&self, ts: i64, symbol: &str) -> Result<f64> {
        // Priority 1: L2 real data
        if let Some(&(_, kappa)) = self.l2_data.get(&(ts, symbol.to_string())) {
            return Ok(kappa);
        }
        // Priority 2: ML model prediction (exact ts or day-start fallback)
        if let Some(score) = self.lookup_ml(ts, symbol) {
            return Ok(score.kappa);
        }
        // Priority 3: Error (or flat test default)
        if self.strict {
            bail!("missing kappa prediction for ({}, {}) — data pipeline issue", ts, symbol);
        }
        Ok(self.test_default_kappa)
    }
}

/// Build ML cost scores map from raw loaded data.
/// Exp-transforms from log-space to raw values.
pub fn build_ml_scores(
    raw: &[(i64, String, f64, f64)],
) -> HashMap<(i64, String), CostScore> {
    let mut map = HashMap::with_capacity(raw.len());
    for (ts, sym, log_spread, log_kappa) in raw {
        map.insert(
            (*ts, sym.clone()),
            CostScore {
                spread_bps: log_spread.exp(),
                kappa: log_kappa.exp(),
            },
        );
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ml_scores() -> HashMap<(i64, String), CostScore> {
        let mut m = HashMap::new();
        m.insert((100, "BTC".into()), CostScore { spread_bps: 1.5, kappa: 0.05 });
        m.insert((100, "ETH".into()), CostScore { spread_bps: 3.0, kappa: 0.10 });
        m.insert((200, "BTC".into()), CostScore { spread_bps: 1.8, kappa: 0.06 });
        m.insert((200, "ETH".into()), CostScore { spread_bps: 3.5, kappa: 0.12 });
        m
    }

    fn sample_l2_data() -> HashMap<(i64, String), (f64, f64)> {
        let mut m = HashMap::new();
        m.insert((100, "BTC".into()), (0.8, 0.03));
        m
    }

    #[test]
    fn test_l2_takes_priority() {
        let provider = CostProvider::new(sample_ml_scores(), sample_l2_data(), 3.5, 1.0);

        let c = provider.c_lin(100, "BTC").unwrap();
        let expected = (3.5 + 0.5 * 0.8 * 1.0) / 10_000.0;
        assert!((c - expected).abs() < 1e-10, "c_lin={c}, expected={expected}");

        let k = provider.kappa(100, "BTC").unwrap();
        assert!((k - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_ml_used_when_no_l2() {
        let provider = CostProvider::new(sample_ml_scores(), sample_l2_data(), 3.5, 1.0);

        let c = provider.c_lin(100, "ETH").unwrap();
        let expected = (3.5 + 0.5 * 3.0 * 1.0) / 10_000.0;
        assert!((c - expected).abs() < 1e-10);

        let k = provider.kappa(100, "ETH").unwrap();
        assert!((k - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_missing_ml_prediction_errors() {
        let provider = CostProvider::new(sample_ml_scores(), HashMap::new(), 3.5, 1.0);

        let result = provider.c_lin(100, "DOGE");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing spread prediction"));

        let result = provider.kappa(100, "DOGE");
        assert!(result.is_err());
    }

    #[test]
    fn test_test_static_flat_defaults() {
        let provider = CostProvider::test_static(3.5, 5.0, 0.1);

        let c = provider.c_lin(100, "ANYTHING").unwrap();
        let expected = (3.5 + 0.5 * 5.0 * 1.0) / 10_000.0;
        assert!((c - expected).abs() < 1e-10);

        let k = provider.kappa(100, "ANYTHING").unwrap();
        assert!((k - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_build_ml_scores_exp_transform() {
        let raw = vec![
            (100, "BTC".into(), 1.0_f64.ln(), (-3.0_f64).exp().ln()),
            (100, "ETH".into(), 2.0_f64.ln(), (-2.0_f64).exp().ln()),
        ];
        let scores = build_ml_scores(&raw);
        assert_eq!(scores.len(), 2);
        assert!((scores[&(100, "BTC".into())].spread_bps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_c_lin_multiplier_scales_spread() {
        let mut scores = HashMap::new();
        scores.insert((100, "BTC".into()), CostScore { spread_bps: 4.0, kappa: 0.1 });

        let provider = CostProvider::new(scores, HashMap::new(), 3.5, 2.0);
        let c = provider.c_lin(100, "BTC").unwrap();
        let expected = (3.5 + 0.5 * 4.0 * 2.0) / 10_000.0;
        assert!((c - expected).abs() < 1e-10);
    }

    #[test]
    fn test_build_lookups() {
        let provider = CostProvider::new(sample_ml_scores(), HashMap::new(), 3.5, 1.0);

        let ts_syms = vec![
            (100, "BTC".into()),
            (100, "ETH".into()),
            (200, "BTC".into()),
        ];

        let (spread_map, kappa) = provider.build_lookups(&ts_syms).unwrap();
        assert_eq!(spread_map.len(), 3);
        assert_eq!(kappa.len(), 3);

        // build_lookups now returns raw spread_bps (not c_lin)
        // BTC spread = exp(log(1.5)) = 1.5 bps (from sample_ml_scores)
        let btc_spread = spread_map[&(100, "BTC".into())];
        assert!((btc_spread - 1.5).abs() < 1e-6);
    }
}
