//! Alpha blending: load multi-horizon scores, z-score, GP13 persistence blend, winsorize.
//!
//! Ports `composite.py` (load + z-score) and `aim.py` (compute_future_weighted_alpha + winsorize).
//!
//! Pipeline:
//!   1. Load 5 per-horizon score parquets → inner join on (timestamp, symbol)
//!   2. Z-score each horizon cross-sectionally per bar
//!   3. Blend with GP13 persistence weights: α_future = Σ_h φ_h · z_h, φ_h = exp(-1/h)
//!   4. Winsorize α_future at ±σ × cross-sectional std per bar

use std::collections::HashMap;

/// Per-bar, per-symbol alpha data after blending.
#[derive(Debug, Clone)]
pub struct AlphaRow {
    pub timestamp: i64,
    pub symbol: String,
    pub alpha_future: f64,
}

/// Z-score each horizon's raw scores cross-sectionally per bar.
/// Input: `scores` is a map from horizon → Vec<(timestamp, symbol, score)>.
/// Output: merged wide-format with z-scored values per horizon.
pub fn zscore_and_blend(
    scores_by_horizon: &HashMap<usize, Vec<(i64, String, f64)>>,
    horizons: &[usize],
    winsorize_sigma: f64,
) -> Vec<AlphaRow> {
    // Step 1: Inner join all horizons on (timestamp, symbol)
    // Build a map: (ts, sym) → [score_h3, score_h6, ...]
    let n_h = horizons.len();
    let mut joined: HashMap<(i64, String), Vec<Option<f64>>> = HashMap::new();

    for (hi, &h) in horizons.iter().enumerate() {
        if let Some(rows) = scores_by_horizon.get(&h) {
            for (ts, sym, score) in rows {
                let entry = joined.entry((*ts, sym.clone())).or_insert_with(|| vec![None; n_h]);
                entry[hi] = Some(*score);
            }
        }
    }

    // Keep only rows with ALL horizons present
    let complete: Vec<((i64, String), Vec<f64>)> = joined
        .into_iter()
        .filter_map(|(key, vals)| {
            let all: Option<Vec<f64>> = vals.into_iter().collect();
            all.map(|v| (key, v))
        })
        .collect();

    // Step 2: Group by timestamp for cross-sectional z-scoring
    let mut by_ts: HashMap<i64, Vec<(String, Vec<f64>)>> = HashMap::new();
    for ((ts, sym), scores) in &complete {
        by_ts.entry(*ts).or_default().push((sym.clone(), scores.clone()));
    }

    // Step 3: Z-score + GP13 blend + winsorize per bar
    let mut result = Vec::new();

    for (&ts, assets) in &by_ts {
        let n_assets = assets.len();
        if n_assets < 2 {
            continue;
        }

        // Z-score each horizon
        let mut z_scores = vec![vec![0.0f64; n_h]; n_assets];
        for hi in 0..n_h {
            let vals: Vec<f64> = assets.iter().map(|(_, s)| s[hi]).collect();
            let mean = vals.iter().sum::<f64>() / n_assets as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n_assets as f64;
            let std = var.sqrt();
            if std < 1e-12 {
                continue;
            }
            for ai in 0..n_assets {
                z_scores[ai][hi] = (vals[ai] - mean) / std;
            }
        }

        // GP13 persistence blend: α_future = Σ_h φ_h · z_h
        let mut alpha_futures = vec![0.0f64; n_assets];
        for ai in 0..n_assets {
            for (hi, &h) in horizons.iter().enumerate() {
                let phi = (-1.0 / h as f64).exp();
                alpha_futures[ai] += phi * z_scores[ai][hi];
            }
        }

        // Winsorize at ±sigma × cross-sectional std
        let af_mean = alpha_futures.iter().sum::<f64>() / n_assets as f64;
        let af_var = alpha_futures.iter().map(|v| (v - af_mean) * (v - af_mean)).sum::<f64>() / n_assets as f64;
        let af_std = af_var.sqrt();
        if af_std > 1e-12 {
            let upper = af_mean + winsorize_sigma * af_std;
            let lower = af_mean - winsorize_sigma * af_std;
            for v in alpha_futures.iter_mut() {
                *v = v.clamp(lower, upper);
            }
        }

        for (ai, (sym, _)) in assets.iter().enumerate() {
            result.push(AlphaRow {
                timestamp: ts,
                symbol: sym.clone(),
                alpha_future: alpha_futures[ai],
            });
        }
    }

    // Sort by timestamp then symbol for deterministic output
    result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp).then(a.symbol.cmp(&b.symbol)));
    result
}

/// Broadcast D1 alpha to H1 timestamps.
///
/// For each D1 alpha row at timestamp T, replicates it to every H1 timestamp
/// in `h1_timestamps` that falls within the same UTC day [day_start, day_start + 86400).
/// This allows D1 alpha targets to drive H1 execution.
pub fn broadcast_d1_to_h1(
    d1_alphas: &[AlphaRow],
    h1_timestamps: &[i64],
) -> Vec<AlphaRow> {
    // Build lookup: day_start → Vec<h1_timestamp>
    let mut h1_by_day: HashMap<i64, Vec<i64>> = HashMap::new();
    for &ts in h1_timestamps {
        let day = (ts / 86400) * 86400;
        h1_by_day.entry(day).or_default().push(ts);
    }

    let mut result = Vec::with_capacity(d1_alphas.len() * 24);
    for row in d1_alphas {
        let day = (row.timestamp / 86400) * 86400;
        if let Some(h1_ts_list) = h1_by_day.get(&day) {
            for &h1_ts in h1_ts_list {
                result.push(AlphaRow {
                    timestamp: h1_ts,
                    symbol: row.symbol.clone(),
                    alpha_future: row.alpha_future,
                });
            }
        }
    }

    result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp).then(a.symbol.cmp(&b.symbol)));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_blend_basic() {
        let mut scores: HashMap<usize, Vec<(i64, String, f64)>> = HashMap::new();

        // 2 bars, 3 assets, 2 horizons
        for &h in &[3, 6] {
            let mut rows = Vec::new();
            rows.push((100, "BTC".to_string(), 0.5 * h as f64));
            rows.push((100, "ETH".to_string(), -0.3 * h as f64));
            rows.push((100, "SOL".to_string(), 0.1 * h as f64));
            rows.push((200, "BTC".to_string(), 0.2 * h as f64));
            rows.push((200, "ETH".to_string(), 0.4 * h as f64));
            rows.push((200, "SOL".to_string(), -0.1 * h as f64));
            scores.insert(h, rows);
        }

        let result = zscore_and_blend(&scores, &[3, 6], 3.0);

        assert_eq!(result.len(), 6); // 2 bars × 3 assets

        // Check dollar-neutral per bar (z-scored input → blend should be ~zero-mean)
        for ts in [100, 200] {
            let bar: Vec<&AlphaRow> = result.iter().filter(|r| r.timestamp == ts).collect();
            let sum: f64 = bar.iter().map(|r| r.alpha_future).sum();
            assert!(sum.abs() < 1e-6, "bar {ts} not zero-mean: sum={sum}");
        }
    }

    #[test]
    fn test_missing_horizon_drops_row() {
        let mut scores: HashMap<usize, Vec<(i64, String, f64)>> = HashMap::new();
        scores.insert(3, vec![(100, "BTC".to_string(), 1.0), (100, "ETH".to_string(), -1.0)]);
        scores.insert(6, vec![(100, "BTC".to_string(), 0.5)]); // ETH missing for h6

        let result = zscore_and_blend(&scores, &[3, 6], 3.0);
        // Only BTC has both horizons, but need ≥2 assets for z-score → empty
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_persistence_weights() {
        // φ_h = exp(-1/h): h=3 → 0.717, h=50 → 0.980
        // Longer horizons should get more weight
        let phi_3 = (-1.0f64 / 3.0).exp();
        let phi_50 = (-1.0f64 / 50.0).exp();
        assert!(phi_50 > phi_3, "h50 should have higher weight than h3");
        assert!((phi_3 - 0.7165).abs() < 0.001);
        assert!((phi_50 - 0.9802).abs() < 0.001);
    }
}
