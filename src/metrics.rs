//! Walk-forward eval metrics: Sharpe, MDD, hit rate, turnover, bootstrap CI, deflated Sharpe.
//!
//! Ports `train/rumpy_train/execution/metrics.py`. All functions are pure math
//! operating on return series — no I/O or state.

use std::collections::HashMap;

use crate::pipeline::WeightOutput;

// ---------------------------------------------------------------------------
// Portfolio-level aggregates
// ---------------------------------------------------------------------------

/// Compute per-bar portfolio return: Σ_i w_i · r_i.
///
/// Returns sorted Vec of (timestamp, portfolio_return).
pub fn portfolio_return_per_bar(
    weights: &[WeightOutput],
    forward_returns: &HashMap<(i64, String), f64>,
) -> Vec<(i64, f64)> {
    let mut by_ts: HashMap<i64, f64> = HashMap::new();
    for w in weights {
        let r = forward_returns
            .get(&(w.timestamp, w.symbol.clone()))
            .copied()
            .unwrap_or(0.0);
        *by_ts.entry(w.timestamp).or_default() += w.weight_final * r;
    }
    let mut result: Vec<(i64, f64)> = by_ts.into_iter().collect();
    result.sort_by_key(|(ts, _)| *ts);
    result
}

/// Compute per-bar turnover: Σ_i |w_i,t - w_i,t-1|.
///
/// First bar treats prior weights as 0.
///
/// Uses BTreeMap for `by_symbol` to make outer iteration sorted by symbol.
/// Otherwise contributions to `deltas_by_ts[ts]` accumulate in random order
/// across machines, producing 1-ULP-level cross-platform differences.
/// See docs/plans/cross-platform-divergence.md (Patch 2).
pub fn turnover_per_bar(weights: &[WeightOutput]) -> Vec<(i64, f64)> {
    use std::collections::BTreeMap;
    // Group by symbol, sorted by timestamp
    let mut by_symbol: BTreeMap<&str, Vec<(i64, f64)>> = BTreeMap::new();
    for w in weights {
        by_symbol
            .entry(&w.symbol)
            .or_default()
            .push((w.timestamp, w.weight_final));
    }
    for rows in by_symbol.values_mut() {
        rows.sort_by_key(|(ts, _)| *ts);
    }

    // Compute |Δw| per (ts, sym), then sum per ts. Outer iteration of
    // `by_symbol` (BTreeMap) is sorted, so symbols contribute in deterministic
    // order to the same-ts accumulator.
    let mut deltas_by_ts: HashMap<i64, f64> = HashMap::new();
    for rows in by_symbol.values() {
        for i in 0..rows.len() {
            let (ts, w) = rows[i];
            let w_prev = if i > 0 { rows[i - 1].1 } else { 0.0 };
            *deltas_by_ts.entry(ts).or_default() += (w - w_prev).abs();
        }
    }

    let mut result: Vec<(i64, f64)> = deltas_by_ts.into_iter().collect();
    result.sort_by_key(|(ts, _)| *ts);
    result
}

// ---------------------------------------------------------------------------
// Scalar metrics
// ---------------------------------------------------------------------------

/// Annualized Sharpe = mean/std × √bars_per_year. Uses sample std (ddof=1).
pub fn sharpe_annualized(returns: &[f64], bars_per_year: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();
    if std <= 1e-12 {
        return 0.0;
    }
    mean / std * bars_per_year.sqrt()
}

/// Max drawdown on additive cumulative return curve.
pub fn max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut equity = 0.0;
    let mut peak = 0.0;
    let mut worst = 0.0;
    for &r in returns {
        equity += r;
        if equity > peak {
            peak = equity;
        }
        let dd = equity - peak;
        if dd < worst {
            worst = dd;
        }
    }
    worst
}

/// Fraction of bars with positive return.
pub fn hit_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n_pos = returns.iter().filter(|&&r| r > 0.0).count();
    n_pos as f64 / returns.len() as f64
}

/// Annualized mean return (additive convention).
pub fn annualized_return(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    mean * bars_per_year
}

/// Ulcer Index: RMS of drawdown percentage from peak (multiplicative equity).
pub fn ulcer_index(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mut cum = 1.0;
    let mut peak = 1.0;
    let mut sum_sq = 0.0;
    let n = returns.len();
    for &r in returns {
        cum *= 1.0 + r;
        if cum > peak {
            peak = cum;
        }
        let dd_pct = (cum - peak) / peak;
        sum_sq += dd_pct * dd_pct;
    }
    (sum_sq / n as f64).sqrt()
}

/// Martin ratio: annualized return / Ulcer Index.
pub fn martin_ratio(returns: &[f64], bars_per_year: f64) -> f64 {
    let ui = ulcer_index(returns);
    if ui <= 1e-10 {
        return 0.0;
    }
    annualized_return(returns, bars_per_year) / ui
}

/// Calmar ratio: annualized return / |MDD|.
pub fn calmar_ratio(returns: &[f64], bars_per_year: f64) -> f64 {
    let mdd = max_drawdown(returns).abs();
    if mdd < 1e-9 {
        return 0.0;
    }
    annualized_return(returns, bars_per_year) / mdd
}

/// Annualized geometric growth rate from multiplicative equity curve.
pub fn geometric_growth(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let cum: f64 = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r));
    if cum <= 0.0 {
        return -1.0;
    }
    let t = returns.len() as f64;
    cum.powf(bars_per_year / t) - 1.0
}

/// Mean log-wealth growth rate (annualized). Concave in leverage (Peters 2009).
pub fn log_growth(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let sum_log: f64 = returns
        .iter()
        .map(|&r| (1.0 + r).max(1e-8).ln())
        .sum();
    sum_log / returns.len() as f64 * bars_per_year
}

/// Fraction of bars spent in drawdown > threshold (default 2%).
pub fn frac_underwater(returns: &[f64], threshold: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mut cum = 1.0;
    let mut peak = 1.0;
    let mut count = 0usize;
    for &r in returns {
        cum *= 1.0 + r;
        if cum > peak {
            peak = cum;
        }
        let dd_pct = (cum - peak) / peak;
        if dd_pct < -threshold {
            count += 1;
        }
    }
    count as f64 / returns.len() as f64
}

/// CDaR (Conditional Drawdown at Risk): mean of worst α-fraction of drawdowns
/// (returned as positive magnitude). The α value is caller-supplied. Production
/// tuning currently uses α=0.12 from `walkforward.rs`; α=0.20 is used in unit
/// tests. Smaller α focuses on a deeper tail and is more punitive.
pub fn cdar(returns: &[f64], alpha: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mut cum = 1.0;
    let mut peak = 1.0;
    let mut dd_abs: Vec<f64> = Vec::with_capacity(returns.len());
    for &r in returns {
        cum *= 1.0 + r;
        if cum > peak {
            peak = cum;
        }
        let dd_pct = (cum - peak) / peak;
        dd_abs.push(-dd_pct); // positive magnitudes
    }
    dd_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_idx = ((1.0 - alpha) * dd_abs.len() as f64).floor() as usize;
    let tail = &dd_abs[threshold_idx..];
    if tail.is_empty() {
        return 0.0;
    }
    tail.iter().sum::<f64>() / tail.len() as f64
}

// ---------------------------------------------------------------------------
// Sample moments
// ---------------------------------------------------------------------------

/// Sample (skew, raw kurtosis) of a return series. Raw kurtosis = 3.0 for Gaussian.
pub fn sharpe_moments(returns: &[f64]) -> (f64, f64) {
    let n = returns.len();
    if n < 4 {
        return (0.0, 3.0);
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    let sigma_sq = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let sigma = sigma_sq.sqrt();
    if sigma <= 1e-12 {
        return (0.0, 3.0);
    }
    let skew: f64 = returns.iter().map(|r| ((r - mean) / sigma).powi(3)).sum::<f64>() / n as f64;
    let kurt: f64 = returns.iter().map(|r| ((r - mean) / sigma).powi(4)).sum::<f64>() / n as f64;
    (skew, kurt)
}

// ---------------------------------------------------------------------------
// Block-bootstrap CI on Sharpe
// ---------------------------------------------------------------------------

/// Block-bootstrap CI95 on annualized Sharpe.
///
/// Blocks preserve short-range autocorrelation. Returns (lower, upper) percentiles.
pub fn block_bootstrap_sharpe_ci(
    returns: &[f64],
    bars_per_year: f64,
    n_iter: usize,
    block_size: usize,
    seed: u64,
) -> (f64, f64) {
    use rand::prelude::*;
    use rand::rngs::StdRng;

    let n = returns.len();
    if n < block_size.max(2) {
        return (0.0, 0.0);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let n_blocks_needed = (n / block_size) + 1;
    let max_start = n.saturating_sub(block_size);
    let mut sharpes: Vec<f64> = Vec::with_capacity(n_iter);

    for _ in 0..n_iter {
        let mut sample = Vec::with_capacity(n_blocks_needed * block_size);
        for _ in 0..n_blocks_needed {
            let start = rng.random_range(0..=max_start);
            sample.extend_from_slice(&returns[start..start + block_size]);
        }
        sample.truncate(n);
        let sr = sharpe_annualized(&sample, bars_per_year);
        sharpes.push(sr);
    }

    sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_idx = (0.025 * n_iter as f64).floor() as usize;
    let hi_idx = (0.975 * n_iter as f64).floor() as usize;
    (sharpes[lo_idx.min(sharpes.len() - 1)], sharpes[hi_idx.min(sharpes.len() - 1)])
}

// ---------------------------------------------------------------------------
// Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
// ---------------------------------------------------------------------------

/// Expected max Sharpe from n_trials i.i.d. N(0,1) noise trials.
fn expected_max_noise_sr(n_trials: usize) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    if n_trials <= 1 {
        return 0.0;
    }
    let norm = Normal::standard();
    let euler = 0.5772156649015329;
    let e = std::f64::consts::E;
    let n = n_trials as f64;
    (1.0 - euler) * norm.inverse_cdf(1.0 - 1.0 / n)
        + euler * norm.inverse_cdf(1.0 - 1.0 / (n * e))
}

/// Deflated Sharpe probability (Bailey & López de Prado, 2014).
///
/// DSR ∈ [0, 1] = probability that observed SR beats the max-of-noise.
/// DSR > 0.95 = statistically significant at 5% after correcting for
/// multiple testing across n_trials.
///
/// All quantities are per-bar inside the formula (not annualized).
/// `kurtosis` is RAW (3.0 for Gaussian, not excess).
pub fn deflated_sharpe(
    sharpe: f64,
    n_trials: usize,
    skew: f64,
    kurtosis: f64,
    n_obs: usize,
    samples_per_year: f64,
) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};

    if n_obs < 4 || n_trials <= 1 {
        return 0.0;
    }

    let sr_pb = if samples_per_year > 0.0 {
        sharpe / samples_per_year.sqrt()
    } else {
        sharpe
    };

    let sr_null_pb = expected_max_noise_sr(n_trials) / ((n_obs - 1) as f64).sqrt();

    let denom_sq = 1.0 - skew * sr_pb + ((kurtosis - 1.0) / 4.0) * sr_pb.powi(2);
    if denom_sq <= 0.0 {
        return 0.0;
    }
    let denom = denom_sq.sqrt();
    let z = (sr_pb - sr_null_pb) * ((n_obs - 1) as f64).sqrt() / denom;

    let norm = Normal::standard();
    norm.cdf(z)
}

// ---------------------------------------------------------------------------
// Summary metrics struct
// ---------------------------------------------------------------------------

/// Summary metrics for a backtest trial.
#[derive(Debug, Clone)]
pub struct TrialMetrics {
    pub sharpe: f64,
    pub ann_return: f64,
    pub mdd: f64,
    pub hit_rate: f64,
    pub avg_turnover: f64,
    pub calmar: f64,
    pub ulcer_index: f64,
    pub martin_ratio: f64,
    pub n_bars: usize,
    pub sharpe_ci95: (f64, f64),
    pub deflated_sharpe: f64,
}

/// Compute summary metrics from per-bar returns and weights.
pub fn compute_trial_metrics(
    returns: &[f64],
    weights: &[WeightOutput],
    bars_per_year: f64,
    n_trials: usize,
) -> TrialMetrics {
    let sr = sharpe_annualized(returns, bars_per_year);
    let (skew, kurt) = sharpe_moments(returns);
    let ci = block_bootstrap_sharpe_ci(returns, bars_per_year, 1000, 20, 42);
    let dsr = deflated_sharpe(sr, n_trials, skew, kurt, returns.len(), bars_per_year);

    let turnover = turnover_per_bar(weights);
    let avg_turnover = if turnover.is_empty() {
        0.0
    } else {
        turnover.iter().map(|(_, t)| t).sum::<f64>() / turnover.len() as f64
    };

    TrialMetrics {
        sharpe: sr,
        ann_return: annualized_return(returns, bars_per_year),
        mdd: max_drawdown(returns),
        hit_rate: hit_rate(returns),
        avg_turnover,
        calmar: calmar_ratio(returns, bars_per_year),
        ulcer_index: ulcer_index(returns),
        martin_ratio: martin_ratio(returns, bars_per_year),
        n_bars: returns.len(),
        sharpe_ci95: ci,
        deflated_sharpe: dsr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_annualized_constant_returns() {
        // Constant positive returns → infinite Sharpe (capped by std threshold)
        let returns = vec![0.001; 100];
        let sr = sharpe_annualized(&returns, 365.0);
        // std is very small but not zero due to floating point
        // With constant returns, sample std (ddof=1) should be ~0
        assert!(sr > 100.0 || sr == 0.0); // either huge or zero if std < threshold
    }

    #[test]
    fn test_sharpe_annualized_known() {
        // mean=0.001, std≈0.01 → per-bar SR = 0.1, annualized = 0.1 × √365 ≈ 1.91
        let returns: Vec<f64> = (0..500).map(|i| {
            if i % 2 == 0 { 0.011 } else { -0.009 }
        }).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        assert!((mean - 0.001).abs() < 1e-10);

        let sr = sharpe_annualized(&returns, 365.0);
        assert!(sr > 1.5 && sr < 2.5, "Sharpe={sr}");
    }

    #[test]
    fn test_max_drawdown() {
        // +1%, +1%, -5%, +1% → MDD at the -5% bar
        let returns = vec![0.01, 0.01, -0.05, 0.01];
        let mdd = max_drawdown(&returns);
        // equity: 0.01, 0.02, -0.03, -0.02
        // peak:   0.01, 0.02,  0.02,  0.02
        // dd:     0.00, 0.00, -0.05, -0.04
        assert!((mdd - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown_no_dd() {
        let returns = vec![0.01, 0.02, 0.03];
        assert_eq!(max_drawdown(&returns), 0.0);
    }

    #[test]
    fn test_hit_rate() {
        let returns = vec![0.01, -0.01, 0.02, 0.0, -0.005];
        assert!((hit_rate(&returns) - 0.4).abs() < 1e-10); // 2/5
    }

    #[test]
    fn test_ulcer_index_no_drawdown() {
        let returns = vec![0.01, 0.01, 0.01, 0.01];
        let ui = ulcer_index(&returns);
        assert!(ui < 1e-10, "UI should be ~0 for monotonically rising equity");
    }

    #[test]
    fn test_ulcer_index_with_drawdown() {
        let returns = vec![0.10, -0.20, -0.10, 0.05];
        let ui = ulcer_index(&returns);
        assert!(ui > 0.0);
    }

    #[test]
    fn test_cdar_zero_drawdown() {
        let returns = vec![0.01; 100];
        let c = cdar(&returns, 0.20);
        assert!(c < 1e-10);
    }

    #[test]
    fn test_cdar_with_drawdown() {
        let mut returns = vec![0.01; 50];
        returns.extend(vec![-0.03; 20]); // drawdown period
        returns.extend(vec![0.01; 30]);
        let c = cdar(&returns, 0.20);
        assert!(c > 0.0, "CDaR should be positive when drawdowns exist");
    }

    #[test]
    fn test_sharpe_moments_gaussian_like() {
        // Symmetric distribution → skew ≈ 0, kurtosis ≈ 3
        let returns: Vec<f64> = (0..10000)
            .map(|i| {
                let x = (i as f64 / 10000.0 * std::f64::consts::TAU).sin() * 0.01;
                x
            })
            .collect();
        let (skew, kurt) = sharpe_moments(&returns);
        assert!(skew.abs() < 0.5, "skew={skew}");
        // Sine is platykurtic (< 3), but we just check it's reasonable
        assert!(kurt > 1.0 && kurt < 5.0, "kurt={kurt}");
    }

    #[test]
    fn test_deflated_sharpe_noise() {
        // Low Sharpe with many trials → DSR should be low
        let dsr = deflated_sharpe(0.5, 500, 0.0, 3.0, 1000, 365.0);
        assert!(dsr < 0.5, "DSR={dsr} should be low for mediocre Sharpe with 500 trials");
    }

    #[test]
    fn test_deflated_sharpe_strong() {
        // High Sharpe with few trials → DSR should be high
        let dsr = deflated_sharpe(3.0, 5, 0.0, 3.0, 1000, 365.0);
        assert!(dsr > 0.9, "DSR={dsr} should be high for strong Sharpe with few trials");
    }

    #[test]
    fn test_geometric_growth() {
        // 1% per bar for 365 bars → ~(1.01)^365 - 1 annualized
        let returns = vec![0.01; 365];
        let g = geometric_growth(&returns, 365.0);
        let expected = 1.01_f64.powi(365) - 1.0;
        assert!((g - expected).abs() / expected < 0.01, "g={g}, expected={expected}");
    }

    #[test]
    fn test_log_growth() {
        let returns = vec![0.01; 365];
        let lg = log_growth(&returns, 365.0);
        let expected = (1.01_f64.ln()) * 365.0;
        assert!((lg - expected).abs() < 0.01, "lg={lg}, expected={expected}");
    }

    #[test]
    fn test_block_bootstrap_ci() {
        // Use non-periodic data to avoid degenerate CI
        let returns: Vec<f64> = (0..500).map(|i| {
            let phase = (i as f64 * 0.1).sin() * 0.003;
            0.001 + phase + if i % 3 == 0 { 0.002 } else { -0.001 }
        }).collect();
        let (lo, hi) = block_bootstrap_sharpe_ci(&returns, 365.0, 500, 10, 42);
        let point = sharpe_annualized(&returns, 365.0);
        assert!(lo <= point, "CI lower={lo} should be <= point={point}");
        assert!(hi >= point, "CI upper={hi} should be >= point={point}");
        assert!(lo < hi, "CI should have width: lo={lo}, hi={hi}");
    }

    #[test]
    fn test_portfolio_return_per_bar() {
        let weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.5 },
            WeightOutput { timestamp: 100, symbol: "ETH".into(), weight_qp: -0.3, weight_final: -0.3 },
            WeightOutput { timestamp: 200, symbol: "BTC".into(), weight_qp: 0.4, weight_final: 0.4 },
        ];
        let mut fwd = HashMap::new();
        fwd.insert((100, "BTC".into()), 0.02);
        fwd.insert((100, "ETH".into()), -0.01);
        fwd.insert((200, "BTC".into()), 0.03);

        let pnl = portfolio_return_per_bar(&weights, &fwd);
        assert_eq!(pnl.len(), 2);
        // bar 100: 0.5*0.02 + (-0.3)*(-0.01) = 0.01 + 0.003 = 0.013
        assert!((pnl[0].1 - 0.013).abs() < 1e-10);
        // bar 200: 0.4*0.03 = 0.012
        assert!((pnl[1].1 - 0.012).abs() < 1e-10);
    }

    #[test]
    fn test_turnover_per_bar() {
        let weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.5 },
            WeightOutput { timestamp: 100, symbol: "ETH".into(), weight_qp: -0.3, weight_final: -0.3 },
            WeightOutput { timestamp: 200, symbol: "BTC".into(), weight_qp: 0.4, weight_final: 0.4 },
            WeightOutput { timestamp: 200, symbol: "ETH".into(), weight_qp: -0.5, weight_final: -0.5 },
        ];
        let to = turnover_per_bar(&weights);
        assert_eq!(to.len(), 2);
        // bar 100: |0.5-0| + |(-0.3)-0| = 0.8
        assert!((to[0].1 - 0.8).abs() < 1e-10);
        // bar 200: |0.4-0.5| + |(-0.5)-(-0.3)| = 0.1 + 0.2 = 0.3
        assert!((to[1].1 - 0.3).abs() < 1e-10);
    }
}
