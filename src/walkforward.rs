//! Purged walk-forward CV for Optuna trials.
//!
//! Ports `train/rumpy_train/execution/walkforward.py`. Slices backtest per-bar
//! returns into N non-overlapping test windows, computes per-fold metrics, and
//! aggregates into objective functions.
//!
//! Purging here guards against state leakage WITHIN the backtest (w_prev chain
//! across fold boundaries). Alpha scores are already out-of-sample from XGBoost
//! walk-forward training.
//!
//! Reference: López de Prado (2018), Advances in Financial Machine Learning, Ch. 7.

use crate::config::WalkForwardConfig;
use crate::metrics;

// ---------------------------------------------------------------------------
// Fold specification
// ---------------------------------------------------------------------------

/// One test fold in the walk-forward evaluation.
#[derive(Debug, Clone)]
pub struct FoldSpec {
    pub fold_idx: usize,
    /// Inclusive start timestamp.
    pub start_ts: i64,
    /// Inclusive end timestamp — last bar in the fold.
    pub end_ts: i64,
    pub n_bars: usize,
}

// ---------------------------------------------------------------------------
// Fold boundary computation
// ---------------------------------------------------------------------------

/// Divide sorted timestamps into N non-overlapping test windows with purge gaps.
///
/// Timeline: `[warmup] [fold 0] [purge] [fold 1] [purge] ... [fold N-1]`
///
/// Last fold absorbs leftover bars.
pub fn compute_fold_boundaries(
    sorted_timestamps: &[i64],
    config: &WalkForwardConfig,
) -> Vec<FoldSpec> {
    let n_total = sorted_timestamps.len();
    if config.warmup_bars >= n_total {
        eprintln!("  warmup_bars ({}) >= n_total ({}), skipping walk-forward", config.warmup_bars, n_total);
        return Vec::new();
    }

    let n_available = n_total - config.warmup_bars;
    let n_purge_total = config.purge_bars * (config.n_folds - 1);
    let n_test_total = n_available.saturating_sub(n_purge_total);
    assert!(
        n_test_total > config.n_folds,
        "After warmup={} and {} purges × {}, only {} bars remain for {} folds",
        config.warmup_bars,
        config.n_folds - 1,
        config.purge_bars,
        n_test_total,
        config.n_folds
    );

    let fold_size = n_test_total / config.n_folds;
    let mut folds = Vec::with_capacity(config.n_folds);
    let mut cursor = config.warmup_bars;

    for i in 0..config.n_folds {
        let start_idx = cursor;
        let mut end_idx = start_idx + fold_size - 1;
        if i == config.n_folds - 1 {
            // Last fold absorbs remainder
            end_idx = n_total - 1;
        }
        end_idx = end_idx.min(n_total - 1);
        folds.push(FoldSpec {
            fold_idx: i,
            start_ts: sorted_timestamps[start_idx],
            end_ts: sorted_timestamps[end_idx],
            n_bars: end_idx - start_idx + 1,
        });
        cursor = end_idx + 1 + config.purge_bars;
    }

    folds
}

// ---------------------------------------------------------------------------
// Per-fold evaluation
// ---------------------------------------------------------------------------

/// Per-fold metrics computed by evaluate_folds.
#[derive(Debug, Clone, Default)]
pub struct FoldMetrics {
    pub sharpe: f64,
    pub ann_return: f64,
    pub mdd: f64,
    pub ulcer: f64,
    pub martin: f64,
    pub log_growth: f64,
    pub geo_growth: f64,
    pub frac_underwater: f64,
    pub cdar: f64,
    pub n_bars: usize,
}

/// Aggregated fold evaluation result.
#[derive(Debug, Clone)]
pub struct FoldEval {
    pub folds: Vec<FoldMetrics>,
    pub mean_sharpe: f64,
    pub std_sharpe: f64,
    pub min_sharpe: f64,
    pub max_sharpe: f64,
    pub n_folds_ok: usize,
}

/// Evaluate all folds: compute per-fold metrics + aggregates.
///
/// `per_bar_returns`: sorted Vec of (timestamp, return).
/// `folds`: from compute_fold_boundaries.
/// `samples_per_year`: bars per year for annualization (e.g. 365 for D1).
pub fn evaluate_folds(
    per_bar_returns: &[(i64, f64)],
    folds: &[FoldSpec],
    samples_per_year: f64,
    min_fold_bars: usize,
) -> FoldEval {
    // Build timestamp → return lookup
    let _ts_map: std::collections::HashMap<i64, f64> = per_bar_returns.iter().copied().collect();

    let mut fold_metrics = Vec::with_capacity(folds.len());

    for spec in folds {
        // Extract returns in this fold's time window
        let fold_returns: Vec<f64> = per_bar_returns
            .iter()
            .filter(|(ts, _)| *ts >= spec.start_ts && *ts <= spec.end_ts)
            .map(|(_, r)| *r)
            .collect();

        let n = fold_returns.len();
        if n < min_fold_bars {
            fold_metrics.push(FoldMetrics {
                n_bars: n,
                sharpe: f64::NAN,
                ann_return: f64::NAN,
                mdd: 0.0,
                ulcer: f64::NAN,
                martin: f64::NAN,
                log_growth: f64::NAN,
                geo_growth: f64::NAN,
                frac_underwater: f64::NAN,
                cdar: f64::NAN,
            });
            continue;
        }

        let std_check: f64 = {
            let mean = fold_returns.iter().sum::<f64>() / n as f64;
            (fold_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        if std_check < 1e-12 {
            fold_metrics.push(FoldMetrics {
                n_bars: n,
                sharpe: f64::NAN,
                ann_return: f64::NAN,
                mdd: 0.0,
                ulcer: f64::NAN,
                martin: f64::NAN,
                log_growth: f64::NAN,
                geo_growth: f64::NAN,
                frac_underwater: f64::NAN,
                cdar: f64::NAN,
            });
            continue;
        }

        fold_metrics.push(FoldMetrics {
            sharpe: metrics::sharpe_annualized(&fold_returns, samples_per_year),
            ann_return: metrics::annualized_return(&fold_returns, samples_per_year),
            mdd: metrics::max_drawdown(&fold_returns),
            ulcer: metrics::ulcer_index(&fold_returns),
            martin: metrics::martin_ratio(&fold_returns, samples_per_year),
            log_growth: metrics::log_growth(&fold_returns, samples_per_year),
            geo_growth: metrics::geometric_growth(&fold_returns, samples_per_year),
            frac_underwater: metrics::frac_underwater(&fold_returns, 0.02),
            // CDaR α=0.12: average of worst 12% of drawdowns (was 0.20).
            // Tightened 2026-04-26 to focus the tail metric on deeper-tail
            // drawdowns. Smaller α → more punitive (mean of a smaller, deeper
            // tail). Tuner λ_cdar weight unchanged at 2.0 — the new α value
            // shifts where the optimum sits, not the penalty multiplier.
            cdar: metrics::cdar(&fold_returns, 0.12),
            n_bars: n,
        });
    }

    // Aggregate — exclude NaN folds
    let valid_sharpes: Vec<f64> = fold_metrics
        .iter()
        .map(|f| f.sharpe)
        .filter(|s| s.is_finite())
        .collect();

    let n_ok = valid_sharpes.len();
    let (mean_sharpe, std_sharpe, min_sharpe, max_sharpe) = if n_ok == 0 {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mean = valid_sharpes.iter().sum::<f64>() / n_ok as f64;
        let std = if n_ok > 1 {
            (valid_sharpes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_ok as f64).sqrt()
        } else {
            0.0
        };
        let min = valid_sharpes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = valid_sharpes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, std, min, max)
    };

    FoldEval {
        folds: fold_metrics,
        mean_sharpe,
        std_sharpe,
        min_sharpe,
        max_sharpe,
        n_folds_ok: n_ok,
    }
}

// ---------------------------------------------------------------------------
// Per-fold aggregation objectives — REMOVED 2026-04-27.
//
// The previous tuner used per-fold-aggregating objectives
// (cdar_penalized_return, pgr_crypto, stability_adjusted_sharpe,
// min_fold_sharpe, log_wealth_growth, ewpm_martin) plus exponential
// recency_weights. These were dropped when adopting the Candidate A
// objective: full-path `ann_return − λ·CDaR_α` computed in Python on the
// in-sample slice. See train/scripts/tune_execution.py module docstring for
// the full design + citations. None of these helpers had peer-reviewed HPO
// support and they were never exposed to the new tuner.
//
// FoldEval and evaluate_folds remain — they're useful as diagnostic per-fold
// breakdowns, just no longer load-bearing in the optimization objective.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timestamps(n: usize) -> Vec<i64> {
        (0..n).map(|i| i as i64 * 86400).collect()
    }

    fn make_returns(n: usize, mean: f64, spread: f64) -> Vec<(i64, f64)> {
        (0..n)
            .map(|i| {
                let ts = i as i64 * 86400;
                let r = if i % 2 == 0 {
                    mean + spread
                } else {
                    mean - spread
                };
                (ts, r)
            })
            .collect()
    }

    #[test]
    fn test_fold_boundaries_basic() {
        let ts = make_timestamps(1000);
        let config = WalkForwardConfig {
            n_folds: 5,
            warmup_bars: 200,
            purge_bars: 50,
            stability_penalty: 0.5,
            min_fold_bars: 10,
        };

        let folds = compute_fold_boundaries(&ts, &config);
        assert_eq!(folds.len(), 5);

        // First fold starts after warmup
        assert_eq!(folds[0].start_ts, ts[200]);

        // Last fold ends at the last timestamp
        assert_eq!(folds[4].end_ts, ts[999]);

        // Folds don't overlap (each fold's start > prev fold's end + purge)
        for i in 1..folds.len() {
            assert!(folds[i].start_ts > folds[i - 1].end_ts);
        }

        // Total bars across folds + purges + warmup ≈ n_total
        let test_bars: usize = folds.iter().map(|f| f.n_bars).sum();
        let purge_bars = config.purge_bars * (config.n_folds - 1);
        assert!(test_bars + purge_bars + config.warmup_bars <= 1000);
    }

    #[test]
    fn test_fold_boundaries_single_fold() {
        let ts = make_timestamps(500);
        let config = WalkForwardConfig {
            n_folds: 1,
            warmup_bars: 100,
            purge_bars: 50,
            stability_penalty: 0.5,
            min_fold_bars: 10,
        };

        let folds = compute_fold_boundaries(&ts, &config);
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].start_ts, ts[100]);
        assert_eq!(folds[0].end_ts, ts[499]);
        assert_eq!(folds[0].n_bars, 400);
    }

    #[test]
    fn test_evaluate_folds_basic() {
        let returns = make_returns(1000, 0.001, 0.005);
        let ts: Vec<i64> = returns.iter().map(|(t, _)| *t).collect();
        let config = WalkForwardConfig::default();
        let folds = compute_fold_boundaries(&ts, &config);
        let eval = evaluate_folds(&returns, &folds, 365.0, 10);

        assert_eq!(eval.folds.len(), 5);
        assert!(eval.n_folds_ok > 0);
        assert!(eval.mean_sharpe.is_finite());
        assert!(eval.std_sharpe.is_finite());

        // All folds should have positive Sharpe (mean=0.001 is positive)
        for fm in &eval.folds {
            if fm.sharpe.is_finite() {
                assert!(fm.sharpe > 0.0, "fold sharpe should be positive: {}", fm.sharpe);
            }
        }
    }

}
