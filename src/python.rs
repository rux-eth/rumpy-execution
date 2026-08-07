//! PyO3 bindings for the execution backtest.
//!
//! Two APIs:
//!
//! 1. `run_backtest(config, parquet)` — standalone, loads data every call (CLI use)
//! 2. `BacktestEngine(config, parquet)` — preloads data once, `run_trial(config)` per trial
//!    (Optuna use — 14 threads share one copy of data)
//!
//! ```python
//! # Standalone
//! from rumpy_execution import run_backtest
//! metrics = run_backtest("/tmp/config.yaml", "data/execution_h1.parquet")
//!
//! # Shared data (Optuna)
//! from rumpy_execution import BacktestEngine
//! engine = BacktestEngine("/tmp/base.yaml", "data/execution_h1.parquet")
//! metrics = engine.run_trial("/tmp/trial.yaml")  # thread-safe
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use std::path::PathBuf;

use crate::backtest::{self, BacktestPaths, PreloadedData};
use crate::config::{ExecutionConfig, WalkForwardConfig};

// ---------------------------------------------------------------------------
// Helper: convert BacktestResult to PyDict
// ---------------------------------------------------------------------------

fn result_to_dict<'py>(py: Python<'py>, result: &backtest::BacktestResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let m = &result.metrics;

    dict.set_item("sharpe", m.sharpe)?;
    dict.set_item("ann_return", m.ann_return)?;
    dict.set_item("mdd", m.mdd)?;
    dict.set_item("hit_rate", m.hit_rate)?;
    dict.set_item("avg_turnover", m.avg_turnover)?;
    dict.set_item("calmar", m.calmar)?;
    dict.set_item("martin_ratio", m.martin_ratio)?;
    dict.set_item("ulcer_index", m.ulcer_index)?;
    dict.set_item("deflated_sharpe", m.deflated_sharpe)?;
    dict.set_item("n_bars", m.n_bars)?;
    dict.set_item("benchmark_sharpe", result.benchmark_sharpe)?;
    dict.set_item("min_margin_ratio", result.min_margin_ratio)?;
    // Path-level risk diagnostics (full backtest path; tuner re-computes on
    // its in-sample slice in Python).
    dict.set_item("path_cdar", result.path_cdar)?;
    dict.set_item("path_max_dd", result.path_max_dd)?;

    // Per-fold arrays — diagnostic only. Tuner does not aggregate per-fold;
    // kept exposed so users can inspect per-fold Sharpe/CDaR breakdown.
    let fold_sharpes: Vec<f64> = result.fold_eval.folds.iter().map(|f| f.sharpe).collect();
    let fold_returns: Vec<f64> = result.fold_eval.folds.iter().map(|f| f.ann_return).collect();
    let fold_cdars: Vec<f64> = result.fold_eval.folds.iter().map(|f| f.cdar).collect();
    let fold_mdds: Vec<f64> = result.fold_eval.folds.iter().map(|f| f.mdd).collect();
    let fold_martins: Vec<f64> = result.fold_eval.folds.iter().map(|f| f.martin).collect();
    dict.set_item("fold_sharpes", fold_sharpes)?;
    dict.set_item("fold_returns", fold_returns)?;
    dict.set_item("fold_cdars", fold_cdars)?;
    dict.set_item("fold_mdds", fold_mdds)?;
    dict.set_item("fold_martins", fold_martins)?;

    // Per-bar returns for cross-platform divergence analysis (bit-precision diff).
    let per_bar_ts: Vec<i64> = result.per_bar_returns.iter().map(|(t, _)| *t).collect();
    let per_bar_ret: Vec<f64> = result.per_bar_returns.iter().map(|(_, r)| *r).collect();
    dict.set_item("per_bar_timestamps", per_bar_ts)?;
    dict.set_item("per_bar_returns", per_bar_ret)?;
    // A4 Layer 2: liquid returns from nav_liquid series.
    let per_bar_ret_liq: Vec<f64> = result.per_bar_returns_liquid.iter().map(|(_, r)| *r).collect();
    dict.set_item("per_bar_returns_liquid", per_bar_ret_liq)?;

    // One-sided NAV cap (no-free-money mode) diagnostics.
    dict.set_item("cumulative_skimmed_usd", result.cumulative_skimmed_usd)?;
    let skim_ts: Vec<i64> = result.per_bar_skimmed.iter().map(|(t, _)| *t).collect();
    let skim_amt: Vec<f64> = result.per_bar_skimmed.iter().map(|(_, a)| *a).collect();
    dict.set_item("skim_timestamps", skim_ts)?;
    dict.set_item("skim_amounts", skim_amt)?;

    // A4 Layer 2: per-bar liquidation haircut decomposition for audit.
    let bar_haircut: Vec<f64> = result.per_bar_haircut.iter().map(|(_, h)| *h).collect();
    let bar_haircut_current: Vec<f64> = result.per_bar_haircut_current.clone();
    let bar_haircut_last_known: Vec<f64> = result.per_bar_haircut_last_known.clone();
    let bar_haircut_dlom: Vec<f64> = result.per_bar_haircut_dlom.clone();
    let bar_n_dlom: Vec<u32> = result.per_bar_n_dlom.clone();
    let bar_nav_liquid: Vec<f64> = result.per_bar_nav_liquid.clone();
    dict.set_item("per_bar_haircut", bar_haircut)?;
    dict.set_item("per_bar_haircut_current", bar_haircut_current)?;
    dict.set_item("per_bar_haircut_last_known", bar_haircut_last_known)?;
    dict.set_item("per_bar_haircut_dlom", bar_haircut_dlom)?;
    dict.set_item("per_bar_n_dlom", bar_n_dlom)?;
    dict.set_item("per_bar_nav_liquid", bar_nav_liquid)?;

    // A4 tradeability gate audit fields.
    dict.set_item("n_rejected_trades", result.n_rejected_trades)?;
    dict.set_item("rejected_notional", result.rejected_notional)?;

    // Per-day cost decomposition for cost-drag analysis (H1).
    // Records are emitted at daily frequency; each is one trading day.
    // Cost units: dollars (i.e. NAV currency). Divide by NAV to get fractional drag.
    let n_recs = result.per_bar_returns.len();
    if n_recs > 0 {
        // Use per_bar_returns ordering as the canonical alignment. The
        // simulation emits records in chronological order, and per_bar_returns
        // is built directly from those records, so we know they correspond.
        // We expose the arrays via a separate accessor on BacktestResult.
        let commission: Vec<f64> = result.cost_breakdown.iter().map(|c| c.commission).collect();
        let spread: Vec<f64> = result.cost_breakdown.iter().map(|c| c.spread).collect();
        let impact: Vec<f64> = result.cost_breakdown.iter().map(|c| c.impact).collect();
        let funding: Vec<f64> = result.cost_breakdown.iter().map(|c| c.funding).collect();
        let turnover: Vec<f64> = result.cost_breakdown.iter().map(|c| c.turnover).collect();
        let nav: Vec<f64> = result.cost_breakdown.iter().map(|c| c.nav).collect();
        dict.set_item("per_bar_commission", commission)?;
        dict.set_item("per_bar_spread", spread)?;
        dict.set_item("per_bar_impact", impact)?;
        dict.set_item("per_bar_funding", funding)?;
        dict.set_item("per_bar_turnover", turnover)?;
        dict.set_item("per_bar_nav", nav)?;
    }

    Ok(dict)
}

// ---------------------------------------------------------------------------
// Standalone function (backward compat)
// ---------------------------------------------------------------------------

/// Run a full backtest from YAML config path + parquet path.
/// Returns a dict of metrics. Loads data every call.
#[pyfunction]
#[pyo3(signature = (config_path, execution_parquet, bars_per_year=None))]
fn run_backtest<'py>(
    py: Python<'py>,
    config_path: &str,
    execution_parquet: &str,
    bars_per_year: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let config: ExecutionConfig = {
        let f = std::fs::File::open(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("config open: {e}")))?;
        serde_yaml::from_reader(f)
            .map_err(|e| PyRuntimeError::new_err(format!("config parse: {e}")))?
    };

    // Records are always emitted at daily frequency in both fill modes; the
    // `bars_per_year` arg is kept only for legacy CLI overrides.
    let bpy = bars_per_year.unwrap_or(365.0);
    let wf_config = WalkForwardConfig::default();
    let paths = BacktestPaths {
        execution_h1: PathBuf::from(execution_parquet),
    };

    let result = py.allow_threads(|| {
        backtest::run_backtest(&paths, &config, &wf_config, bpy)
    }).map_err(|e| PyRuntimeError::new_err(format!("backtest: {e}")))?;

    result_to_dict(py, &result)
}

// ---------------------------------------------------------------------------
// BacktestEngine (shared data for Optuna)
// ---------------------------------------------------------------------------

/// Preloaded backtest data. Created once, shared across Optuna worker threads.
///
/// `frozen` = all methods take `&self` = safe for concurrent access from
/// multiple Python threads without the GIL.
#[pyclass(frozen)]
struct BacktestEngine {
    data: PreloadedData,
    wf_config: WalkForwardConfig,
}

#[pymethods]
impl BacktestEngine {
    /// Load parquet + compute config-independent data (covariance, aim, lookups).
    /// Call once before study.optimize().
    #[new]
    #[pyo3(signature = (config_path, execution_parquet, bars_per_year=None))]
    fn new(
        py: Python,
        config_path: &str,
        execution_parquet: &str,
        bars_per_year: Option<f64>,
    ) -> PyResult<Self> {
        let config: ExecutionConfig = {
            let f = std::fs::File::open(config_path)
                .map_err(|e| PyRuntimeError::new_err(format!("config open: {e}")))?;
            serde_yaml::from_reader(f)
                .map_err(|e| PyRuntimeError::new_err(format!("config parse: {e}")))?
        };

        let paths = BacktestPaths {
            execution_h1: PathBuf::from(execution_parquet),
        };

        let data = py.allow_threads(|| {
            PreloadedData::load(&paths, &config)
        }).map_err(|e| PyRuntimeError::new_err(format!("preload: {e}")))?;

        Ok(Self {
            data,
            wf_config: WalkForwardConfig::default(),
        })
    }

    /// Run one trial with trial-specific config. Thread-safe (&self).
    /// GIL is released during the Rust computation.
    #[pyo3(signature = (config_path))]
    fn run_trial<'py>(
        &self,
        py: Python<'py>,
        config_path: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let config: ExecutionConfig = {
            let f = std::fs::File::open(config_path)
                .map_err(|e| PyRuntimeError::new_err(format!("config open: {e}")))?;
            serde_yaml::from_reader(f)
                .map_err(|e| PyRuntimeError::new_err(format!("config parse: {e}")))?
        };

        let wf_config = self.wf_config.clone();

        let result = py.allow_threads(|| {
            backtest::run_trial(&self.data, &config, &wf_config, false)
        }).map_err(|e| PyRuntimeError::new_err(format!("trial: {e}")))?;

        result_to_dict(py, &result)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python module definition.
#[pymodule]
fn rumpy_execution(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_class::<BacktestEngine>()?;
    Ok(())
}
