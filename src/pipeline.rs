//! Public types emitted by the execution simulator.
//!
//! Historically this module also contained `run_qp_impact` — a parallel
//! QP-solver loop used by `execution solve` and (downstream) by maestro's
//! live `ExecutionComputeService`. That implementation diverged from
//! `simulation::SimulationEngine` (the loop used by `execution backtest`
//! and the tuner): different per-day vs per-bar solve cadence, no
//! covariance shrinkage, no funding-aware alpha, no `blend_rho`, simpler
//! cost accounting. The 2026-05-17 parity test on identical prod_v5 inputs
//! measured Pearson r = 0.75 and 7% sign-flips between the two paths.
//!
//! Phase 2 consolidation (2026-05-18) deleted `run_qp_impact`. `execution
//! solve` now composes `features::execution::build_execution` (build the
//! unified H1 parquet in a temp dir) followed by `backtest::run_backtest`
//! (the SimulationEngine path). Live + backtest share the same QP loop by
//! construction. See `memory:session_2026_05_17_cost_consolidation_plan.md`.
//!
//! The `WeightOutput` and `SolveStats` types remain here because many
//! sibling modules (gates, risk, sizing, universe, diagnostics, metrics,
//! io, backtest) already `use crate::pipeline::{WeightOutput, SolveStats}`.
//! Renaming the module to e.g. `types.rs` is Phase 3 cleanup.

/// Output weight for one (timestamp, symbol).
#[derive(Debug, Clone)]
pub struct WeightOutput {
    pub timestamp: i64,
    pub symbol: String,
    /// Raw QP-output weight (warm-start for the next bar's solve).
    pub weight_qp: f64,
    /// Weight after any post-QP scaling (vol target, sizing, etc.).
    pub weight_final: f64,
}

/// Per-bar stats from the QP solve loop. Populated by the simulator.
#[derive(Debug, Default, Clone)]
pub struct SolveStats {
    pub n_bars: usize,
    pub n_solved: usize,
    pub n_failed_cov: usize,
    pub n_failed_solve: usize,
}
