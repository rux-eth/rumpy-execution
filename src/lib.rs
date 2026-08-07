//! QP-based portfolio optimizer for crypto L/S execution.
//!
//! Ports the champion execution model from Python (`train/rumpy_train/execution/`)
//! to Rust for production use via maestro sidecar.
//!
//! Pipeline:
//!   1. Load alpha scores (5 horizons) → z-score → GP13 persistence blend → winsorize
//!   2. EWMA factor covariance (8 PCA, 45d halflife)
//!   3. Aim portfolio (Markowitz Σ⁻¹·α/γ, dollar-neutral)
//!   4. Cost model (fees + spread + sqrt-impact, NAV-aware kappa)
//!   5. OSQP impact QP per bar (warm-started, 2 impact iterations)
//!   6. Dynamic gamma (dispersion-based counter-cyclical)
//!   7. Vol targeting + sizing → final weights

pub mod solver;
pub mod covariance;
pub mod alpha;
pub mod aim;
pub mod cost;
pub mod ml_cost;
pub mod metrics;
pub mod walkforward;
pub mod universe;
pub mod gates;
pub mod backtest;
pub mod diagnostics;
pub mod position;
pub mod exchange;
pub mod simulation;
pub mod config;
pub mod pipeline;
pub mod io;
pub mod preproc;

#[cfg(feature = "python")]
pub mod python;
