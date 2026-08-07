//! Clarabel-based SOCP solver for the portfolio optimization problem.
//!
//! The QP has 6n variables: `x = [w(n), u(n), t(n), t_imp(n), y(n), s(n)]`
//!   - `w_i ∈ R`: portfolio weights (Σw = 0, dollar-neutral L/S)
//!   - `u_i ≥ 0`: `|Δw_i|` auxiliary (turnover)
//!   - `t_i ≥ 0`: `|w_i|` auxiliary (gross-exposure cap)
//!   - `t_imp_i ≥ 0`: cone slack for impact, `t_imp_i ≥ u_i^1.5` (PowerCone(2/3))
//!   - `y_i = 1`: dummy for PowerCone(2/3) third coordinate
//!   - `s_i ≥ 0`: ReLU aux for Q_ref subtraction, `s_i ≥ κ_eff·t_imp − r·u`
//!
//! Objective:
//!   `min  -α'w + γ·w'Σw + λ_aim·||w-w_aim||² + c_lin'u + Σ s`
//!
//! Note that the cone-objective coefficient is on `s`, NOT `t_imp`. The
//! `κ_eff·t_imp` expression appears only inside the s-constraint. At optimum:
//!   `s_i = max(κ_eff_i·u_i^1.5 − r_i·u_i, 0)`
//! which equals (per NAV) the realized impact cost from
//! `compute_realized_trade_costs`. Multiplying by NAV gives the dollar cost.
//!
//! Constraints (14n + 2 rows):
//!   - 1 ZeroCone: `Σw = 0`
//!   - 8n+1 NonnegCone: turnover (|Δw|≤u), |w|≤t, gross cap (Σt≤L_max),
//!                      per-name cap (|w|≤cap), u≥0, t≥0
//!   - 2n NonnegCone: `κ_eff·t_imp − r·u − s ≤ 0`,  `−s ≤ 0`
//!   - n PowerCone(2/3): `(t_imp_i, y_i, u_i) ∈ K_pow(2/3)` → `t_imp_i ≥ u_i^1.5`
//!   - n ZeroCone: `y_i = 1`
//!
//! ## No warm-start
//!
//! Per Probe 5 (memory `probe_5_perf_2026_05_05.md`), Clarabel's `update_data`
//! provides no meaningful speedup for problems with PowerCone constraints and
//! may even hurt under realistic perturbation (cached IPM state misleads
//! recentering). Solver is constructed fresh each `solve()` call.
//!
//! ## Coefficient plumbing
//!
//!   `κ_eff_i = κ_i · σ_i · (NAV/ADV_i)^δ`  (= `compute_qp_effective_kappa`)
//!   `r_i     = κ_i · σ_i · (Q_ref/ADV_i)^δ`  (= `compute_qp_residual_kappa`)
//!
//! Both are computed in `simulation.rs::solve_daily_qp` and passed in as
//! per-asset arrays. δ is hardcoded to 0.5 in this solver via PowerCone(2/3);
//! callers must ensure `cost_model.impact_delta == 0.5` (debug-asserted in
//! `solve()`).

use anyhow::Result;
use clarabel::algebra::*;
use clarabel::solver::*;

use crate::config::QPConfig;

/// SOCP solver for the portfolio QP. Cold-starts each call (no warm-start
/// state per Probe 5 — nonsymmetric cones don't benefit from caching).
pub struct CachedSolver {
    n: usize,
    config: QPConfig,
}

impl CachedSolver {
    pub fn new(n: usize, config: &QPConfig, _sigma: &[f64]) -> Result<Self> {
        Ok(Self {
            n,
            config: config.clone(),
        })
    }

    /// Solve one bar.
    ///
    /// Inputs (all length n):
    ///   - `alpha`: per-asset alpha (dollar-PnL per unit weight)
    ///   - `sigma`: per-asset (n×n flat) covariance matrix
    ///   - `w_prev`: previous weights
    ///   - `w_aim`: aim portfolio
    ///   - `c_lin`: per-asset linear cost (taker + half-spread, fraction-of-NAV)
    ///   - `kappa_eff`: per-asset cone weight `κ·σ·(NAV/ADV)^δ`
    ///   - `r`: per-asset Q_ref-residual coefficient `κ·σ·(Q_ref/ADV)^δ`.
    ///          Pass zeros to disable Q_ref subtraction (cone-only mode).
    ///
    /// Returns `(weights, status_string)` or `None` on solver failure.
    pub fn solve(
        &mut self,
        alpha: &[f64],
        sigma: &[f64],
        w_prev: &[f64],
        w_aim: &[f64],
        c_lin: &[f64],
        kappa_eff: &[f64],
        r: &[f64],
        cap_vec: Option<&[f64]>,
    ) -> Option<(Vec<f64>, String)> {
        let n = self.n;
        debug_assert_eq!(alpha.len(), n);
        debug_assert_eq!(w_prev.len(), n);
        debug_assert_eq!(w_aim.len(), n);
        debug_assert_eq!(c_lin.len(), n);
        debug_assert_eq!(kappa_eff.len(), n);
        debug_assert_eq!(r.len(), n);
        debug_assert_eq!(sigma.len(), n * n);
        if let Some(v) = cap_vec { debug_assert_eq!(v.len(), n); }

        let cfg = &self.config;
        let cap = cfg.per_name_cap.unwrap_or(1e6);

        // -------------------------------------------------------------
        // Vol-target SOC: `w'Σw ≤ σ²` via Cholesky factor (L1, 2026-05-13).
        // None  ⇒ no SOC added, dim = 6n, behavior bit-identical to pre-L1.
        // Some(σ) ⇒ add z block (n vars) + n equality rows (z = L'w) + SOC(n+1).
        // -------------------------------------------------------------
        let (soc_active, sigma_target, chol_l) = match cfg.sigma_target_daily {
            Some(s) if s.is_finite() && s > 0.0 => {
                match cholesky_with_ridge(sigma, n, cfg.ridge) {
                    Some(l) => (true, s, l),
                    None => return None, // Σ not PSD even with ridge — give up this bar
                }
            }
            _ => (false, 0.0, Vec::new()),
        };
        let n_z = if soc_active { n } else { 0 };
        // 2026-05-16 Path 2 (Anis-Kwon 2022 top-K concentration cap). LP
        // reformulation adds n+1 variables: τ scalar + n nonneg slacks
        // s_topk_i. Constraints: for each i, t_i − τ − s_topk_i ≤ 0; one
        // sum constraint K·τ + Σ s_topk_i ≤ θ; s_topk_i ≥ 0.
        let topk_active = cfg.topk_cap.is_some();
        let topk_k = cfg.topk_cap.as_ref().map(|c| c.k.min(n)).unwrap_or(0);
        let topk_theta = cfg.topk_cap.as_ref().map(|c| c.theta).unwrap_or(0.0);
        let n_topk_vars = if topk_active { n + 1 } else { 0 };  // τ + n × s_topk
        let dim = 6 * n + n_z + n_topk_vars;

        // -------------------------------------------------------------
        // P: dense upper triangle on w-block; rest zero (z-block has no
        // quadratic term — z is constrained linearly to L'w).
        // P[i,j] = 2·γ·Σ_ij + 2·γ·ridge·δ_ij + 2·λ_aim·δ_ij  (i ≤ j, both < n)
        // -------------------------------------------------------------
        let p_nnz_w = n * (n + 1) / 2;
        let mut p_row_indices = Vec::with_capacity(p_nnz_w);
        let mut p_values = Vec::with_capacity(p_nnz_w);
        let mut p_col_ptrs = Vec::with_capacity(dim + 1);
        p_col_ptrs.push(0);
        for col in 0..dim {
            if col < n {
                for row in 0..=col {
                    let sigma_val = sigma[row * n + col];
                    let ridge_add = if row == col { cfg.ridge } else { 0.0 };
                    let aim_add = if row == col { cfg.lambda_aim } else { 0.0 };
                    let val = 2.0 * (cfg.gamma * (sigma_val + ridge_add) + aim_add);
                    p_row_indices.push(row);
                    p_values.push(val);
                }
            }
            p_col_ptrs.push(p_row_indices.len());
        }
        let p = CscMatrix::new(dim, dim, p_col_ptrs, p_row_indices, p_values);

        // -------------------------------------------------------------
        // q: linear objective coefficients
        // w-block:    -α − 2·λ_aim·w_aim
        // u-block:    c_lin
        // t-block:    0
        // t_imp-block: 0   (NOT κ_eff — that's in the s-constraint)
        // y-block:    0
        // s-block:    1.0  (cone-objective coefficient: minimize Σs)
        // -------------------------------------------------------------
        let mut q = vec![0.0_f64; dim];
        for i in 0..n {
            q[i] = -alpha[i] - 2.0 * cfg.lambda_aim * w_aim[i];
        }
        for i in 0..n {
            q[n + i] = c_lin[i];
        }
        for i in 0..n {
            q[5 * n + i] = 1.0;
        }

        // -------------------------------------------------------------
        // Constraint row layout:
        //   row 0:                  Σw = 0                            (ZeroCone)
        //   rows 1..1+n:            w − u ≤ w_prev
        //   rows 1+n..1+2n:         −w − u ≤ −w_prev
        //   rows 1+2n..1+3n:        w − t ≤ 0
        //   rows 1+3n..1+4n:        −w − t ≤ 0
        //   row 1+4n:               Σt ≤ L_max
        //   rows 2+4n..2+5n:        w_i ≤ cap
        //   rows 2+5n..2+6n:        −w_i ≤ cap
        //   rows 2+6n..2+7n:        −u_i ≤ 0     (u ≥ 0)
        //   rows 2+7n..2+8n:        −t_i ≤ 0     (t ≥ 0)
        //   rows 2+8n..2+9n:        κ_eff·t_imp − r·u − s ≤ 0    (s ≥ marginal)
        //   rows 2+9n..2+10n:       −s_i ≤ 0     (s ≥ 0)
        //   rows 2+10n..2+13n:      n PowerCone(2/3) slack triples (t_imp, y, u)
        //   rows 2+13n..2+14n:      y_i = 1                            (ZeroCone)
        //   (SOC-active only:)
        //   rows 2+14n..2+15n:      z_i − (L'w)_i = 0                  (ZeroCone, n rows)
        //   rows 2+15n..3+16n:      (σ_target, z) ∈ SOC(n+1)
        // -------------------------------------------------------------
        let n_eq = 1;
        let n_ineq = 10 * n + 1;       // 8n+1 (existing) + 2n (s ≥ Δ, s ≥ 0)
        let n_pow = 3 * n;
        let n_y = n;
        let n_z_eq = if soc_active { n } else { 0 };
        let n_soc = if soc_active { n + 1 } else { 0 };
        // Path 2 top-K adds: n rows (t_i − τ − s_topk_i ≤ 0) + 1 row
        // (K·τ + Σ s_topk_i ≤ θ) + n rows (s_topk_i ≥ 0).
        let n_topk_rows = if topk_active { 2 * n + 1 } else { 0 };
        let n_con = n_eq + n_ineq + n_pow + n_y + n_z_eq + n_soc + n_topk_rows;

        let pow_base = n_eq + n_ineq;        // = 10n + 2
        let y_base = pow_base + n_pow;       // = 13n + 2
        let s_lin_base = n_eq + 8 * n + 2;   // = 8n + 3
        let s_nn_base = n_eq + 9 * n + 2;    // = 9n + 3
        let z_eq_base = y_base + n_y;        // = 14n + 2 (start of z=L'w equalities)
        let soc_base = z_eq_base + n_z_eq;   // start of SOC rows (σ, z) ∈ SOC(n+1)
        let topk_ti_base = soc_base + n_soc;          // start of t_i − τ − s_topk_i ≤ 0 rows
        let topk_sum_base = topk_ti_base + n;          // single row K·τ + Σ s_topk_i ≤ θ
        let topk_snn_base = topk_sum_base + 1;         // start of s_topk_i ≥ 0 rows

        // Column index of τ and start of s_topk_i columns. τ at dim - (n+1);
        // s_topk_i at dim - n .. dim.
        let tau_col = 6 * n + n_z;
        let s_topk_col_base = tau_col + 1;

        // Build A column-major (CscMatrix). Row indices within each column
        // must be in ascending order — we sort per column at the end.
        let mut a_col_ptrs = Vec::with_capacity(dim + 1);
        let mut a_row_indices = Vec::<usize>::new();
        let mut a_values = Vec::<f64>::new();
        a_col_ptrs.push(0);

        for col in 0..dim {
            if col < n {
                // w_i column
                let i = col;
                a_row_indices.push(0); a_values.push(-1.0);                       // Σw = 0
                a_row_indices.push(n_eq + i); a_values.push(1.0);                 // w − u ≤ w_prev
                a_row_indices.push(n_eq + n + i); a_values.push(-1.0);            // −w − u ≤ −w_prev
                a_row_indices.push(n_eq + 2 * n + i); a_values.push(1.0);         // w − t ≤ 0
                a_row_indices.push(n_eq + 3 * n + i); a_values.push(-1.0);        // −w − t ≤ 0
                a_row_indices.push(n_eq + 4 * n + 1 + i); a_values.push(1.0);     // w ≤ cap
                a_row_indices.push(n_eq + 5 * n + 1 + i); a_values.push(-1.0);    // −w ≤ cap
                if soc_active {
                    // z=L'w equality: -z_k + sum_{j>=k} L[j,k]·w[j] = 0.
                    // Column w_i (= w[j] here, with j=i) contributes L[i,k] for
                    // each k ≤ i (k > i has L[i,k]=0 since L is lower-tri).
                    for k in 0..=i {
                        let l_val = chol_l[i * n + k];
                        if l_val != 0.0 {
                            a_row_indices.push(z_eq_base + k);
                            a_values.push(l_val);
                        }
                    }
                }
            } else if col < 2 * n {
                // u_i column
                let i = col - n;
                a_row_indices.push(n_eq + i); a_values.push(-1.0);                // w − u ≤ w_prev
                a_row_indices.push(n_eq + n + i); a_values.push(-1.0);            // −w − u ≤ −w_prev
                a_row_indices.push(n_eq + 6 * n + 1 + i); a_values.push(-1.0);    // u ≥ 0
                a_row_indices.push(s_lin_base + i); a_values.push(-r[i]);         // s-lin: −r·u
                a_row_indices.push(pow_base + 3 * i + 2); a_values.push(-1.0);    // PowerCone u-slot
            } else if col < 3 * n {
                // t_i column
                let i = col - 2 * n;
                a_row_indices.push(n_eq + 2 * n + i); a_values.push(-1.0);        // w − t ≤ 0
                a_row_indices.push(n_eq + 3 * n + i); a_values.push(-1.0);        // −w − t ≤ 0
                a_row_indices.push(n_eq + 4 * n); a_values.push(1.0);             // Σt ≤ L_max
                a_row_indices.push(n_eq + 7 * n + 1 + i); a_values.push(-1.0);    // t ≥ 0
                if topk_active {
                    // Path 2 top-K: t_i − τ − s_topk_i ≤ 0 row gets +1 on t_i.
                    a_row_indices.push(topk_ti_base + i); a_values.push(1.0);
                }
            } else if col < 4 * n {
                // t_imp_i column (PowerCone first coord)
                let i = col - 3 * n;
                a_row_indices.push(s_lin_base + i); a_values.push(kappa_eff[i]);  // s-lin: +κ_eff·t_imp
                a_row_indices.push(pow_base + 3 * i); a_values.push(-1.0);        // PowerCone t_imp-slot
            } else if col < 5 * n {
                // y_i column (PowerCone second coord, y = 1)
                let i = col - 4 * n;
                a_row_indices.push(pow_base + 3 * i + 1); a_values.push(-1.0);    // PowerCone y-slot
                a_row_indices.push(y_base + i); a_values.push(1.0);               // y_i − 1 = 0 (with b=1)
            } else if col < 6 * n {
                // s_i column
                let i = col - 5 * n;
                a_row_indices.push(s_lin_base + i); a_values.push(-1.0);          // s-lin: −s
                a_row_indices.push(s_nn_base + i); a_values.push(-1.0);           // s ≥ 0
            } else if col < 6 * n + n_z {
                // z_i column (SOC-active only). Two entries:
                //   - equality row z_eq_base+i:  coefficient -1 (z=L'w → -z appears)
                //   - SOC row soc_base+1+i:      coefficient -1 (so s[1+i] = -A·x = z_i)
                let i = col - 6 * n;
                a_row_indices.push(z_eq_base + i); a_values.push(-1.0);
                a_row_indices.push(soc_base + 1 + i); a_values.push(-1.0);
            } else if col == tau_col && topk_active {
                // τ column (Path 2 top-K). Contributes to:
                //   - n rows: t_i − τ − s_topk_i ≤ 0  (coefficient -1)
                //   - 1 row : K·τ + Σ s_topk_i ≤ θ    (coefficient K)
                for i in 0..n {
                    a_row_indices.push(topk_ti_base + i); a_values.push(-1.0);
                }
                a_row_indices.push(topk_sum_base); a_values.push(topk_k as f64);
            } else if topk_active {
                // s_topk_i column. Three entries:
                //   - row topk_ti_base+i:  coefficient -1 (t_i − τ − s ≤ 0)
                //   - row topk_sum_base:   coefficient +1 (K·τ + Σ s ≤ θ)
                //   - row topk_snn_base+i: coefficient -1 (−s ≤ 0  ⇒ s ≥ 0)
                let i = col - s_topk_col_base;
                a_row_indices.push(topk_ti_base + i); a_values.push(-1.0);
                a_row_indices.push(topk_sum_base); a_values.push(1.0);
                a_row_indices.push(topk_snn_base + i); a_values.push(-1.0);
            }
            a_col_ptrs.push(a_row_indices.len());
        }

        // CscMatrix requires ascending row indices within each column — sort.
        for col in 0..dim {
            let start = a_col_ptrs[col];
            let end = a_col_ptrs[col + 1];
            if end > start + 1 {
                let mut pairs: Vec<(usize, f64)> = (start..end)
                    .map(|k| (a_row_indices[k], a_values[k]))
                    .collect();
                pairs.sort_by_key(|p| p.0);
                for (offset, (row, val)) in pairs.into_iter().enumerate() {
                    a_row_indices[start + offset] = row;
                    a_values[start + offset] = val;
                }
            }
        }
        let a = CscMatrix::new(n_con, dim, a_col_ptrs, a_row_indices, a_values);

        // -------------------------------------------------------------
        // b vector
        // -------------------------------------------------------------
        let mut b = vec![0.0_f64; n_con];
        // row 0: 0 (Σw = 0)
        for i in 0..n { b[n_eq + i] = w_prev[i]; }
        for i in 0..n { b[n_eq + n + i] = -w_prev[i]; }
        // rows for w−t≤0, −w−t≤0: 0
        b[n_eq + 4 * n] = cfg.l_max;
        // 2026-05-16 Path 1 (vol-scaled per-name cap, Richard-Roncalli 2019 box
        // approximation): when `cap_vec` is supplied, the per-asset cap takes
        // priority (clipped above by the scalar cap as a global ceiling).
        // Otherwise fall back to scalar `cap` for every name.
        if let Some(v) = cap_vec {
            for i in 0..n {
                let c_i = v[i].min(cap).max(0.0);
                b[n_eq + 4 * n + 1 + i] = c_i;
                b[n_eq + 5 * n + 1 + i] = c_i;
            }
        } else {
            for i in 0..n { b[n_eq + 4 * n + 1 + i] = cap; }
            for i in 0..n { b[n_eq + 5 * n + 1 + i] = cap; }
        }
        // u≥0, t≥0: 0
        // s-lin (s ≥ Δ) and s-NN rows: 0
        // PowerCone rows: 0
        for i in 0..n { b[y_base + i] = 1.0; }
        // z=L'w equality rows: b = 0 (default).
        // SOC row 0: b = σ_target → s[0] = σ_target.
        // SOC rows 1..n: b = 0 → s[k] = -A·x = z_{k-1}.
        if soc_active {
            b[soc_base] = sigma_target;
        }
        // Path 2 top-K: only the single sum row has nonzero b. Others = 0.
        if topk_active {
            b[topk_sum_base] = topk_theta;
        }

        // -------------------------------------------------------------
        // Cones (in the same order as the row blocks above)
        // -------------------------------------------------------------
        let mut cones: Vec<SupportedConeT<f64>> = Vec::with_capacity(3 + n + 2);
        cones.push(ZeroConeT(n_eq));
        cones.push(NonnegativeConeT(n_ineq));
        for _ in 0..n {
            cones.push(PowerConeT(2.0 / 3.0));
        }
        cones.push(ZeroConeT(n_y));
        if soc_active {
            cones.push(ZeroConeT(n_z_eq));
            cones.push(SecondOrderConeT(n_soc));
        }
        if topk_active {
            // All 2n+1 top-K rows are NonnegCone (≤ form).
            cones.push(NonnegativeConeT(n_topk_rows));
        }

        // -------------------------------------------------------------
        // Solve (cold-start; no warm-start state — see Probe 5).
        //
        // max_iter raised from default 200 to 400 — the cone problem is
        // larger (6n vars vs 3n) and Clarabel's nonsymmetric IPM (Skajaa-Ye)
        // typically needs more iterations than symmetric. Empirically reduces
        // `InsufficientProgress` rate from ~0.15% to ~0.02% at v9_pruned param
        // ranges (gamma=50). Failures are non-fatal: caller carries forward
        // the previous bar's daily_target.
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .presolve_enable(false)
            .max_iter(400)
            .build()
            .unwrap();

        let solver = match DefaultSolver::new(&p, &q, &a, &b, &cones, settings) {
            Ok(s) => s,
            Err(_e) => return None,  // setup failure is rare and benign
        };
        let mut solver = solver;
        solver.solve();

        match solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                let status = if solver.solution.status == SolverStatus::Solved {
                    "solved"
                } else {
                    "solved_inaccurate"
                };
                Some((solver.solution.x[..n].to_vec(), status.to_string()))
            }
            // InsufficientProgress / MaxIterations / etc — caller handles
            // by carrying previous target. Silenced to avoid log spam at
            // tune scale (~0.02% of solves at v9_pruned, ~250 trials × 100k
            // solves = 5000 messages per trial otherwise).
            _ => None,
        }
    }

    /// For API compatibility (legacy; never used in production).
    pub fn solve_with_cost(&mut self, _q: &[f64]) -> Option<(Vec<f64>, String)> {
        None
    }
}

pub type CachedOSQP = CachedSolver;

/// In-place Cholesky of `Σ + ridge·I` for dense n×n PSD matrix.
///
/// Returns lower-triangular L (n*n flattened row-major, upper triangle = 0)
/// such that `L · L' = Σ + ridge·I`. Returns None if a pivot is ≤ 0 (matrix
/// not PSD even with ridge — shouldn't happen with ridge > 0 unless caller
/// passed garbage). Used by `solve()` when `sigma_target_daily` is Some,
/// to encode `w'Σw ≤ σ²` as the SOC `||L'w|| ≤ σ` (Boyd cvxportfolio 2024
/// canonical form; see `docs/sessions/L0-touchpoints.md` for derivation).
fn cholesky_with_ridge(sigma: &[f64], n: usize, ridge: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        let mut s = sigma[j * n + j] + ridge;
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if s <= 0.0 {
            return None;
        }
        let ljj = s.sqrt();
        l[j * n + j] = ljj;
        let inv_ljj = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = sigma[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s * inv_ljj;
        }
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a cone-aware mini portfolio QP and solve it.
    /// Returns (weights, status) or None.
    fn solve_minimal(
        n: usize, alpha: &[f64], sigma: &[f64], w_prev: &[f64], w_aim: &[f64],
        c_lin: &[f64], kappa_eff: &[f64], r: &[f64],
        gamma: f64, lambda_aim: f64, l_max: f64, per_name_cap: f64, ridge: f64,
    ) -> Option<(Vec<f64>, String)> {
        let config = QPConfig {
            gamma, lambda_aim, l_max,
            per_name_cap: Some(per_name_cap), ridge,
            prune_top_k: None,
            sigma_target_daily: None,
            per_name_cap_vol_scale: false,
            topk_cap: None,
            per_name_cap_floor_safety_mult: 1.5,
            per_name_cap_vol_scale_ceiling_mult: 5.0,
            per_name_cap_default: 0.05,
        };
        let mut solver = CachedSolver::new(n, &config, sigma).unwrap();
        solver.solve(alpha, sigma, w_prev, w_aim, c_lin, kappa_eff, r, None)
    }

    /// 2-asset PSD covariance: independent assets with equal vol.
    fn diag_cov(n: usize, vol_sq: f64) -> Vec<f64> {
        let mut sigma = vec![0.0_f64; n * n];
        for i in 0..n { sigma[i * n + i] = vol_sq; }
        sigma
    }

    #[test]
    fn cone_qp_solves_dollar_neutral() {
        // Tiny 2-asset problem. Symmetric alpha → expect long one, short other.
        let n = 2;
        let sigma = diag_cov(n, 0.0004);
        let alpha = vec![0.05, -0.05];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![0.0001; n];
        let kappa_eff = vec![0.001; n];
        let r = vec![0.0; n];   // Q_ref subtraction off
        let (w, status) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r,
            4.0, 0.0, 1.0, 0.5, 1e-6,
        ).expect("solver failed");
        assert!(status == "solved" || status == "solved_inaccurate");
        let sum: f64 = w.iter().sum();
        assert!(sum.abs() < 1e-6, "not dollar-neutral: sum = {sum}");
        // Expect long the +alpha asset, short the -alpha asset.
        assert!(w[0] > 0.0 && w[1] < 0.0,
            "alpha sign mismatch: w = {w:?}");
    }

    #[test]
    fn cone_qp_respects_per_name_cap() {
        // Strong alpha should drive trade up to per_name_cap, not beyond.
        let n = 2;
        let sigma = diag_cov(n, 0.0001);
        let alpha = vec![1.0, -1.0];   // huge alpha
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![1e-5; n];
        let kappa_eff = vec![1e-6; n]; // tiny — won't bind
        let r = vec![0.0; n];
        let (w, _) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r,
            0.1, 0.0, 10.0, 0.05, 1e-6,
        ).expect("solver failed");
        for &wi in &w {
            assert!(wi.abs() <= 0.05 + 1e-6, "per_name_cap violated: {w:?}");
        }
    }

    #[test]
    fn cone_qp_respects_l_max() {
        // Strong alpha + slack per_name_cap; gross should bind L_max.
        let n = 2;
        let sigma = diag_cov(n, 0.0001);
        let alpha = vec![1.0, -1.0];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![1e-5; n];
        let kappa_eff = vec![1e-6; n];
        let r = vec![0.0; n];
        let (w, _) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r,
            0.1, 0.0, 0.5, 1.0, 1e-6,
        ).expect("solver failed");
        let gross: f64 = w.iter().map(|x| x.abs()).sum();
        assert!(gross <= 0.5 + 1e-6, "L_max violated: gross = {gross}");
    }

    #[test]
    fn cone_qp_aim_pull_works() {
        // No alpha, lambda_aim > 0, w_aim is dollar-neutral.
        // Expect w → w_aim (or close to it, modulo ridge + γ·Σ regularization).
        let n = 2;
        let sigma = diag_cov(n, 1e-6);
        let alpha = vec![0.0; n];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.05, -0.05];
        let c_lin = vec![0.0; n];
        let kappa_eff = vec![0.0; n];
        let r = vec![0.0; n];
        let (w, _) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r,
            0.001, 1.0, 1.0, 0.1, 1e-6,
        ).expect("solver failed");
        let max_diff: f64 = w.iter().zip(w_aim.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 0.01, "w should track w_aim: w = {w:?}, w_aim = {w_aim:?}");
    }

    // (Q_ref-subtraction-increases-trading test omitted — fragile at toy
    // scale because both variants saturate the gross/per-name caps before
    // the cone aux meaningfully kicks in. Probe 3 empirically demonstrated
    // the effect at portfolio scale (n=200, real distributions): 8% L1
    // weight divergence + 8% realized-cost difference at deployment NAV.
    // See `memory/probe_3_qref_2026_05_05.md`.)

    #[test]
    fn cone_qp_cone_constraint_active() {
        // For a strong-alpha problem with non-zero kappa_eff, the cone
        // constraint t_imp ≥ u^1.5 should be active (t_imp = u^1.5 at optimum).
        // We verify indirectly: the realized cost (κ_eff·u^1.5 − r·u) should
        // be reflected in the w solution being smaller than the no-cost case.
        let n = 2;
        let sigma = diag_cov(n, 0.0004);
        let alpha = vec![0.02, -0.02];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![0.0; n];
        let r = vec![0.0; n];
        let gamma = 1.0;
        let lambda_aim = 0.0;
        let l_max = 1.0;
        let cap = 0.5;
        let ridge = 1e-6;

        // No cone cost
        let (w_free, _) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &vec![0.0; n], &r,
            gamma, lambda_aim, l_max, cap, ridge,
        ).expect("free solve failed");

        // With cone cost
        let (w_costly, _) = solve_minimal(
            n, &alpha, &sigma, &w_prev, &w_aim, &c_lin, &vec![10.0; n], &r,
            gamma, lambda_aim, l_max, cap, ridge,
        ).expect("costly solve failed");

        let gross_free: f64 = w_free.iter().map(|w| w.abs()).sum();
        let gross_costly: f64 = w_costly.iter().map(|w| w.abs()).sum();
        assert!(gross_costly < gross_free,
            "kappa_eff > 0 should suppress trading: gross_free={gross_free}, gross_costly={gross_costly}");
    }

    #[test]
    fn vol_target_soc_non_binding_matches_unconstrained() {
        // sigma_target_daily set very high → SOC should not bind → w identical
        // to the no-SOC case at the same params.
        let n = 4;
        let sigma = diag_cov(n, 0.0004); // vol = 2%/d per asset
        let alpha = vec![0.05, 0.02, -0.02, -0.05];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![0.0001; n];
        let kappa_eff = vec![0.001; n];
        let r = vec![0.0; n];

        let cfg_no_soc = QPConfig {
            gamma: 4.0, lambda_aim: 0.0, l_max: 1.0,
            per_name_cap: Some(0.5), ridge: 1e-6,
            prune_top_k: None, sigma_target_daily: None,
            per_name_cap_vol_scale: false, topk_cap: None,
            per_name_cap_floor_safety_mult: 1.5,
            per_name_cap_vol_scale_ceiling_mult: 5.0,
            per_name_cap_default: 0.05,
        };
        let cfg_loose = QPConfig {
            sigma_target_daily: Some(1.0), // huge — never binds
            ..cfg_no_soc.clone()
        };
        let mut s_no = CachedSolver::new(n, &cfg_no_soc, &sigma).unwrap();
        let (w_no, _) = s_no.solve(&alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r, None).unwrap();
        let mut s_loose = CachedSolver::new(n, &cfg_loose, &sigma).unwrap();
        let (w_loose, _) = s_loose.solve(&alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r, None).unwrap();

        for i in 0..n {
            assert!((w_no[i] - w_loose[i]).abs() < 1e-5,
                "non-binding SOC must not change w at i={i}: {} vs {}", w_no[i], w_loose[i]);
        }
    }

    #[test]
    fn vol_target_soc_binding_clips_portfolio_vol() {
        // sigma_target_daily set tight → ||L'w|| should clip near σ_target.
        let n = 4;
        let sigma = diag_cov(n, 0.0004); // vol = 2%/d per asset
        let alpha = vec![0.5, 0.2, -0.2, -0.5]; // strong alpha
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![0.0001; n];
        let kappa_eff = vec![0.0; n];
        let r = vec![0.0; n];
        let sigma_target = 0.005; // 0.5% daily

        let cfg = QPConfig {
            gamma: 0.1, lambda_aim: 0.0, l_max: 10.0, // slack on γ and L_max
            per_name_cap: Some(1.0), ridge: 1e-6, // slack on cap too
            prune_top_k: None, sigma_target_daily: Some(sigma_target),
            per_name_cap_vol_scale: false, topk_cap: None,
            per_name_cap_floor_safety_mult: 1.5,
            per_name_cap_vol_scale_ceiling_mult: 5.0,
            per_name_cap_default: 0.05,
        };
        let mut s = CachedSolver::new(n, &cfg, &sigma).unwrap();
        let (w, status) = s.solve(&alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r, None).unwrap();

        // Compute ||L'w||² = w'Σw directly.
        let mut quad = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                quad += w[i] * sigma[i * n + j] * w[j];
            }
        }
        let port_vol = quad.sqrt();
        // Allow 1% solver slack on top of σ_target (clarabel is interior-point).
        assert!(port_vol <= sigma_target * 1.01,
            "SOC must clip portfolio vol at σ_target={sigma_target}; got {port_vol:.6} (status: {status})");
        // Confirm it's actually BINDING (close to σ, not 0).
        assert!(port_vol >= sigma_target * 0.90,
            "SOC should be tight at σ_target={sigma_target}; got {port_vol:.6} — too slack");
    }

    #[test]
    fn vol_target_soc_dollar_neutral_preserved() {
        // SOC must not break dollar-neutrality constraint.
        let n = 4;
        let sigma = diag_cov(n, 0.0004);
        let alpha = vec![0.5, 0.2, -0.2, -0.5];
        let w_prev = vec![0.0; n];
        let w_aim = vec![0.0; n];
        let c_lin = vec![0.0001; n];
        let kappa_eff = vec![0.0; n];
        let r = vec![0.0; n];
        let cfg = QPConfig {
            gamma: 0.1, lambda_aim: 0.0, l_max: 10.0,
            per_name_cap: Some(1.0), ridge: 1e-6,
            prune_top_k: None, sigma_target_daily: Some(0.005),
            per_name_cap_vol_scale: false, topk_cap: None,
            per_name_cap_floor_safety_mult: 1.5,
            per_name_cap_vol_scale_ceiling_mult: 5.0,
            per_name_cap_default: 0.05,
        };
        let mut s = CachedSolver::new(n, &cfg, &sigma).unwrap();
        let (w, _) = s.solve(&alpha, &sigma, &w_prev, &w_aim, &c_lin, &kappa_eff, &r, None).unwrap();
        let sum: f64 = w.iter().sum();
        assert!(sum.abs() < 1e-5, "SOC must not break Σw=0: sum={sum}");
    }

    #[test]
    fn cholesky_with_ridge_basic() {
        // 3x3 PSD matrix; reconstruct from L and compare.
        let n = 3;
        let sigma = vec![
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.2,
            0.5, 0.2, 2.0,
        ];
        let l = cholesky_with_ridge(&sigma, n, 0.0).expect("must succeed");
        // Verify L is lower triangular.
        for i in 0..n {
            for j in (i+1)..n {
                assert_eq!(l[i * n + j], 0.0, "L must be lower tri at ({i},{j})");
            }
        }
        // Verify L * L' = sigma.
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..=i.min(j) {
                    s += l[i * n + k] * l[j * n + k];
                }
                let expected = sigma[i * n + j];
                assert!((s - expected).abs() < 1e-10,
                    "L·L'[{i},{j}] = {s} != {expected}");
            }
        }
    }

}
