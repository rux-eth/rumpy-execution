//! Probe 3 (Q_ref residual portfolio magnitude).
//!
//! Goal: measure how much the optimizer's solution differs between:
//!   Variant A — cone term `Σ κ_eff·t_imp,  t_imp ≥ u^1.5`
//!               (penalizes the FULL realized cost from Q=0 upward; over-counts
//!                below Q_ref where realized cost is 0)
//!   Variant C — cone + ReLU aux:
//!               `t_imp ≥ u^1.5`,  `s ≥ κ_eff·t_imp − r·u`,  `s ≥ 0`
//!               objective: `Σ s`
//!               (penalizes only the impact ABOVE the Q_ref baseline,
//!                exactly matching the realized cost formula)
//!
//! ## Why this matters
//!
//! At deployment NAV ≈ $1M, per-name-cap × NAV ≈ $52k > Q_ref ($13,827), so
//! most trades are in the "above Q_ref" regime → variant A and C should
//! agree closely. At small NAV ≈ $10k, max trade ≈ $520 << Q_ref, so all
//! trades are "below Q_ref" → variant A over-charges the entire impact while
//! variant C correctly charges zero. The two formulations should diverge
//! materially at small NAV and converge at large NAV.
//!
//! ## Math derivation (the load-bearing simplification)
//!
//! Realized formula (simulation.rs:1547-1559):
//!   impact_usd = κ·σ·|Q|·max((|Q|/ADV)^δ − (Q_ref/ADV)^δ, 0)
//!
//! For δ=0.5 and |Q| ≥ Q_ref-effective, this expands as:
//!   impact = κ·σ·|Q|^1.5/ADV^0.5  −  κ·σ·(Q_ref/ADV)^0.5·|Q|
//!          = κ_eff·NAV·u^1.5      −  r·u·NAV
//!     where κ_eff = κ·σ·(NAV/ADV)^0.5  (matches compute_qp_effective_kappa)
//!     and  r = κ·σ·(Q_ref/ADV)^0.5
//!
//! For |Q| < Q_ref-effective, impact = 0, so we want max(...) at the
//! per-asset level. ReLU aux `s ≥ max(κ_eff·u^1.5 − r·u, 0)` captures this.
//!
//! Threshold (where s flips from 0 to active):
//!   κ_eff·u^1.5 = r·u   ⇒   u^0.5 = r/κ_eff = (Q_ref/NAV)^0.5
//!   u_threshold = Q_ref/NAV     ← uniform across assets
//!
//! ## Tests
//!
//! T1 — variant C unit identity: at solution, `s_i · NAV` matches
//!      `compute_realized_trade_costs(Δw_i·NAV, ..., q_ref=Q_ref).impact`
//!      per asset. Pin via formula on solver outputs.
//!
//! T2 — A-vs-C divergence across NAV regimes (10k, 100k, 1M, 10M):
//!      ||w_A − w_C||_1 / gross_A          (weight similarity)
//!      |realized(A)·NAV − realized(C)·NAV| / realized(C)·NAV  (cost similarity)
//!      Frac of trades below Q_ref in each
//!      Cone/ReLU obj contribution
//!
//! T3 — drop-subtraction recommendation at deployment NAV ($1M):
//!      L1 weight diff / gross < 5%  AND  realized_diff / realized_C < 10%
//!
//! Run via: `cargo run --release -p rumpy-execution --example probe_qref`

use clarabel::{algebra::*, solver::*};
use rumpy_execution::simulation::compute_realized_trade_costs;

const N: usize = 200;
const N_FACTORS: usize = 8;
const DELTA: f64 = 0.5;
const Q_REF_USD: f64 = 13_827.0;

struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self { Self(seed.max(1)) }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x; x
    }
    fn uniform(&mut self) -> f64 { self.next_u32() as f64 / (u32::MAX as f64) }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Per-asset raw market parameters (NAV-independent).
#[derive(Clone)]
struct AssetParams {
    kappa: f64,
    sigma_d: f64,
    adv: f64,
}

struct Problem {
    sigma: Vec<f64>,         // n*n cov
    alpha: Vec<f64>,
    w_prev: Vec<f64>,
    w_aim: Vec<f64>,
    c_lin: Vec<f64>,
    assets: Vec<AssetParams>,
    gamma: f64,
    lambda_aim: f64,
    l_max: f64,
    per_name_cap: f64,
    ridge: f64,
}

fn build_problem(seed: u32) -> Problem {
    let mut rng = Rng::new(seed);

    // Σ from factor model
    let mut f = vec![0.0_f64; N * N_FACTORS];
    for i in 0..N {
        for k in 0..N_FACTORS { f[i * N_FACTORS + k] = 0.02 * rng.normal(); }
    }
    let mut sigma = vec![0.0_f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for k in 0..N_FACTORS { s += f[i * N_FACTORS + k] * f[j * N_FACTORS + k]; }
            sigma[i * N + j] = s;
        }
        sigma[i * N + i] += 0.0004 + 0.0001 * rng.uniform();
    }

    let alpha: Vec<f64> = (0..N).map(|_| 0.05 * rng.normal()).collect();

    let mut w_prev: Vec<f64> = (0..N).map(|_| 0.005 * rng.normal()).collect();
    let mean: f64 = w_prev.iter().sum::<f64>() / N as f64;
    for w in &mut w_prev { *w -= mean; }

    let mut w_aim: Vec<f64> = (0..N).map(|_| 0.005 * rng.normal()).collect();
    let mean: f64 = w_aim.iter().sum::<f64>() / N as f64;
    for w in &mut w_aim { *w -= mean; }

    let c_lin: Vec<f64> = (0..N).map(|_| 8.5e-4).collect();

    let assets: Vec<AssetParams> = (0..N).map(|_| {
        let kappa = 0.2 + 1.3 * rng.uniform();
        let sigma_d = 0.02 + 0.06 * rng.uniform();
        let log_adv = 5.0 + 4.0 * rng.uniform();
        AssetParams { kappa, sigma_d, adv: 10.0_f64.powf(log_adv) }
    }).collect();

    Problem {
        sigma, alpha, w_prev, w_aim, c_lin, assets,
        gamma: 4.795,
        lambda_aim: 0.054,
        l_max: 1.5,
        per_name_cap: 0.052,
        ridge: 1e-6,
    }
}

/// Compute κ_eff and r for variant C (NAV-dependent).
fn compute_coefficients(p: &Problem, nav: f64) -> (Vec<f64>, Vec<f64>) {
    let kappa_eff: Vec<f64> = p.assets.iter().map(|a| {
        a.kappa * a.sigma_d * (nav / a.adv).powf(DELTA)
    }).collect();
    let r: Vec<f64> = p.assets.iter().map(|a| {
        a.kappa * a.sigma_d * (Q_REF_USD / a.adv).powf(DELTA)
    }).collect();
    (kappa_eff, r)
}

// =============================================================
// Variant A: cone-only, no Q_ref subtraction
// 5n vars: [w, u, t, t_imp, y]
// objective: -α'w + γw'Σw + λ_aim||w-w_aim||² + c_lin'u + Σ κ_eff·t_imp
// =============================================================
fn build_variant_a(p: &Problem, nav: f64) -> (
    CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>
) {
    let (kappa_eff, _r) = compute_coefficients(p, nav);
    let n = N;
    let dim = 5 * n;

    // P
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    for col in 0..dim {
        if col < n {
            for row in 0..=col {
                let s = p.sigma[row * n + col];
                let ridge = if row == col { p.ridge } else { 0.0 };
                let aim = if row == col { p.lambda_aim } else { 0.0 };
                let val = 2.0 * (p.gamma * (s + ridge) + aim);
                p_indices.push(row);
                p_data.push(val);
            }
        }
        p_indptr.push(p_indices.len());
    }
    let p_csc = CscMatrix::new(dim, dim, p_indptr, p_indices, p_data);

    // q
    let mut q = vec![0.0_f64; dim];
    for i in 0..n { q[i] = -p.alpha[i] - 2.0 * p.lambda_aim * p.w_aim[i]; }
    for i in 0..n { q[n + i] = p.c_lin[i]; }
    for i in 0..n { q[3 * n + i] = kappa_eff[i]; }

    let n_eq = 1;
    let n_ineq = 8 * n + 1;
    let n_pow = 3 * n;
    let n_y = n;
    let n_con = n_eq + n_ineq + n_pow + n_y;

    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();
    let pow_base = n_eq + n_ineq;
    let y_base = pow_base + n_pow;

    for col in 0..dim {
        if col < n {
            let i = col;
            a_indices.push(0); a_data.push(-1.0);
            a_indices.push(n_eq + i); a_data.push(1.0);
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 2 * n + i); a_data.push(1.0);
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n + 1 + i); a_data.push(1.0);
            a_indices.push(n_eq + 5 * n + 1 + i); a_data.push(-1.0);
        } else if col < 2 * n {
            let i = col - n;
            a_indices.push(n_eq + i); a_data.push(-1.0);
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 6 * n + 1 + i); a_data.push(-1.0);
            a_indices.push(pow_base + 3 * i + 2); a_data.push(-1.0);
        } else if col < 3 * n {
            let i = col - 2 * n;
            a_indices.push(n_eq + 2 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n); a_data.push(1.0);
            a_indices.push(n_eq + 7 * n + 1 + i); a_data.push(-1.0);
        } else if col < 4 * n {
            let i = col - 3 * n;
            a_indices.push(pow_base + 3 * i); a_data.push(-1.0);
        } else {
            let i = col - 4 * n;
            a_indices.push(pow_base + 3 * i + 1); a_data.push(-1.0);
            a_indices.push(y_base + i); a_data.push(1.0);
        }
        a_indptr.push(a_indices.len());
    }
    sort_csc_columns(&a_indptr, &mut a_indices, &mut a_data);
    let a_csc = CscMatrix::new(n_con, dim, a_indptr, a_indices, a_data);

    let mut b = vec![0.0_f64; n_con];
    for i in 0..n { b[n_eq + i] = p.w_prev[i]; }
    for i in 0..n { b[n_eq + n + i] = -p.w_prev[i]; }
    b[n_eq + 4 * n] = p.l_max;
    for i in 0..n { b[n_eq + 4 * n + 1 + i] = p.per_name_cap; }
    for i in 0..n { b[n_eq + 5 * n + 1 + i] = p.per_name_cap; }
    for i in 0..n { b[y_base + i] = 1.0; }

    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n { cones.push(PowerConeT(2.0 / 3.0)); }
    cones.push(ZeroConeT(n_y));

    (p_csc, q, a_csc, b, cones)
}

// =============================================================
// Variant C: cone + ReLU-aux for Q_ref subtraction
// 6n vars: [w, u, t, t_imp, y, s]
// new linear constraints (n of them):
//   s_i ≥ κ_eff_i·t_imp_i − r_i·u_i
//     ⇔  κ_eff_i·t_imp_i − r_i·u_i − s_i ≤ 0
//   s_i ≥ 0   ⇔  −s_i ≤ 0
// objective: -α'w + γw'Σw + λ_aim||w-w_aim||² + c_lin'u + Σ s
//   (the cone's t_imp does NOT appear in objective; only enforces shape via cone)
// =============================================================
fn build_variant_c(p: &Problem, nav: f64) -> (
    CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>,
    Vec<f64>, Vec<f64>,  // returns kappa_eff, r for downstream verification
) {
    let (kappa_eff, r) = compute_coefficients(p, nav);
    let n = N;
    let dim = 6 * n;

    // P (same shape as A; just bigger total dim)
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    for col in 0..dim {
        if col < n {
            for row in 0..=col {
                let s = p.sigma[row * n + col];
                let ridge = if row == col { p.ridge } else { 0.0 };
                let aim = if row == col { p.lambda_aim } else { 0.0 };
                let val = 2.0 * (p.gamma * (s + ridge) + aim);
                p_indices.push(row);
                p_data.push(val);
            }
        }
        p_indptr.push(p_indices.len());
    }
    let p_csc = CscMatrix::new(dim, dim, p_indptr, p_indices, p_data);

    // q: w-block has -α-2λ_aim·w_aim, u-block has c_lin, s-block has 1.0 (objective);
    //    t-block, t_imp-block, y-block are 0
    let mut q = vec![0.0_f64; dim];
    for i in 0..n { q[i] = -p.alpha[i] - 2.0 * p.lambda_aim * p.w_aim[i]; }
    for i in 0..n { q[n + i] = p.c_lin[i]; }
    // q[3n..4n] = 0 (t_imp NOT in objective in variant C)
    for i in 0..n { q[5 * n + i] = 1.0; }  // s in objective (coef 1; gets multiplied by NAV externally)

    // Layout of constraint rows:
    //   row 0:                    Σw = 0   (ZeroCone)
    //   rows 1..1+n:              w − u ≤ w_prev
    //   rows 1+n..1+2n:           −w − u ≤ −w_prev
    //   rows 1+2n..1+3n:          w − t ≤ 0
    //   rows 1+3n..1+4n:          −w − t ≤ 0
    //   row 1+4n:                 Σt ≤ L_max
    //   rows 2+4n..2+5n:          w_i ≤ cap
    //   rows 2+5n..2+6n:          −w_i ≤ cap
    //   rows 2+6n..2+7n:          u_i ≥ 0  → −u_i ≤ 0
    //   rows 2+7n..2+8n:          t_i ≥ 0  → −t_i ≤ 0
    //   rows 2+8n..2+9n:          [NEW] κ_eff·t_imp − r·u − s ≤ 0    (s ≥ Δ)
    //   rows 2+9n..2+10n:         [NEW] −s ≤ 0                       (s ≥ 0)
    //   rows 2+10n..2+13n:        n PowerCone slacks (t_imp, y, u)
    //   rows 2+13n..2+14n:        y_i = 1   (ZeroCone)
    //
    // n_ineq_total = (8n+1) + 2n = 10n+1
    // n_con = 1 + (10n+1) + 3n + n = 14n + 2
    let n_eq = 1;
    let n_ineq = 10 * n + 1;
    let n_pow = 3 * n;
    let n_y = n;
    let n_con = n_eq + n_ineq + n_pow + n_y;

    let pow_base = n_eq + n_ineq;
    let y_base = pow_base + n_pow;
    let s_lin_base = n_eq + 8 * n + 2; // s ≥ Δ rows base
    let s_nn_base = n_eq + 9 * n + 2;  // s ≥ 0 rows base

    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();

    for col in 0..dim {
        if col < n {
            // w_i column — same as variant A
            let i = col;
            a_indices.push(0); a_data.push(-1.0);
            a_indices.push(n_eq + i); a_data.push(1.0);
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 2 * n + i); a_data.push(1.0);
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n + 1 + i); a_data.push(1.0);
            a_indices.push(n_eq + 5 * n + 1 + i); a_data.push(-1.0);
        } else if col < 2 * n {
            // u_i column — adds: -r_i in s ≥ Δ row, and PowerCone third coord
            let i = col - n;
            a_indices.push(n_eq + i); a_data.push(-1.0);
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 6 * n + 1 + i); a_data.push(-1.0);
            a_indices.push(s_lin_base + i); a_data.push(-r[i]);  // u contributes -r·u
            a_indices.push(pow_base + 3 * i + 2); a_data.push(-1.0);
        } else if col < 3 * n {
            // t_i column
            let i = col - 2 * n;
            a_indices.push(n_eq + 2 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n); a_data.push(1.0);
            a_indices.push(n_eq + 7 * n + 1 + i); a_data.push(-1.0);
        } else if col < 4 * n {
            // t_imp_i column — adds: +κ_eff_i in s-lin row, PowerCone first coord
            let i = col - 3 * n;
            a_indices.push(s_lin_base + i); a_data.push(kappa_eff[i]);
            a_indices.push(pow_base + 3 * i); a_data.push(-1.0);
        } else if col < 5 * n {
            // y_i column
            let i = col - 4 * n;
            a_indices.push(pow_base + 3 * i + 1); a_data.push(-1.0);
            a_indices.push(y_base + i); a_data.push(1.0);
        } else {
            // s_i column — adds: -1 in s-lin row, -1 in s-NN row
            let i = col - 5 * n;
            a_indices.push(s_lin_base + i); a_data.push(-1.0);
            a_indices.push(s_nn_base + i); a_data.push(-1.0);
        }
        a_indptr.push(a_indices.len());
    }
    sort_csc_columns(&a_indptr, &mut a_indices, &mut a_data);
    let a_csc = CscMatrix::new(n_con, dim, a_indptr, a_indices, a_data);

    let mut b = vec![0.0_f64; n_con];
    for i in 0..n { b[n_eq + i] = p.w_prev[i]; }
    for i in 0..n { b[n_eq + n + i] = -p.w_prev[i]; }
    b[n_eq + 4 * n] = p.l_max;
    for i in 0..n { b[n_eq + 4 * n + 1 + i] = p.per_name_cap; }
    for i in 0..n { b[n_eq + 5 * n + 1 + i] = p.per_name_cap; }
    // s-lin and s-NN rows: b = 0
    // PowerCone rows: b = 0
    for i in 0..n { b[y_base + i] = 1.0; }

    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n { cones.push(PowerConeT(2.0 / 3.0)); }
    cones.push(ZeroConeT(n_y));

    (p_csc, q, a_csc, b, cones, kappa_eff, r)
}

fn sort_csc_columns(indptr: &[usize], indices: &mut Vec<usize>, data: &mut Vec<f64>) {
    let dim = indptr.len() - 1;
    for col in 0..dim {
        let start = indptr[col];
        let end = indptr[col + 1];
        if end > start + 1 {
            let mut pairs: Vec<(usize, f64)> = (start..end)
                .map(|k| (indices[k], data[k])).collect();
            pairs.sort_by_key(|p| p.0);
            for (offset, (r, v)) in pairs.into_iter().enumerate() {
                indices[start + offset] = r;
                data[start + offset] = v;
            }
        }
    }
}

fn solve(
    p: &CscMatrix<f64>, q: &[f64], a: &CscMatrix<f64>, b: &[f64],
    cones: &[SupportedConeT<f64>],
) -> Option<Vec<f64>> {
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .presolve_enable(false)
        .build()
        .unwrap();
    let mut solver = DefaultSolver::new(p, q, a, b, cones, settings).ok()?;
    solver.solve();
    if matches!(solver.solution.status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        Some(solver.solution.x.clone())
    } else {
        eprintln!("solver status: {:?}", solver.solution.status);
        None
    }
}

fn main() {
    println!("=== Probe 3: Q_ref residual portfolio magnitude ===\n");

    let problem = build_problem(42);

    // -----------------------------------------------------------
    // T1 — variant C unit identity: at solution, s_i · NAV matches realized
    // -----------------------------------------------------------
    println!("T1: variant C unit identity (s_i · NAV ≡ realized impact at Δw·NAV)\n");
    let nav_t1 = 1_000_000.0;
    let (pp, q, a, b, cones, kappa_eff_t1, _r_t1) = build_variant_c(&problem, nav_t1);
    let x = solve(&pp, &q, &a, &b, &cones).expect("variant C didn't converge");
    let n = N;
    let w: &[f64] = &x[..n];
    let _u: &[f64] = &x[n..2*n];
    let _t_imp: &[f64] = &x[3*n..4*n];
    let s: &[f64] = &x[5*n..6*n];

    // For each asset, compare: s_i·NAV vs compute_realized_trade_costs(Δw_i·NAV, ...).
    // Use Q_ref = Q_REF_USD on the realized side (matches what variant C is approximating).
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut n_active = 0;
    let mut n_zero = 0;
    for i in 0..n {
        let dw = w[i] - problem.w_prev[i];
        let q_dollars = dw * nav_t1;
        let cone_s_usd = s[i] * nav_t1;
        let (_spread, realized) = compute_realized_trade_costs(
            q_dollars, 0.0, problem.assets[i].kappa, problem.assets[i].sigma_d,
            problem.assets[i].adv, DELTA, Q_REF_USD,
        );
        let abs_err = (cone_s_usd - realized).abs();
        let rel_err = if realized.max(cone_s_usd).abs() > 1e-9 {
            abs_err / realized.max(cone_s_usd).abs()
        } else { 0.0 };
        if realized > 1e-9 || cone_s_usd > 1e-9 { n_active += 1; }
        else { n_zero += 1; }
        if rel_err > max_rel { max_rel = rel_err; }
        if abs_err > max_abs { max_abs = abs_err; }
    }
    println!("  active assets: {n_active}, zero-cost assets: {n_zero}");
    println!("  max abs err per asset: ${:.4}", max_abs);
    println!("  max rel err per asset: {:.2e}", max_rel);
    let t1_pass = max_rel < 1e-3;  // solver tolerance ~1e-3 acceptable for s
    println!("  T1: {}\n", if t1_pass { "PASS" } else { "FAIL" });

    // Pre-compute κ_eff for asset coverage diagnostics
    let _ = kappa_eff_t1;

    // -----------------------------------------------------------
    // T2 — A vs C divergence across NAV regimes
    // -----------------------------------------------------------
    println!("T2: variant A vs C divergence across NAV regimes\n");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>12}  {:>12}  {:>14}  {:>14}  {:>10}  {:>10}",
        "NAV", "u_thresh", "max|w-w|", "L1_diff/gross", "max|Δw_diff|",
        "real(A) ($)", "real(C) ($)", "frac<Qref_A", "frac<Qref_C"
    );

    let nav_levels = [10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0];
    let mut t2_per_nav: Vec<(f64, f64, f64, f64, f64)> = Vec::new(); // (nav, l1_diff_pct, real_diff_pct, frac_below_A, frac_below_C)
    for &nav in nav_levels.iter() {
        let u_thresh = Q_REF_USD / nav;

        // Solve variant A
        let (pa, qa, aa, ba, conesa) = build_variant_a(&problem, nav);
        let xa = match solve(&pa, &qa, &aa, &ba, &conesa) {
            Some(x) => x, None => { println!("  NAV=${:.0e}: variant A failed", nav); continue; }
        };
        let wa: Vec<f64> = xa[..n].to_vec();

        // Solve variant C
        let (pc, qc, ac, bc, conesc, _, _) = build_variant_c(&problem, nav);
        let xc = match solve(&pc, &qc, &ac, &bc, &conesc) {
            Some(x) => x, None => { println!("  NAV=${:.0e}: variant C failed", nav); continue; }
        };
        let wc: Vec<f64> = xc[..n].to_vec();

        // Weight diff
        let max_abs_diff = wa.iter().zip(wc.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let l1_diff: f64 = wa.iter().zip(wc.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        let gross_a: f64 = wa.iter().map(|w| w.abs()).sum();
        let l1_pct = if gross_a > 1e-9 { l1_diff / gross_a } else { 0.0 };

        // Δw diff (turnover, more relevant than weight)
        let max_dw_diff = (0..n).map(|i| {
            let dwa = wa[i] - problem.w_prev[i];
            let dwc = wc[i] - problem.w_prev[i];
            (dwa - dwc).abs()
        }).fold(0.0, f64::max);

        // Realized cost on each
        let mut real_a_total = 0.0_f64;
        let mut real_c_total = 0.0_f64;
        let mut n_below_qref_a = 0;
        let mut n_below_qref_c = 0;
        let mut n_traded_a = 0;
        let mut n_traded_c = 0;
        for i in 0..n {
            let dwa = wa[i] - problem.w_prev[i];
            let dwc = wc[i] - problem.w_prev[i];
            let qa = dwa * nav;
            let qc = dwc * nav;
            let (_, ra) = compute_realized_trade_costs(qa, 0.0,
                problem.assets[i].kappa, problem.assets[i].sigma_d,
                problem.assets[i].adv, DELTA, Q_REF_USD);
            let (_, rc) = compute_realized_trade_costs(qc, 0.0,
                problem.assets[i].kappa, problem.assets[i].sigma_d,
                problem.assets[i].adv, DELTA, Q_REF_USD);
            real_a_total += ra;
            real_c_total += rc;
            if qa.abs() > 1.0 {
                n_traded_a += 1;
                if qa.abs() < Q_REF_USD { n_below_qref_a += 1; }
            }
            if qc.abs() > 1.0 {
                n_traded_c += 1;
                if qc.abs() < Q_REF_USD { n_below_qref_c += 1; }
            }
        }
        let frac_below_a = if n_traded_a > 0 { n_below_qref_a as f64 / n_traded_a as f64 } else { 0.0 };
        let frac_below_c = if n_traded_c > 0 { n_below_qref_c as f64 / n_traded_c as f64 } else { 0.0 };
        let real_diff_pct = if real_c_total.abs() > 1e-9 {
            (real_a_total - real_c_total).abs() / real_c_total.abs()
        } else { 0.0 };

        println!(
            "{:>10.0}  {:>10.4}  {:>10.4}  {:>12.4}  {:>12.4}  {:>14.4}  {:>14.4}  {:>10.2}  {:>10.2}",
            nav, u_thresh, max_abs_diff, l1_pct, max_dw_diff,
            real_a_total, real_c_total, frac_below_a, frac_below_c
        );

        t2_per_nav.push((nav, l1_pct, real_diff_pct, frac_below_a, frac_below_c));
    }

    // -----------------------------------------------------------
    // T3 — drop-subtraction recommendation
    // -----------------------------------------------------------
    println!();
    println!("T3: drop-subtraction recommendation");
    println!();
    println!(
        "{:>10}  {:>14}  {:>14}  {:>14}  {:>5}",
        "NAV", "L1diff/gross", "real_diff_pct", "frac<Qref(C)", "drop?"
    );
    let mut t3_decisions = Vec::new();
    for &(nav, l1_pct, real_diff_pct, _, frac_below_c) in &t2_per_nav {
        // Drop is justified when both: weight similarity AND realized-cost similarity.
        let weight_ok = l1_pct < 0.05;
        let cost_ok = real_diff_pct < 0.10;
        let drop_ok = weight_ok && cost_ok;
        t3_decisions.push((nav, drop_ok));
        println!(
            "{:>10.0}  {:>14.4}  {:>14.4}  {:>14.2}  {:>5}",
            nav, l1_pct, real_diff_pct, frac_below_c,
            if drop_ok { "yes" } else { "NO" }
        );
    }

    // Decision at deployment NAV ($1M):
    let depl = t3_decisions.iter().find(|(n, _)| (*n - 1_000_000.0).abs() < 1.0);
    let recommendation = match depl {
        Some((_, true)) => "DROP — at deployment NAV the formulations agree closely",
        Some((_, false)) => "KEEP ReLU-aux — variant A diverges from variant C at deployment NAV",
        None => "INCONCLUSIVE — deployment NAV not in test grid",
    };
    println!();
    println!("Recommendation: {}", recommendation);

    // Final verdict
    println!();
    let probe_pass = t1_pass;
    println!("=== PROBE 3 RESULT: {} ===",
        if probe_pass { "PASS" } else { "FAIL" }
    );
    println!();
    if t1_pass {
        println!("T1 (unit identity) confirmed variant C correctly encodes the realized formula.");
        println!("T2/T3 are characterization tests (not pass/fail) — see recommendation above.");
    }
}
