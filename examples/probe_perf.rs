//! Probe 5 (production-scale + warm-start performance).
//!
//! Goal: confirm the full n=200 cone QP solves at acceptable speed and that
//! Clarabel's `update_data()` works for cone problems. Research Q2 expectation:
//! nonsymmetric-cone warm-start gives no major speedup vs cold-start (MOSEK
//! explicitly documents this, and Skajaa-Ye barriers are not self-scaled).
//!
//! ## Variable layout (5n)
//!
//! `x = [w(n), u(n), t(n), t_imp(n), y(n)]`
//!   - w_i ∈ R: portfolio weight (sum to 0)
//!   - u_i ≥ 0: |Δw_i| auxiliary (turnover)
//!   - t_i ≥ 0: |w_i| auxiliary (gross-exposure cap)
//!   - t_imp_i ≥ 0: cone slack for impact (t_imp_i ≥ u_i^1.5)
//!   - y_i = 1: dummy for PowerCone(2/3) third coordinate
//!
//! ## Constraint blocks (in order)
//!
//! Block A (existing QP shape):
//!   1 ZeroCone:   Σw = 0
//!   8n+1 NonNeg:  w − u ≤ w_prev,  −w − u ≤ −w_prev
//!                 w − t ≤ 0,       −w − t ≤ 0
//!                 Σt ≤ L_max
//!                 |w_i| ≤ cap (×2)
//!                 u ≥ 0,  t ≥ 0
//!
//! Block B (new cone term, n PowerCones in R^3):
//!   For each i: (t_imp_i, y_i, u_i) ∈ K_pow(2/3) → t_imp_i ≥ u_i^1.5
//!
//! Block C (y_i pinning):
//!   n ZeroCones:  y_i − 1 = 0
//!
//! ## Objective
//!
//! `min  -α'w + γ·w'Σw + λ_aim·||w − w_aim||² + c_lin'u + Σ κ_eff_i · t_imp_i`
//!
//! P quadratic block: w-only. q linear: w-block, u-block (c_lin), t_imp-block (κ_eff),
//! u/t/y-blocks zero.
//!
//! Run via: `cargo run --release -p rumpy-execution --example probe_perf`

use clarabel::{algebra::*, solver::*};

const N: usize = 200;
const N_FACTORS: usize = 8;
const N_ITERS: usize = 100;
const NAV: f64 = 100_000.0;
const DELTA: f64 = 0.5;

/// Deterministic xorshift32 RNG so probe is bit-reproducible.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self { Self(seed.max(1)) }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 { self.next_u32() as f64 / (u32::MAX as f64) }
    fn normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Synthetic factor-model covariance: Σ = F·F' + diag(idio).
/// F is n × k with iid normal entries scaled to give realistic vol levels.
fn synth_covariance(rng: &mut Rng) -> Vec<f64> {
    let mut f = vec![0.0_f64; N * N_FACTORS];
    for i in 0..N {
        for k in 0..N_FACTORS {
            f[i * N_FACTORS + k] = 0.02 * rng.normal();
        }
    }
    let mut sigma = vec![0.0_f64; N * N];
    // Σ = F·F'
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for k in 0..N_FACTORS {
                s += f[i * N_FACTORS + k] * f[j * N_FACTORS + k];
            }
            sigma[i * N + j] = s;
        }
    }
    // + diag(idio) — idio vol uniform ~0.02
    for i in 0..N {
        let idio = 0.0004 + 0.0001 * rng.uniform();
        sigma[i * N + i] += idio;
    }
    sigma
}

/// Synthetic per-asset alpha, ADV, kappa, sigma_d, w_prev, w_aim — production-realistic.
struct ProblemData {
    sigma: Vec<f64>,        // n*n covariance
    alpha: Vec<f64>,        // n alpha
    w_prev: Vec<f64>,       // n previous weights (sum to 0)
    w_aim: Vec<f64>,        // n aim weights (sum to 0)
    c_lin: Vec<f64>,        // n linear cost (taker + half-spread)
    kappa_eff: Vec<f64>,    // n cone weights = κ·σ·(NAV/ADV)^δ
    gamma: f64,
    lambda_aim: f64,
    l_max: f64,
    per_name_cap: f64,
    ridge: f64,
}

fn build_problem(seed: u32) -> ProblemData {
    let mut rng = Rng::new(seed);

    let sigma = synth_covariance(&mut rng);

    // alpha ~ N(0, 0.05)
    let alpha: Vec<f64> = (0..N).map(|_| 0.05 * rng.normal()).collect();

    // w_prev: small dollar-neutral portfolio with ~0.5 gross
    let mut w_prev: Vec<f64> = (0..N).map(|_| 0.005 * rng.normal()).collect();
    let mean: f64 = w_prev.iter().sum::<f64>() / N as f64;
    for w in &mut w_prev { *w -= mean; }

    // w_aim: similar magnitude, dollar-neutral
    let mut w_aim: Vec<f64> = (0..N).map(|_| 0.005 * rng.normal()).collect();
    let mean: f64 = w_aim.iter().sum::<f64>() / N as f64;
    for w in &mut w_aim { *w -= mean; }

    // c_lin: 5 bps half-spread + 3.5 bps taker = 8.5 bps fractional
    let c_lin: Vec<f64> = (0..N).map(|_| 8.5e-4).collect();

    // kappa_eff = κ·σ·(NAV/ADV)^δ. Sample (κ, σ, ADV) realistically:
    //   κ ~ U[0.2, 1.5]
    //   σ ~ U[0.02, 0.08]
    //   ADV ~ logU[$1e5, $1e9]
    let kappa_eff: Vec<f64> = (0..N).map(|_| {
        let kappa = 0.2 + 1.3 * rng.uniform();
        let sigma_d = 0.02 + 0.06 * rng.uniform();
        let log_adv = 5.0 + 4.0 * rng.uniform();
        let adv = 10.0_f64.powf(log_adv);
        kappa * sigma_d * (NAV / adv).powf(DELTA)
    }).collect();

    ProblemData {
        sigma, alpha, w_prev, w_aim, c_lin, kappa_eff,
        gamma: 4.795,
        lambda_aim: 0.054,
        l_max: 1.5,
        per_name_cap: 0.052,
        ridge: 1e-6,
    }
}

/// Build the cone-QP P, q, A, b, cones for a given ProblemData.
/// Returns (p, q, a, b, cones) ready for DefaultSolver::new.
fn build_cone_qp(data: &ProblemData) -> (
    CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>
) {
    let n = N;
    let dim = 5 * n; // [w, u, t, t_imp, y]

    // -------------------------------------------------------------
    // P: dense upper triangle on w-block; rest zero.
    // P[i,j] = 2·γ·Σ_ij + 2·γ·ridge·δ_ij + 2·λ_aim·δ_ij  (i ≤ j, i,j < n)
    // -------------------------------------------------------------
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    for col in 0..dim {
        if col < n {
            for row in 0..=col {
                let s = data.sigma[row * n + col];
                let ridge = if row == col { data.ridge } else { 0.0 };
                let aim = if row == col { data.lambda_aim } else { 0.0 };
                let val = 2.0 * (data.gamma * (s + ridge) + aim);
                p_indices.push(row);
                p_data.push(val);
            }
        }
        p_indptr.push(p_indices.len());
    }
    let p = CscMatrix::new(dim, dim, p_indptr, p_indices, p_data);

    // -------------------------------------------------------------
    // q: w-block has -α - 2λ_aim·w_aim; u-block has c_lin; t_imp-block has κ_eff.
    // t-block and y-block are zero in the linear objective term.
    // -------------------------------------------------------------
    let mut q = vec![0.0_f64; dim];
    for i in 0..n { q[i] = -data.alpha[i] - 2.0 * data.lambda_aim * data.w_aim[i]; }
    for i in 0..n { q[n + i] = data.c_lin[i]; }
    // t-block (2n..3n): zero
    for i in 0..n { q[3 * n + i] = data.kappa_eff[i]; }
    // y-block (4n..5n): zero

    // -------------------------------------------------------------
    // A and b: 4 logical row blocks
    //   Block A1 (ZeroCone(1))         : Σw = 0
    //   Block A2 (NonnegCone(8n+1))    : existing QP inequalities
    //   Block A3 (n PowerCone(2/3) in R^3) : (t_imp_i, y_i, u_i) ∈ K_pow
    //   Block A4 (ZeroCone(n))         : y_i = 1
    //
    // For Clarabel, slack s = b - A·x ∈ K. So:
    //   ZeroCone:    s = 0     → A·x = b
    //   NonnegCone:  s ≥ 0     → A·x ≤ b
    //   PowerCone:   s ∈ K_pow → A_block = -I_3, b_block = 0 places the variable
    //                            triple itself as the slack. (Just like socp_probe.)
    //
    // Layout of rows (total n_con = 1 + (8n+1) + 3n + n = 12n + 2):
    //   row 0:                 Σw = 0
    //   rows 1..1+n:           w − u ≤ w_prev          (turnover upper)
    //   rows 1+n..1+2n:        −w − u ≤ −w_prev        (turnover lower)
    //   rows 1+2n..1+3n:       w − t ≤ 0
    //   rows 1+3n..1+4n:       −w − t ≤ 0
    //   row 1+4n:              Σt ≤ L_max              (gross cap)
    //   rows 2+4n..2+5n:       w_i ≤ cap
    //   rows 2+5n..2+6n:       −w_i ≤ cap
    //   rows 2+6n..2+7n:       u_i ≥ 0  → −u_i ≤ 0
    //   rows 2+7n..2+8n:       t_i ≥ 0  → −t_i ≤ 0
    //
    //   rows 2+8n..2+11n:      n PowerCone slack triples [t_imp_i, y_i, u_i]
    //                          A·x for these rows = -t_imp_i, -y_i, -u_i
    //                          (so slack s = -A·x = (t_imp_i, y_i, u_i))
    //
    //   rows 2+11n..2+12n:     y_i = 1                 → y_i − 1 = 0
    // -------------------------------------------------------------
    let n_eq = 1;
    let n_ineq = 8 * n + 1;
    let n_pow_rows = 3 * n;
    let n_y_eq = n;
    let n_con = n_eq + n_ineq + n_pow_rows + n_y_eq;

    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();

    let pow_base = n_eq + n_ineq;        // = 8n + 2
    let y_eq_base = pow_base + n_pow_rows; // = 11n + 2

    for col in 0..dim {
        if col < n {
            // w_i column
            let i = col;
            // Σw = 0: row 0, coef +1 (A·x = 0 with ZeroCone gives Σw = 0; convention
            // matches existing solver.rs)
            // Actually existing solver.rs uses A[0]·w = -1 with b[0]=0 (ZeroCone);
            // s = b - A·x = 0 - (-Σw) = Σw → Σw = 0. So coef -1.
            a_indices.push(0); a_data.push(-1.0);
            // w − u ≤ w_prev: row n_eq + i, coef +1 on w
            a_indices.push(n_eq + i); a_data.push(1.0);
            // −w − u ≤ −w_prev: row n_eq + n + i, coef -1 on w
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            // w − t ≤ 0: row n_eq + 2n + i, coef +1 on w
            a_indices.push(n_eq + 2 * n + i); a_data.push(1.0);
            // −w − t ≤ 0: row n_eq + 3n + i, coef -1 on w
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            // w ≤ cap: row n_eq + 4n + 1 + i, coef +1 on w
            a_indices.push(n_eq + 4 * n + 1 + i); a_data.push(1.0);
            // −w ≤ cap: row n_eq + 5n + 1 + i, coef -1 on w
            a_indices.push(n_eq + 5 * n + 1 + i); a_data.push(-1.0);
        } else if col < 2 * n {
            // u_i column
            let i = col - n;
            // w − u ≤ w_prev: coef -1 on u
            a_indices.push(n_eq + i); a_data.push(-1.0);
            // −w − u ≤ −w_prev: coef -1 on u
            a_indices.push(n_eq + n + i); a_data.push(-1.0);
            // u_i ≥ 0: row n_eq + 6n + 1 + i, coef -1 on u
            a_indices.push(n_eq + 6 * n + 1 + i); a_data.push(-1.0);
            // PowerCone third coord: row pow_base + 3i + 2 (the u slot)
            a_indices.push(pow_base + 3 * i + 2); a_data.push(-1.0);
        } else if col < 3 * n {
            // t_i column
            let i = col - 2 * n;
            // w − t ≤ 0: coef -1 on t
            a_indices.push(n_eq + 2 * n + i); a_data.push(-1.0);
            // −w − t ≤ 0: coef -1 on t
            a_indices.push(n_eq + 3 * n + i); a_data.push(-1.0);
            // Σt ≤ L_max: row n_eq + 4n, coef +1 on t
            a_indices.push(n_eq + 4 * n); a_data.push(1.0);
            // t_i ≥ 0: row n_eq + 7n + 1 + i, coef -1 on t
            a_indices.push(n_eq + 7 * n + 1 + i); a_data.push(-1.0);
        } else if col < 4 * n {
            // t_imp_i column (cone first coord)
            let i = col - 3 * n;
            // PowerCone first coord: row pow_base + 3i, coef -1
            a_indices.push(pow_base + 3 * i); a_data.push(-1.0);
        } else {
            // y_i column (cone second coord)
            let i = col - 4 * n;
            // PowerCone second coord: row pow_base + 3i + 1, coef -1
            a_indices.push(pow_base + 3 * i + 1); a_data.push(-1.0);
            // y_i = 1: row y_eq_base + i, coef +1 (A·x = b → y - 1 = 0 if b=1)
            a_indices.push(y_eq_base + i); a_data.push(1.0);
        }
        a_indptr.push(a_indices.len());
    }
    // Row indices within each column must be sorted ascending for CSC.
    // Since we wrote rows in mixed order above (especially y_i column has
    // pow row + y_eq row), sort each column's slice now.
    let nnz = a_indices.len();
    // Re-walk indptr to sort per-column.
    let _ = nnz;
    for col in 0..dim {
        let start = a_indptr[col];
        let end = a_indptr[col + 1];
        if end > start + 1 {
            let mut pairs: Vec<(usize, f64)> = (start..end)
                .map(|k| (a_indices[k], a_data[k]))
                .collect();
            pairs.sort_by_key(|p| p.0);
            for (offset, (r, v)) in pairs.into_iter().enumerate() {
                a_indices[start + offset] = r;
                a_data[start + offset] = v;
            }
        }
    }

    let a = CscMatrix::new(n_con, dim, a_indptr, a_indices, a_data);

    // -------------------------------------------------------------
    // b vector
    // -------------------------------------------------------------
    let mut b = vec![0.0_f64; n_con];
    // row 0: 0 (Σw = 0 via ZeroCone)
    for i in 0..n { b[n_eq + i] = data.w_prev[i]; }
    for i in 0..n { b[n_eq + n + i] = -data.w_prev[i]; }
    // rows for w−t ≤ 0, −w−t ≤ 0, Σt ≤ L_max — most are 0
    b[n_eq + 4 * n] = data.l_max;
    for i in 0..n { b[n_eq + 4 * n + 1 + i] = data.per_name_cap; }
    for i in 0..n { b[n_eq + 5 * n + 1 + i] = data.per_name_cap; }
    // u ≥ 0, t ≥ 0: b = 0
    // PowerCone rows: b = 0 (slack = -A·x = the variable triple itself)
    // y_i = 1 rows: b = 1
    for i in 0..n { b[y_eq_base + i] = 1.0; }

    // -------------------------------------------------------------
    // Cones
    // -------------------------------------------------------------
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    cones.push(ZeroConeT(n_eq));
    cones.push(NonnegativeConeT(n_ineq));
    for _ in 0..n {
        cones.push(PowerConeT(2.0 / 3.0));
    }
    cones.push(ZeroConeT(n_y_eq));

    (p, q, a, b, cones)
}

fn main() {
    println!("=== Probe 5: production-scale n={N} cone QP performance ===\n");

    // Build the problem
    let data = build_problem(42);
    let (p, q, a, b, cones) = build_cone_qp(&data);

    // -----------------------------------------------------------
    // Sanity: solve once, check status, print key stats
    // -----------------------------------------------------------
    println!("Sanity solve:");
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .presolve_enable(false)
        .build()
        .unwrap();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
    let t0 = std::time::Instant::now();
    solver.solve();
    let first_solve_ms = t0.elapsed().as_millis();
    println!("  status: {:?}", solver.solution.status);
    println!("  iterations: {}", solver.info.iterations);
    println!("  first solve: {first_solve_ms} ms");

    // Validate: sum(w) ≈ 0, |w_i| ≤ cap, gross ≤ L_max, t_imp_i ≈ u_i^1.5
    let w: &[f64] = &solver.solution.x[..N];
    let u: &[f64] = &solver.solution.x[N..2 * N];
    let t_imp: &[f64] = &solver.solution.x[3 * N..4 * N];
    let y: &[f64] = &solver.solution.x[4 * N..5 * N];
    let sum_w: f64 = w.iter().sum();
    let max_abs_w = w.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let gross: f64 = w.iter().map(|x| x.abs()).sum();
    let cone_residual: f64 = u.iter().zip(t_imp.iter())
        .map(|(ui, ti)| (ti - ui.abs().powf(1.5)).max(0.0))
        .sum();
    let y_dev: f64 = y.iter().map(|yi| (yi - 1.0).abs()).fold(0.0, f64::max);
    println!("  sum(w) = {:.3e}", sum_w);
    println!("  max|w_i| = {:.4}  (cap = {})", max_abs_w, data.per_name_cap);
    println!("  gross Σ|w_i| = {:.4}  (L_max = {})", gross, data.l_max);
    println!("  cone residual Σmax(t_imp - u^1.5, 0) = {:.3e}", cone_residual);
    println!("  max|y_i - 1| = {:.3e}", y_dev);
    let cone_obj: f64 = data.kappa_eff.iter().zip(t_imp.iter())
        .map(|(k, t)| k * t).sum();
    println!("  cone objective contribution Σκ·t_imp = {:.6}", cone_obj);
    println!("  cone obj × NAV = ${:.2}", cone_obj * NAV);

    let sanity_pass = matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) && sum_w.abs() < 1e-4
        && max_abs_w <= data.per_name_cap + 1e-4
        && gross <= data.l_max + 1e-4
        && y_dev < 1e-4;
    println!("  sanity: {}\n", if sanity_pass { "PASS" } else { "FAIL" });

    // -----------------------------------------------------------
    // Timing: cold (build + solve from scratch) over N_ITERS
    // -----------------------------------------------------------
    println!("Cold solve timing ({N_ITERS} iters, full DefaultSolver::new + solve):");
    let mut cold_total_ms = 0u128;
    let mut cold_solved = 0;
    for _ in 0..N_ITERS {
        let t0 = std::time::Instant::now();
        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
        solver.solve();
        cold_total_ms += t0.elapsed().as_micros();
        if matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) { cold_solved += 1; }
    }
    let cold_avg_us = cold_total_ms as f64 / N_ITERS as f64;
    println!("  avg: {:.2} ms / solve  ({}/{} converged)", cold_avg_us / 1000.0, cold_solved, N_ITERS);

    // -----------------------------------------------------------
    // Timing: warm-same (cached solver, update_data with same inputs)
    // -----------------------------------------------------------
    println!();
    println!("Warm-same solve timing ({N_ITERS} iters, cached solver, update_data with same):");
    let mut solver_cached = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
    solver_cached.solve(); // prime
    let mut warm_total_us = 0u128;
    let mut warm_solved = 0;
    for _ in 0..N_ITERS {
        let t0 = std::time::Instant::now();
        let _ = solver_cached.update_q(&q);
        let _ = solver_cached.update_b(&b);
        // P doesn't change
        solver_cached.solve();
        warm_total_us += t0.elapsed().as_micros();
        if matches!(
            solver_cached.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) { warm_solved += 1; }
    }
    let warm_avg_us = warm_total_us as f64 / N_ITERS as f64;
    println!("  avg: {:.2} ms / solve  ({}/{} converged)", warm_avg_us / 1000.0, warm_solved, N_ITERS);

    // -----------------------------------------------------------
    // Timing: warm-perturbed (cached solver, perturb α + κ_eff each iter)
    // -----------------------------------------------------------
    println!();
    println!("Warm-perturbed solve timing ({N_ITERS} iters, perturb α + κ_eff bar-to-bar):");
    let mut solver_pert = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
    solver_pert.solve(); // prime
    let mut rng = Rng::new(99);
    let mut q_perturbed = q.clone();
    let mut pert_total_us = 0u128;
    let mut pert_solved = 0;
    for _ in 0..N_ITERS {
        // perturb α (q[0..N]) and κ_eff (q[3N..4N]) by ~5%
        for i in 0..N {
            q_perturbed[i] = q[i] * (1.0 + 0.05 * rng.normal());
            q_perturbed[3 * N + i] = q[3 * N + i] * (1.0 + 0.05 * rng.normal()).max(0.1);
        }
        let t0 = std::time::Instant::now();
        let _ = solver_pert.update_q(&q_perturbed);
        solver_pert.solve();
        pert_total_us += t0.elapsed().as_micros();
        if matches!(
            solver_pert.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) { pert_solved += 1; }
    }
    let pert_avg_us = pert_total_us as f64 / N_ITERS as f64;
    println!("  avg: {:.2} ms / solve  ({}/{} converged)", pert_avg_us / 1000.0, pert_solved, N_ITERS);

    // -----------------------------------------------------------
    // Pass criteria
    // -----------------------------------------------------------
    println!();
    let cold_ok = cold_avg_us / 1000.0 <= 50.0;          // cold under 50 ms
    let cold_all = cold_solved == N_ITERS;                 // 100% converge
    let warm_all = warm_solved == N_ITERS;
    let pert_all = pert_solved == N_ITERS;
    let warm_ratio = pert_avg_us as f64 / warm_avg_us.max(1.0) as f64;
    let warm_ratio_ok = warm_ratio < 2.0;                  // pert ≤ 2× warm
    let pass = sanity_pass && cold_ok && cold_all && warm_all && pert_all && warm_ratio_ok;
    println!("Pass criteria:");
    println!("  cold ≤ 50 ms        : {:.2} ms     {}", cold_avg_us / 1000.0, if cold_ok { "✓" } else { "✗" });
    println!("  cold {N_ITERS}/{N_ITERS} converged   : {cold_solved}/{N_ITERS}        {}", if cold_all { "✓" } else { "✗" });
    println!("  warm {N_ITERS}/{N_ITERS} converged   : {warm_solved}/{N_ITERS}        {}", if warm_all { "✓" } else { "✗" });
    println!("  pert {N_ITERS}/{N_ITERS} converged   : {pert_solved}/{N_ITERS}        {}", if pert_all { "✓" } else { "✗" });
    println!("  pert/warm ratio < 2× : {:.2}×   {}", warm_ratio, if warm_ratio_ok { "✓" } else { "✗" });

    // Speedup commentary (informational; research said likely no major speedup)
    let warm_speedup = cold_avg_us / warm_avg_us.max(1.0);
    let pert_speedup = cold_avg_us / pert_avg_us.max(1.0);
    println!();
    println!("Warm-start speedup vs cold:");
    println!("  warm-same      : {:.2}×", warm_speedup);
    println!("  warm-perturbed : {:.2}×", pert_speedup);
    println!("  (research Q2 expectation: nonsymmetric cones → minor speedup or none)");

    println!();
    println!("=== PROBE 5 RESULT: {} ===", if pass { "PASS" } else { "FAIL" });
}
