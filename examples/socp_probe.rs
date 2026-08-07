//! SOCP feasibility probe — Clarabel PowerCone determinism + speed + accuracy.
//!
//! Tests three properties:
//!   1. CORRECTNESS: PowerCone(2/3) correctly encodes t ≥ u^1.5.
//!      Reference: 1-D problem with closed-form KKT.
//!   2. SPEED: solve time vs the standard QP.
//!   3. DETERMINISM: bit-exact f64 output for cross-platform comparison.
//!      Run on mac and on desktop; outputs must match bit-exactly.
//!
//! Run via `cargo run --release -p rumpy-execution --example socp_probe`.

use clarabel::{algebra::*, solver::*};

/// 1-D problem: minimize  -alpha·w + (gamma/2)·w² + kappa·w^1.5  subject to  w ≥ 0
///
/// Closed-form KKT: -alpha + gamma·w + 1.5·kappa·sqrt(w) = 0
/// Let x = sqrt(w), so:  x² + 1.5·(kappa/gamma)·x − alpha/gamma = 0
/// → x = [-1.5·(kappa/gamma) + sqrt((1.5·kappa/gamma)² + 4·alpha/gamma)] / 2
/// → w* = x²
fn analytical_1d(alpha: f64, gamma: f64, kappa: f64) -> f64 {
    let b = 1.5 * kappa / gamma;
    let c = alpha / gamma;
    let disc = b * b + 4.0 * c;
    let x = (-b + disc.sqrt()) / 2.0;
    x * x
}

/// Solve the 1-D problem via Clarabel with a PowerCone(2/3) constraint.
///
/// Variables: [w, t, y] (3 variables)
///   t ≥ w^1.5 encoded as (t, y, w) ∈ PowerCone(2/3) with y == 1
///   w ≥ 0 (NonnegativeCone)
///
/// Objective:
///   min  -alpha·w + (gamma/2)·w² + kappa·t
///   = (1/2) [w t y]·diag(gamma, 0, 0)·[w t y]' + [-alpha kappa 0]·[w t y]'
fn solve_1d_via_socp(alpha: f64, gamma: f64, kappa: f64) -> (f64, f64, u128) {
    let n_vars = 3; // [w, t, y]

    // P: diag(gamma, 0, 0), upper triangular.
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    // Column 0: P[0,0] = gamma
    p_indices.push(0);
    p_data.push(gamma);
    p_indptr.push(p_indices.len());
    // Column 1, 2: empty
    p_indptr.push(p_indices.len());
    p_indptr.push(p_indices.len());
    let p = CscMatrix::new(n_vars, n_vars, p_indptr, p_indices, p_data);

    // q: [-alpha, kappa, 0]
    let q = vec![-alpha, kappa, 0.0];

    // Constraints:
    //   y == 1                                (ZeroCone, dim 1)
    //   (t, y, w) ∈ PowerCone(2/3)            (PowerCone, dim 3)
    //   w ≥ 0                                 (NonnegCone, dim 1)
    //
    // Layout: A·x + s = b, s ∈ cone
    //
    // Row 0 (ZeroCone):       y - 1 = 0           A[0] = [0, 0, 1], b[0] = 1
    // Rows 1,2,3 (PowerCone): -t = -t_slack       A[1] = [0, -1, 0], b[1] = 0
    //                         -y = -y_slack       A[2] = [0, 0, -1], b[2] = 0
    //                         -w = -w_slack       A[3] = [-1, 0, 0], b[3] = 0
    // Row 4 (NonnegCone):     -w ≤ 0              A[4] = [-1, 0, 0], b[4] = 0
    //
    // Wait — for NonnegCone {s ≥ 0}, b - Ax = s ≥ 0. So A = [1,0,0], b = 0 means
    //   s = b - Ax = -w. Need s ≥ 0 → -w ≥ 0 → w ≤ 0. WRONG.
    // For w ≥ 0:  A = [-1,0,0], b = 0, then s = b - Ax = w. s ≥ 0 → w ≥ 0. ✓

    // CSC for A. Easier to build column-major, n_vars columns.
    // Column = variable, Row = constraint.
    //
    // var w (col 0): row 3 (-1), row 4 (-1)
    // var t (col 1): row 1 (-1)
    // var y (col 2): row 0 (1), row 2 (-1)

    let n_cons = 5;
    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();
    // Col 0 (w):
    a_indices.push(3); a_data.push(-1.0);
    a_indices.push(4); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    // Col 1 (t):
    a_indices.push(1); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    // Col 2 (y):
    a_indices.push(0); a_data.push(1.0);
    a_indices.push(2); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    let a = CscMatrix::new(n_cons, n_vars, a_indptr, a_indices, a_data);

    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let cones = vec![
        ZeroConeT(1),       // y == 1
        PowerConeT(2.0 / 3.0),  // (t, y, w) ∈ K_pow(2/3) → t^(2/3) · y^(1/3) ≥ w
        NonnegativeConeT(1), // w ≥ 0
    ];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .unwrap();

    let t0 = std::time::Instant::now();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
    solver.solve();
    let elapsed_ns = t0.elapsed().as_nanos();

    let w = solver.solution.x[0];
    let t = solver.solution.x[1];
    (w, t, elapsed_ns)
}

/// Time the standard QP at similar size (no power cone, just a 5-var dense QP).
/// Uses our existing CachedSolver via a tiny n=5 problem.
fn time_standard_qp() -> u128 {
    use clarabel::{algebra::*, solver::*};
    let n = 5;
    let dim = 3 * n; // mirroring rumpy's QP layout (w + u + t auxiliary)

    // P: 2·γ·I on the w block (5×5) + 2·λ_aim·I (mock = 1)
    let gamma = 4.0;
    let lambda_aim = 1.0;
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    for col in 0..dim {
        if col < n {
            // diag = 2·γ + 2·λ_aim
            p_indices.push(col);
            p_data.push(2.0 * gamma + 2.0 * lambda_aim);
        }
        p_indptr.push(p_indices.len());
    }
    let p = CscMatrix::new(dim, dim, p_indptr, p_indices, p_data);

    let mut q = vec![0.0; dim];
    for i in 0..n {
        q[i] = -0.05;
    } // alpha
    for i in n..2 * n {
        q[i] = 0.001;
    } // c_lin

    // Trivial constraints: just dollar-neutral and box bounds.
    // Layout per rumpy: 1 eq + 8n + 1 ineq = total 8n + 2
    let n_eq = 1;
    let n_ineq = 8 * n + 1;
    let n_con = n_eq + n_ineq;

    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();
    let cap = 0.2;
    for col in 0..dim {
        if col < n {
            let i = col;
            a_indices.push(0);
            a_data.push(-1.0);
            a_indices.push(n_eq + i);
            a_data.push(1.0);
            a_indices.push(n_eq + n + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + 2 * n + i);
            a_data.push(1.0);
            a_indices.push(n_eq + 3 * n + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n + 1 + i);
            a_data.push(1.0);
            a_indices.push(n_eq + 5 * n + 1 + i);
            a_data.push(-1.0);
        } else if col < 2 * n {
            let i = col - n;
            a_indices.push(n_eq + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + n + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + 6 * n + 1 + i);
            a_data.push(-1.0);
        } else {
            let i = col - 2 * n;
            a_indices.push(n_eq + 2 * n + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + 3 * n + i);
            a_data.push(-1.0);
            a_indices.push(n_eq + 4 * n);
            a_data.push(1.0);
            a_indices.push(n_eq + 7 * n + 1 + i);
            a_data.push(-1.0);
        }
        a_indptr.push(a_indices.len());
    }
    let a = CscMatrix::new(n_con, dim, a_indptr, a_indices, a_data);

    let mut b = vec![0.0; n_con];
    let l_max = 1.5;
    b[n_eq + 4 * n] = l_max;
    for i in 0..n {
        b[n_eq + 4 * n + 1 + i] = cap;
    }
    for i in 0..n {
        b[n_eq + 5 * n + 1 + i] = cap;
    }

    let cones = vec![ZeroConeT(n_eq), NonnegativeConeT(n_ineq)];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .presolve_enable(false)
        .build()
        .unwrap();

    let t0 = std::time::Instant::now();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
    solver.solve();
    t0.elapsed().as_nanos()
}

fn main() {
    println!("=== SOCP feasibility probe ===");
    println!();

    // ===== TEST 1: correctness vs analytical =====
    println!("Test 1: correctness (1-D problem with PowerCone(2/3))");
    let test_cases = [
        (0.10, 1.0, 0.05),
        (0.20, 1.0, 0.05),
        (0.05, 0.5, 0.10),
        (0.30, 2.0, 0.20),
    ];
    for (alpha, gamma, kappa) in test_cases {
        let analytical = analytical_1d(alpha, gamma, kappa);
        let (w_socp, t_socp, _) = solve_1d_via_socp(alpha, gamma, kappa);
        let t_expected = analytical.powf(1.5);
        let w_err = (w_socp - analytical).abs();
        let t_err = (t_socp - t_expected).abs();
        let pass = w_err < 1e-6 && t_err < 1e-6;
        println!(
            "  α={alpha}, γ={gamma}, κ={kappa}:  w*={analytical:.10} w_socp={w_socp:.10}  err_w={w_err:.2e}  err_t={t_err:.2e}  {}",
            if pass { "✓" } else { "✗" }
        );
    }
    println!();

    // ===== TEST 2: bit-exact output =====
    println!("Test 2: bit-exact f64 output for cross-platform comparison");
    let (w, t, _) = solve_1d_via_socp(0.10, 1.0, 0.05);
    println!("  w = 0x{:016x} ({})", w.to_bits(), w);
    println!("  t = 0x{:016x} ({})", t.to_bits(), t);
    let (w2, t2, _) = solve_1d_via_socp(0.20, 1.0, 0.05);
    println!("  w_case2 = 0x{:016x} ({})", w2.to_bits(), w2);
    println!("  t_case2 = 0x{:016x} ({})", t2.to_bits(), t2);
    println!();

    // ===== TEST 3: speed =====
    println!("Test 3: speed (avg over 100 solves)");
    const N: usize = 100;

    // SOCP
    let mut total_socp_ns = 0u128;
    for _ in 0..N {
        let (_, _, ns) = solve_1d_via_socp(0.10, 1.0, 0.05);
        total_socp_ns += ns;
    }
    println!("  SOCP (3-var, 1 PowerCone):  avg {:.2} µs / solve", total_socp_ns as f64 / N as f64 / 1000.0);

    // Standard QP at n=5 mirroring rumpy structure
    let mut total_qp_ns = 0u128;
    for _ in 0..N {
        total_qp_ns += time_standard_qp();
    }
    println!("  Standard QP (n=5 portfolio): avg {:.2} µs / solve", total_qp_ns as f64 / N as f64 / 1000.0);
    println!();

    // ===== TEST 4: scaling =====
    println!("Test 4: scaling — solve N portfolios with PowerCone, simulate full-portfolio cost");
    // Solve n_portfolios independent 1-D PowerCone problems sequentially
    // (proxy for one "n-asset" portfolio with power cone per asset).
    for n_assets in [10, 50, 150] {
        let t0 = std::time::Instant::now();
        for _ in 0..n_assets {
            solve_1d_via_socp(0.10, 1.0, 0.05);
        }
        let elapsed = t0.elapsed().as_nanos();
        println!(
            "  n_assets={n_assets}: total {:.2} ms (avg {:.2} µs / asset-cone)",
            elapsed as f64 / 1e6,
            elapsed as f64 / n_assets as f64 / 1000.0
        );
    }
}
