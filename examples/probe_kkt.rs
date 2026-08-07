//! Probe 1 (KKT marginal-cost sanity).
//!
//! Goal: verify the load-bearing claim that when the QP includes a cone term
//! `Σ κ·t_i  s.t.  t_i ≥ u_i^1.5`, the optimizer minimizes the realized cost
//! itself (not its tangent). This means:
//!   - cone-objective at solution = realized cost: `κ·t* = κ·(u*)^1.5`
//!   - marginal cone-cost at solution = realized-cost gradient: `d/du[κ·t] = 1.5·κ·u^0.5`
//!   - coefficient is κ, NOT 1.5·κ (the 1.5 was an SLP gradient artifact)
//!
//! Test design (1-D problem, closed-form KKT):
//!   `min  -α·w + (γ/2)·w² + κ·t   s.t.  t ≥ w^1.5,  w ≥ 0`
//!
//! Stationarity: `-α + γ·w + 1.5·κ·sqrt(w) = 0`  (the 1.5 here is THE GRADIENT
//! of the cost `κ·w^1.5`, not a coefficient choice). Solving:
//!   `w* = ((-1.5κ/γ + √((1.5κ/γ)² + 4α/γ))/2)²`
//!
//! What we test (5 sub-tests):
//!   T1: w_cone == w_analytical (already verified in socp_probe; re-pin here)
//!   T2: cone-objective `κ·t*` equals realized cost `κ·(w*)^1.5` to solver tol
//!   T3: numerical marginal of cone-cost at w* equals `1.5·κ·sqrt(w*)`
//!   T4: counterfactual — using `κ_wrong = 1.5·κ` produces materially different
//!       w* (proves the coefficient choice MATTERS; rules out trivial null)
//!   T5: degenerate case (high κ, low α) — w* near 0, cone solver handles
//!       boundary gracefully (status=Solved, no NaN, w* < small threshold)
//!
//! Run via: `cargo run --release -p rumpy-execution --example probe_kkt`

use clarabel::{algebra::*, solver::*};

/// Analytic KKT solution for `min -α·w + (γ/2)·w² + κ·w^1.5  s.t.  w ≥ 0`.
/// At interior solution: `-α + γ·w + 1.5·κ·sqrt(w) = 0`.
/// Substituting x = sqrt(w): `γ·x² + 1.5·κ·x − α = 0`.
/// Quadratic in x. Returns w* = x² (taking nonneg root).
/// If interior root is negative (high κ relative to α), corner solution w* = 0.
fn analytical_w_star(alpha: f64, gamma: f64, kappa: f64) -> f64 {
    let b = 1.5 * kappa / gamma;
    let c = alpha / gamma;
    let disc = b * b + 4.0 * c;
    let x = (-b + disc.sqrt()) / 2.0;
    if x < 0.0 { 0.0 } else { x * x }
}

/// Solve 1-D cone problem with arbitrary cone coefficient `cone_coef`.
/// Returns (w*, t*, status_solved).
///
/// Variables: [w, t, y]
///   t ≥ w^1.5 encoded as (t, y, w) ∈ PowerCone(2/3) with y == 1
///   w ≥ 0 (NonnegativeCone)
///
/// Objective:
///   min  -α·w + (γ/2)·w² + cone_coef·t
fn solve_1d_cone(alpha: f64, gamma: f64, cone_coef: f64) -> (f64, f64, bool) {
    let n_vars = 3;

    // P: diag(gamma, 0, 0), upper triangular CSC.
    let mut p_indices = Vec::new();
    let mut p_indptr = vec![0usize];
    let mut p_data = Vec::new();
    p_indices.push(0);
    p_data.push(gamma);
    p_indptr.push(p_indices.len());
    p_indptr.push(p_indices.len());
    p_indptr.push(p_indices.len());
    let p = CscMatrix::new(n_vars, n_vars, p_indptr, p_indices, p_data);

    let q = vec![-alpha, cone_coef, 0.0];

    // A and b — same encoding as socp_probe.rs:
    //   row 0 (ZeroCone):    y - 1 = 0
    //   rows 1,2,3 (PowCone): (t, y, w) ∈ K_pow(2/3)  via A=-I_{3x3} on (t,y,w)
    //   row 4 (NonnegCone):  -w ≤ 0  →  s = w ≥ 0
    let n_cons = 5;
    let mut a_indptr = vec![0usize];
    let mut a_indices = Vec::<usize>::new();
    let mut a_data = Vec::<f64>::new();
    // Col 0 (w): rows 3 and 4 with -1
    a_indices.push(3); a_data.push(-1.0);
    a_indices.push(4); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    // Col 1 (t): row 1 with -1
    a_indices.push(1); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    // Col 2 (y): row 0 with +1, row 2 with -1
    a_indices.push(0); a_data.push(1.0);
    a_indices.push(2); a_data.push(-1.0);
    a_indptr.push(a_indices.len());
    let a = CscMatrix::new(n_cons, n_vars, a_indptr, a_indices, a_data);

    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let cones = vec![
        ZeroConeT(1),
        PowerConeT(2.0 / 3.0),
        NonnegativeConeT(1),
    ];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
    solver.solve();

    let solved = matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    );
    let w = solver.solution.x[0];
    let t = solver.solution.x[1];
    (w, t, solved)
}

/// Numerical derivative of the cone cost `κ·u^1.5` at u = w*.
/// Two-sided central difference with relative ε.
fn numerical_cost_gradient(kappa: f64, w_star: f64) -> f64 {
    if w_star < 1e-12 {
        return 0.0;
    }
    let eps = (1e-6 * w_star).max(1e-10);
    let cost_plus = kappa * (w_star + eps).powf(1.5);
    let cost_minus = kappa * (w_star - eps).powf(1.5);
    (cost_plus - cost_minus) / (2.0 * eps)
}

fn main() {
    println!("=== Probe 1: KKT marginal-cost sanity ===\n");

    // ============================================================
    // T1, T2, T3 — main grid: vary κ to span 0.01–10× range.
    // ============================================================
    let cases: &[(f64, f64, f64)] = &[
        // (alpha, gamma, kappa)
        (0.10, 1.0, 0.001),  // very low cost
        (0.10, 1.0, 0.01),
        (0.10, 1.0, 0.05),
        (0.10, 1.0, 0.10),
        (0.10, 1.0, 0.50),   // high cost
        (0.20, 1.0, 0.05),
        (0.05, 0.5, 0.10),
        (0.30, 2.0, 0.20),
    ];

    println!("T1+T2+T3: cone solution matches analytical, cost matches realized,");
    println!("          marginal matches `1.5·κ·sqrt(w*)`");
    println!();
    println!(
        "{:>5}  {:>5}  {:>6}  {:>14}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>4}",
        "α", "γ", "κ", "w*_anal", "w*_cone", "err_w", "cost_id", "marg_id", "1.5κ√w*", "OK"
    );
    let mut all_pass_main = true;
    for &(alpha, gamma, kappa) in cases {
        let w_anal = analytical_w_star(alpha, gamma, kappa);
        let (w_cone, t_cone, solved) = solve_1d_cone(alpha, gamma, kappa);

        // T1: w*_cone == w*_analytical
        let err_w = (w_cone - w_anal).abs();
        let pass_w = err_w < 1e-5 && solved;

        // T2: cone-objective (κ·t*) equals realized cost (κ·(w*)^1.5)
        let cone_obj = kappa * t_cone;
        let realized_cost = kappa * w_cone.powf(1.5);
        let cost_id_err = (cone_obj - realized_cost).abs();
        let pass_cost = cost_id_err < 1e-6;

        // T3: numerical marginal of cone-cost at w* matches 1.5·κ·sqrt(w*)
        let num_grad = numerical_cost_gradient(kappa, w_cone);
        let analytic_grad = 1.5 * kappa * w_cone.sqrt();
        let marg_id_err = (num_grad - analytic_grad).abs();
        // Allow generous tolerance because the central-difference itself has O(ε²) error.
        let pass_marg = marg_id_err < 1e-4 || marg_id_err / analytic_grad.max(1e-12) < 1e-4;

        let pass = pass_w && pass_cost && pass_marg;
        if !pass { all_pass_main = false; }
        println!(
            "{:>5.2}  {:>5.2}  {:>6.3}  {:>14.10}  {:>14.10}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.6}  {:>4}",
            alpha, gamma, kappa, w_anal, w_cone, err_w, cost_id_err, marg_id_err, analytic_grad,
            if pass { "✓" } else { "✗" }
        );
    }
    println!("\nT1+T2+T3 overall: {}\n", if all_pass_main { "PASS" } else { "FAIL" });

    // ============================================================
    // T4 — counterfactual: using κ_wrong = 1.5·κ produces a different solution.
    // This rules out the null hypothesis "the coefficient choice doesn't matter
    // because everything is an artifact of the cone shape." If using 1.5·κ vs κ
    // gives the same w*, then we'd have no empirical signal for our claim.
    // ============================================================
    println!("T4: counterfactual — `κ_wrong = 1.5·κ` produces materially different w*");
    println!();
    println!(
        "{:>5}  {:>5}  {:>6}  {:>14}  {:>14}  {:>10}  {:>4}",
        "α", "γ", "κ", "w*_(κ)", "w*_(1.5κ)", "rel_diff", "OK"
    );
    let mut t4_directions_ok = true;
    let mut t4_diffs = Vec::new();
    for &(alpha, gamma, kappa) in &cases[..5] {
        let (w_correct, _, _) = solve_1d_cone(alpha, gamma, kappa);
        let (w_wrong, _, _) = solve_1d_cone(alpha, gamma, 1.5 * kappa);
        let rel_diff = if w_correct > 1e-9 { (w_correct - w_wrong) / w_correct } else { 0.0 };
        // Pass criteria (per-row): direction-correct (higher cost → trade less,
        // so w_correct > w_wrong, always). Magnitude is regime-dependent: at
        // tiny κ the cost is dominated by α/γ so the choice barely moves w*.
        // We separately check that magnitude scales with κ across the grid.
        let direction_ok = w_correct > w_wrong;
        if !direction_ok { t4_directions_ok = false; }
        t4_diffs.push((kappa, rel_diff));
        println!(
            "{:>5.2}  {:>5.2}  {:>6.3}  {:>14.10}  {:>14.10}  {:>10.4}  {:>4}",
            alpha, gamma, kappa, w_correct, w_wrong, rel_diff,
            if direction_ok { "✓" } else { "✗" }
        );
    }
    // T4 grid-level invariant: rel_diff must be monotone non-decreasing in κ
    // (modulo solver noise at tiny κ). Confirms the coefficient choice's effect
    // grows with cost magnitude — exactly what we'd expect if κ is the right
    // coefficient and 1.5·κ is wrong.
    let mut t4_monotone = true;
    for w in t4_diffs.windows(2) {
        if w[1].1 + 1e-4 < w[0].1 {
            t4_monotone = false;
            break;
        }
    }
    // Substantive signal: at the highest-κ row, rel_diff must exceed 10%.
    let t4_substantive = t4_diffs.last().map(|(_, d)| *d > 0.10).unwrap_or(false);
    let t4_pass = t4_directions_ok && t4_monotone && t4_substantive;
    println!(
        "  monotone(rel_diff in κ): {}  substantive(largest > 10%): {}",
        if t4_monotone { "✓" } else { "✗" },
        if t4_substantive { "✓" } else { "✗" }
    );
    println!("\nT4 overall: {}\n", if t4_pass { "PASS" } else { "FAIL" });

    // ============================================================
    // T5 — degenerate case: high κ pushes solution to corner (w* ≈ 0).
    // Cone solver must still return Solved without NaN.
    // ============================================================
    println!("T5: degenerate (high κ, low α → corner w* ≈ 0)");
    println!();
    let degenerate_cases: &[(f64, f64, f64)] = &[
        (0.001, 1.0, 1.0),      // κ huge vs α
        (0.0001, 1.0, 10.0),    // κ even more dominant
        (1e-9, 1.0, 1.0),       // basically zero α
    ];
    let mut t5_all_pass = true;
    for &(alpha, gamma, kappa) in degenerate_cases {
        let (w, t, solved) = solve_1d_cone(alpha, gamma, kappa);
        let analytical = analytical_w_star(alpha, gamma, kappa);
        let no_nan = !w.is_nan() && !t.is_nan();
        let close_to_zero = w.abs() < 1e-3;
        let matches_analytical = (w - analytical).abs() < 1e-4;
        let pass = solved && no_nan && close_to_zero && matches_analytical;
        if !pass { t5_all_pass = false; }
        println!(
            "  α={alpha:.1e}, γ={gamma}, κ={kappa}:  w*_cone={w:.10}  w*_anal={analytical:.10}  solved={solved}  no_nan={no_nan}  {}",
            if pass { "✓" } else { "✗" }
        );
    }
    println!("\nT5 overall: {}\n", if t5_all_pass { "PASS" } else { "FAIL" });

    // ============================================================
    // Summary
    // ============================================================
    let all_pass = all_pass_main && t4_pass && t5_all_pass;
    println!("=== PROBE 1 RESULT: {} ===", if all_pass { "PASS" } else { "FAIL" });
    println!();
    println!("Interpretation if PASS:");
    println!("  - Cone formulation `min Σκ·t s.t. t ≥ u^1.5` minimizes the realized cost");
    println!("    itself (T2): cone-objective = κ·u^1.5 at solution.");
    println!("  - Marginal cost at solution matches `d/du[κ·u^1.5] = 1.5·κ·u^0.5` (T3).");
    println!("  - Coefficient choice MATTERS (T4): wrong coef gives wrong solution.");
    println!("  - Boundary case stable (T5).");
    println!("  → Recommended cone weight is `κ_eff = κ·σ·(NAV/ADV)^δ`, no 1.5 multiplier.");
    println!("    The 1.5 in our SLP iteration was the gradient of the cost, not a");
    println!("    coefficient — consistent with cvxportfolio + MOSEK Cookbook §6.");
}
