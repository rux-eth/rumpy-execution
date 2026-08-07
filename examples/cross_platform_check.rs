//! Cross-platform FP-determinism diagnostic.
//!
//! Run on Mac, run on Desktop, diff stdout. Each section reports byte-level
//! representations so the diff catches sub-ULP differences that printf would hide.
//!
//! Build:   cargo build --release --example cross_platform_check -p rumpy-execution
//! Run:     ./target/release/examples/cross_platform_check > mac.txt
//! Compare: diff mac.txt desktop.txt

use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};

fn bytes(x: f64) -> u64 {
    x.to_bits()
}

fn print_f(label: &str, x: f64) {
    println!("  {label:<40} = {x:.20e}  bits=0x{:016x}", bytes(x));
}

fn ulp_gap(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { b - a }
}

// =============================================================================
// 1. libm transcendentals
// =============================================================================
fn test_libm() {
    println!("=== 1. libm transcendentals ===");
    print_f("(-ln(2)/60).exp()  [EWMA seed h=60d]", (-2.0_f64.ln() / 60.0).exp());
    print_f("(-ln(2)/30).exp()  [EWMA seed h=30d]", (-2.0_f64.ln() / 30.0).exp());
    print_f("(-ln(2)/24).exp()  [EWMA seed funding]", (-2.0_f64.ln() / 24.0).exp());
    print_f("0.123.sqrt()  [IEEE-mandated, must match]", (0.123_f64).sqrt());
    print_f("1.5.ln()", (1.5_f64).ln());
    print_f("0.7.cos()", (0.7_f64).cos());
    println!("  GP13 phi_h = exp(-1/h):");
    for &h in &[3usize, 6, 12, 24, 50] {
        print_f(&format!("    phi_{h}"), (-1.0_f64 / h as f64).exp());
    }
    println!();
}

// =============================================================================
// 2. HashMap iteration order (within-machine non-determinism)
// =============================================================================
fn test_hashmap_order() {
    println!("=== 2. HashMap iteration order (3 builds, same machine) ===");
    println!("    If hashes differ across the 3 runs => RandomState non-determinism live");
    for run in 1..=3 {
        let mut m: HashMap<(i64, String), f64> = HashMap::new();
        for i in 0..100 {
            // Same insertion order each run; HashMap perturbs via RandomState seeded at startup
            m.insert((i as i64, format!("SYM{i}")), i as f64 * 0.01);
        }
        // Also make a BTreeMap for comparison
        let bt: BTreeMap<(i64, String), f64> = m.iter().map(|(k, v)| (k.clone(), *v)).collect();

        let mut h_hashmap = DefaultHasher::new();
        for (k, _) in &m { k.hash(&mut h_hashmap); }
        let mut h_btreemap = DefaultHasher::new();
        for (k, _) in &bt { k.hash(&mut h_btreemap); }

        // Sum the values in iteration order — this is what GP13 blend does.
        let mut sum_hashmap = 0.0_f64;
        for (_, v) in &m { sum_hashmap += v; }
        let mut sum_btreemap = 0.0_f64;
        for (_, v) in &bt { sum_btreemap += v; }

        println!("  run {run}: hashmap_iter_order_hash=0x{:016x}  btreemap_iter_order_hash=0x{:016x}",
                 h_hashmap.finish(), h_btreemap.finish());
        println!("           hashmap_sum_bits=0x{:016x}  btreemap_sum_bits=0x{:016x}",
                 bytes(sum_hashmap), bytes(sum_btreemap));
    }
    println!();
}

// =============================================================================
// 3. Naive vs Welford variance (catastrophic cancellation)
// =============================================================================
fn test_variance() {
    println!("=== 3. Naive vs Welford variance ===");
    // Cancellation-prone sequence: large mean offset + small noise
    // mean ~= 1e8, std ~= 0.01.  Naive E[x^2] - E[x]^2 cancels disastrously.
    let n = 1000;
    let xs: Vec<f64> = (0..n)
        .map(|i| 1.0e8 + ((i as f64) * 0.0001).sin())
        .collect();

    // --- Naive ---
    let mean_naive: f64 = xs.iter().sum::<f64>() / n as f64;
    let mean_sq: f64 = xs.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let var_naive = (mean_sq - mean_naive * mean_naive).max(0.0);

    // --- Welford ---
    let mut m = 0.0_f64;
    let mut s = 0.0_f64;
    let mut count = 0usize;
    for &x in &xs {
        count += 1;
        let delta = x - m;
        m += delta / count as f64;
        let delta2 = x - m;
        s += delta * delta2;
    }
    let var_welford = s / n as f64;

    // True variance of (i*0.0001).sin() over 0..1000: ~0.5 * (1 - sinc(...))
    // Reference computed offline.
    print_f("mean (both algorithms)", mean_naive);
    print_f("E[x^2]", mean_sq);
    print_f("E[x]^2", mean_naive * mean_naive);
    print_f("E[x^2] - E[x]^2 (cancellation)", mean_sq - mean_naive * mean_naive);
    print_f("var_naive", var_naive);
    print_f("var_welford", var_welford);
    println!("  cancellation ULP gap from welford: {} ULPs",
             ulp_gap(bytes(var_naive), bytes(var_welford)));
    println!();
}

// =============================================================================
// 4. Sum order sensitivity
// =============================================================================
fn test_sum_order() {
    println!("=== 4. Summation order ===");
    // Mix of magnitudes — typical for a GP13 alpha blend across heterogeneous assets
    let xs: Vec<f64> = (0..500)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * (1.0 + (i as f64 * 0.137).sin()) * 1e-3 + 1e9 * if i == 250 { 1.0 } else { 0.0 }
        })
        .collect();

    let s_fwd: f64 = xs.iter().sum();
    let s_rev: f64 = xs.iter().rev().sum();
    let s_pair: f64 = pairwise_sum(&xs);
    let s_kahan: f64 = kahan_sum(&xs);

    print_f("forward sum", s_fwd);
    print_f("reverse sum", s_rev);
    print_f("pairwise sum", s_pair);
    print_f("kahan sum (ground truth)", s_kahan);
    println!("  fwd vs rev:   {} ULPs", ulp_gap(bytes(s_fwd), bytes(s_rev)));
    println!("  fwd vs kahan: {} ULPs", ulp_gap(bytes(s_fwd), bytes(s_kahan)));
    println!();
}

fn pairwise_sum(xs: &[f64]) -> f64 {
    if xs.len() <= 8 {
        return xs.iter().sum();
    }
    let mid = xs.len() / 2;
    pairwise_sum(&xs[..mid]) + pairwise_sum(&xs[mid..])
}

fn kahan_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &x in xs {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// =============================================================================
// 5. FMA fusion (a*b + c)
// =============================================================================
fn test_fma() {
    println!("=== 5. FMA fusion ===");
    // Cases where fused vs separate differ measurably
    let cases = [
        (0.1_f64, 0.2_f64, 0.3_f64),
        (1.234567e-8, 9.876543e-9, -1.218874e-16), // near-cancellation
        ((-2.0_f64.ln() / 60.0_f64).exp(), 0.00012345, 0.99987655),
        (1.0e15, 1.0e-15, 1.0_f64),
    ];
    for (i, (a, b, c)) in cases.iter().enumerate() {
        let fused = a.mul_add(*b, *c);
        let separate = a * b + c;
        println!("  case {i}: a={a:.6e} b={b:.6e} c={c:.6e}");
        print_f("    fused (mul_add)", fused);
        print_f("    separate (a*b + c)", separate);
        println!("    delta: {} ULPs", ulp_gap(bytes(fused), bytes(separate)));
    }
    println!();
}

// =============================================================================
// 6. EWMA dispersion seed (mimics dynamic gamma init)
// =============================================================================
fn test_ewma_seed() {
    println!("=== 6. EWMA seed used in dynamic gamma ===");
    // disp_halflife_days values from v5 sweep champions
    for hl in [7, 30, 60, 87, 120] {
        let alpha = 1.0 - (-2.0_f64.ln() / hl as f64).exp();
        print_f(&format!("alpha (hl={hl})"), alpha);
    }
    println!();
}

// =============================================================================
// 7. Summary header
// =============================================================================
fn print_header() {
    println!("Cross-Platform FP-Determinism Diagnostic");
    println!("OS:        {}", std::env::consts::OS);
    println!("ARCH:      {}", std::env::consts::ARCH);
    println!("FAMILY:    {}", std::env::consts::FAMILY);
    println!("rustc:     compiled-time only — see binary metadata");
    println!();
}

// =============================================================================
// 7. Clarabel solve — does the QP solver itself diverge cross-platform?
// =============================================================================
//
// Tests S1 from docs/plans/cross-platform-divergence.md. Constructs a small
// synthetic QP using ONLY hardcoded float literals (no libm in input
// generation) and solves once. If output weights differ bit-by-bit between
// machines, Clarabel's BLAS/LAPACK dependency is the residual divergence
// source. If they match, S1 is wrong and we look elsewhere.
fn test_clarabel_solve() {
    use rumpy_execution::config::QPConfig;
    use rumpy_execution::solver::CachedOSQP;

    println!("=== 7. Clarabel solve (S1 test) ===");

    // Realistic-ish n=20 problem matching the V5 #651 champion's QP shape.
    // All inputs are hardcoded literals — no libm, no RNG, no HashMap, no FMA
    // ambiguity in input construction. Any divergence here is solver-internal.
    let n = 20;

    // Symmetric positive-definite covariance.
    // Diagonal dominates (asset-specific vol ~10-30%/sqrt(252) per day)
    // Off-diagonal coupling small but nonzero (mild correlation).
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        // diagonal: 0.0004..0.0042 (vol 2-6.5% daily)
        cov[i * n + i] = 0.0004 + (i as f64) * 0.0002;
    }
    // off-diagonals: deterministic small values
    for i in 0..n {
        for j in (i + 1)..n {
            // Use only int arithmetic + division to avoid sin/cos
            let v = ((((i as i64) * 7 + (j as i64) * 13) % 17) as f64 - 8.0) * 1e-5;
            cov[i * n + j] = v;
            cov[j * n + i] = v;
        }
    }

    // Alpha: alternating sign, small magnitude (dollar-alpha scale ~bps).
    let alpha: Vec<f64> = (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * (0.0001 + (i as f64) * 0.00005)
        })
        .collect();

    // Prior weights — small positive holdings.
    let w_prev: Vec<f64> = (0..n)
        .map(|i| ((i as i64 - (n as i64) / 2) as f64) * 0.005)
        .collect();

    // Aim weights — slightly different from w_prev.
    let w_aim: Vec<f64> = (0..n)
        .map(|i| ((i as i64 - (n as i64) / 2) as f64) * 0.004)
        .collect();

    // Linear cost (fee + half-spread, in fractional units).
    let c_lin = vec![0.00045_f64; n];

    // SOCP cone weights (added 2026-05-05). κ_eff drives the impact term;
    // r is the Q_ref-residual coefficient. For the determinism probe we use
    // simple constant values across assets — the goal is to verify identical
    // bit-level output across mac/linux, not to test realistic distributions.
    let kappa_eff = vec![0.001_f64; n];
    let r_cone = vec![0.0_f64; n]; // no Q_ref subtraction in this probe

    // Verify symmetry of cov for paranoia
    for i in 0..n {
        for j in 0..n {
            assert_eq!(cov[i * n + j].to_bits(), cov[j * n + i].to_bits());
        }
    }

    // QP config — V5 #651 champion's tunables.
    let qp_cfg = QPConfig {
        gamma: 15.426384203801595,
        lambda_aim: 2.575255830295259,
        l_max: 1.5,
        per_name_cap: Some(0.03985912930880578),
        ridge: 1e-6,
        prune_top_k: None,
        sigma_target_daily: None,
        per_name_cap_vol_scale: false,
        topk_cap: None,
        per_name_cap_floor_safety_mult: 1.5,
        per_name_cap_vol_scale_ceiling_mult: 5.0,
        per_name_cap_default: 0.05,
    };

    // Hash inputs for sanity (so the diff confirms inputs match cross-platform)
    let mut input_hash: u64 = 0;
    for &v in &cov {
        input_hash = input_hash.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
    }
    for &v in &alpha {
        input_hash = input_hash.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
    }
    for &v in &w_prev {
        input_hash = input_hash.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
    }
    for &v in &w_aim {
        input_hash = input_hash.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
    }
    println!("  input_hash (FNV-style of cov+alpha+w_prev+w_aim bits) = 0x{:016x}", input_hash);

    let mut solver = CachedOSQP::new(n, &qp_cfg, &cov).expect("solver setup");

    // First solve (cold) — full setup
    let result1 = solver.solve(&alpha, &cov, &w_prev, &w_aim, &c_lin, &kappa_eff, &r_cone, None);
    match result1 {
        Some((w, status)) => {
            println!("  cold solve status = {status}");
            for i in 0..n {
                println!("    w[{i:>2}] = {:.20e}  bits=0x{:016x}", w[i], w[i].to_bits());
            }
            // Aggregate hash so a one-line diff catches drift even if printf rounding differs
            let mut wh: u64 = 0;
            for &v in &w {
                wh = wh.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
            }
            println!("  cold output_hash = 0x{:016x}", wh);
        }
        None => {
            println!("  cold solve FAILED");
        }
    }

    // Second solve (warm) — exercises the update_P / update_q / update_b path
    // that the per-bar simulation actually hits most of the time.
    let result2 = solver.solve(&alpha, &cov, &w_prev, &w_aim, &c_lin, &kappa_eff, &r_cone, None);
    match result2 {
        Some((w, status)) => {
            println!("  warm solve status = {status}");
            let mut wh: u64 = 0;
            for &v in &w {
                wh = wh.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
            }
            println!("  warm output_hash = 0x{:016x}", wh);
        }
        None => {
            println!("  warm solve FAILED");
        }
    }

    println!();
}

// =============================================================================
// 8. faer linalg primitives (S1b — does faer's QR / eigendecomp diverge?)
// =============================================================================
//
// The Clarabel test (#7) used a hardcoded cov. In the real backtest, cov
// flows from compute_ewma_factor_covariance → randomized_pca, which calls
// three faer ops: col_piv_qr, compute_thin_q, selfadjoint_eigendecomposition.
// If faer dispatches to a platform-specific BLAS, the cov matrix fed to
// Clarabel would already differ cross-platform.
fn test_faer_linalg() {
    use faer::Mat;

    println!("=== 8. faer linalg primitives (S1b test) ===");

    let n = 16;

    // Build a deterministic symmetric positive-definite matrix from int arithmetic
    // (no libm, no RNG). c[i,j] = c[j,i] = ((i*7+j*13) mod 17 - 8) * 1e-3 with
    // diagonal dominance to ensure PD.
    let c = Mat::from_fn(n, n, |i, j| {
        let off = ((((i as i64) * 7 + (j as i64) * 13) % 17) as f64 - 8.0) * 1e-3;
        if i == j {
            0.5 + (i as f64) * 0.01 + off // diag dominates
        } else {
            off
        }
    });

    // ---- Test 1: random projection target Y = C * Omega (with deterministic Omega) ----
    // Use only int arithmetic to avoid any libm in input gen.
    let omega = Mat::from_fn(n, 8, |i, j| {
        let v = ((((i as i64) * 31 + (j as i64) * 17) % 23) as f64 - 11.0) / 11.0;
        v
    });
    let y = &c * &omega;
    let mut yh: u64 = 0;
    for i in 0..n {
        for j in 0..8 {
            yh = yh.wrapping_mul(0x100000001b3).wrapping_add(y.read(i, j).to_bits());
        }
    }
    println!("  Y = C * Omega   hash = 0x{:016x}", yh);

    // ---- Test 2: column-pivoted QR ----
    let qr = y.col_piv_qr();
    let q = qr.compute_thin_q();
    let qrows = q.nrows();
    let qcols = q.ncols();
    let mut qh: u64 = 0;
    for i in 0..qrows {
        for j in 0..qcols {
            qh = qh.wrapping_mul(0x100000001b3).wrapping_add(q.read(i, j).to_bits());
        }
    }
    println!("  Q (thin) {qrows}x{qcols}  hash = 0x{:016x}", qh);

    // ---- Test 3: B_small = Q' * C * Q ----
    let b_small = q.transpose() * &c * &q;
    let bs = b_small.nrows();
    let mut bh: u64 = 0;
    for i in 0..bs {
        for j in 0..bs {
            bh = bh.wrapping_mul(0x100000001b3).wrapping_add(b_small.read(i, j).to_bits());
        }
    }
    println!("  B_small {bs}x{bs}  hash = 0x{:016x}", bh);

    // ---- Test 4: self-adjoint eigendecomposition ----
    let eig = b_small.selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let u = eig.u();
    let mut sh: u64 = 0;
    for i in 0..bs {
        let v = s.column_vector().read(i);
        sh = sh.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits());
    }
    let mut uh: u64 = 0;
    for i in 0..u.nrows() {
        for j in 0..u.ncols() {
            uh = uh.wrapping_mul(0x100000001b3).wrapping_add(u.read(i, j).to_bits());
        }
    }
    println!("  eigenvalues  hash = 0x{:016x}", sh);
    println!("  eigenvectors hash = 0x{:016x}", uh);

    // Print a few specific values for human-readable diff
    println!("  s[0..4]:");
    for i in 0..4.min(bs) {
        let v = s.column_vector().read(i);
        println!("    s[{i}] = {:.20e}  bits=0x{:016x}", v, v.to_bits());
    }

    println!();
}

fn main() {
    print_header();
    test_libm();
    test_hashmap_order();
    test_variance();
    test_sum_order();
    test_fma();
    test_ewma_seed();
    test_clarabel_solve();
    test_faer_linalg();
}
