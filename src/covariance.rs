//! EWMA factor covariance estimation: Σ = BFB' + D.
//!
//! Optimized pipeline:
//!   1. EWMA rank-1 covariance update: O(N²) per bar (unavoidable)
//!   2. Randomized PCA every P bars: O(N²k) instead of O(N³)
//!   3. Factor storage: (B, F, D) per timestamp instead of full N×N
//!   4. Submatrix reconstruction on demand in get(): O(n·k²)
//!
//! References:
//!   - Halko, Martinsson, Tropp (2011) "Finding structure with randomness"
//!   - Boyd et al. (2017) for the factor covariance QP formulation

use std::collections::HashMap;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// Factor model storage (replaces full N×N per timestamp)
// ---------------------------------------------------------------------------

/// Per-timestamp factor model: Σ = BFB' + D.
/// B: N×k loadings, F: k×k factor covariance (diagonal = eigenvalues),
/// D: N idiosyncratic variances.
#[derive(Clone)]
struct FactorModel {
    /// Factor loadings, row-major N×k.
    b: Vec<f64>,
    /// Eigenvalues (k values, F = diag(eigenvalues)).
    eigenvalues: Vec<f64>,
    /// Idiosyncratic variances (N values).
    d: Vec<f64>,
    /// Ridge added to diagonal.
    ridge: f64,
    n: usize,
    k: usize,
}

impl FactorModel {
    /// Reconstruct the (i,j) element of Σ = BFB' + D + ridge·I.
    #[inline]
    fn element(&self, i: usize, j: usize) -> f64 {
        let mut val = 0.0;
        for f in 0..self.k {
            val += self.eigenvalues[f] * self.b[i * self.k + f] * self.b[j * self.k + f];
        }
        if i == j {
            val += self.d[i] + self.ridge;
        }
        val
    }

    /// Extract submatrix for a subset of indices.
    fn submatrix(&self, indices: &[usize]) -> Vec<f64> {
        let m = indices.len();
        let mut sub = vec![0.0f64; m * m];
        for (ri, &i) in indices.iter().enumerate() {
            for (ci, &j) in indices.iter().enumerate() {
                sub[ri * m + ci] = self.element(i, j);
            }
        }
        sub
    }
}

// ---------------------------------------------------------------------------
// RollingCovariance (public API unchanged)
// ---------------------------------------------------------------------------

/// Per-timestamp covariance stored as factor models.
pub struct RollingCovariance {
    /// ts → (symbols, factor_model)
    factors: HashMap<i64, (Vec<String>, FactorModel)>,
}

impl Default for RollingCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl RollingCovariance {
    pub fn new() -> Self {
        Self {
            factors: HashMap::new(),
        }
    }

    /// Insert a precomputed factor model for a timestamp.
    fn insert_factor(&mut self, ts: i64, symbols: Vec<String>, model: FactorModel) {
        self.factors.insert(ts, (symbols, model));
    }

    /// Legacy insert for full covariance matrix (used by diagonal fallback).
    pub fn insert(&mut self, ts: i64, symbols: Vec<String>, cov: Vec<f64>) {
        let n = symbols.len();
        // Store as trivial factor model: k=0, D = diagonal of cov
        let d: Vec<f64> = (0..n).map(|i| cov[i * n + i]).collect();
        let model = FactorModel {
            b: vec![],
            eigenvalues: vec![],
            d,
            ridge: 0.0,
            n,
            k: 0,
        };
        self.factors.insert(ts, (symbols, model));
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Write the cov bytes (no magic header) to any `Write`. Used by the
    /// preproc sidecar to embed cov inside a larger file.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut tss: Vec<i64> = self.factors.keys().copied().collect();
        tss.sort();
        w.write_all(&(tss.len() as u64).to_le_bytes())?;
        for ts in tss {
            let (syms, model) = &self.factors[&ts];
            w.write_all(&ts.to_le_bytes())?;
            w.write_all(&(syms.len() as u32).to_le_bytes())?;
            for s in syms {
                let b = s.as_bytes();
                w.write_all(&(b.len() as u32).to_le_bytes())?;
                w.write_all(b)?;
            }
            w.write_all(&(model.n as u64).to_le_bytes())?;
            w.write_all(&(model.k as u64).to_le_bytes())?;
            w.write_all(&model.ridge.to_le_bytes())?;
            for v in &model.b { w.write_all(&v.to_le_bytes())?; }
            for v in &model.eigenvalues { w.write_all(&v.to_le_bytes())?; }
            for v in &model.d { w.write_all(&v.to_le_bytes())?; }
        }
        Ok(())
    }

    /// Read the cov bytes (no magic header) from any `Read`.
    pub fn read_from<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n_ts = u64::from_le_bytes(buf8) as usize;
        let mut factors: HashMap<i64, (Vec<String>, FactorModel)> = HashMap::with_capacity(n_ts);
        for _ in 0..n_ts {
            r.read_exact(&mut buf8)?;
            let ts = i64::from_le_bytes(buf8);
            r.read_exact(&mut buf4)?;
            let n_syms = u32::from_le_bytes(buf4) as usize;
            let mut syms = Vec::with_capacity(n_syms);
            for _ in 0..n_syms {
                r.read_exact(&mut buf4)?;
                let len = u32::from_le_bytes(buf4) as usize;
                let mut sb = vec![0u8; len];
                r.read_exact(&mut sb)?;
                syms.push(String::from_utf8(sb).map_err(|e|
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e))?);
            }
            r.read_exact(&mut buf8)?;
            let n = u64::from_le_bytes(buf8) as usize;
            r.read_exact(&mut buf8)?;
            let k = u64::from_le_bytes(buf8) as usize;
            r.read_exact(&mut buf8)?;
            let ridge = f64::from_le_bytes(buf8);
            let mut b_vec = vec![0.0f64; n * k];
            for v in b_vec.iter_mut() {
                r.read_exact(&mut buf8)?;
                *v = f64::from_le_bytes(buf8);
            }
            let mut eigenvalues = vec![0.0f64; k];
            for v in eigenvalues.iter_mut() {
                r.read_exact(&mut buf8)?;
                *v = f64::from_le_bytes(buf8);
            }
            let mut d = vec![0.0f64; n];
            for v in d.iter_mut() {
                r.read_exact(&mut buf8)?;
                *v = f64::from_le_bytes(buf8);
            }
            let model = FactorModel { b: b_vec, eigenvalues, d, ridge, n, k };
            factors.insert(ts, (syms, model));
        }
        Ok(RollingCovariance { factors })
    }


    /// Extract the Σ submatrix for a subset of symbols at timestamp ts.
    /// Returns (flattened submatrix row-major, symbols_present).
    ///
    /// If exact timestamp not found, floors to day-start (H1 → D1 lookup).
    pub fn get(&self, ts: i64, symbols: &[String]) -> (Vec<f64>, Vec<String>) {
        let entry = self.factors.get(&ts).or_else(|| {
            let day_start = (ts / 86400) * 86400;
            self.factors.get(&day_start)
        });
        let Some((all_syms, model)) = entry else {
            return (vec![], vec![]);
        };
        let _n = all_syms.len();
        let sym_to_idx: HashMap<&str, usize> = all_syms.iter().enumerate()
            .map(|(i, s)| (s.as_str(), i)).collect();

        let mut indices = Vec::new();
        let mut syms_present = Vec::new();
        for s in symbols {
            if let Some(&idx) = sym_to_idx.get(s.as_str()) {
                indices.push(idx);
                syms_present.push(s.clone());
            }
        }

        let sub = model.submatrix(&indices);
        (sub, syms_present)
    }
}

// ---------------------------------------------------------------------------
// Randomized PCA
// ---------------------------------------------------------------------------

/// Randomized PCA via Halko-Martinsson-Tropp (2011).
/// Returns (B: N×k row-major, eigenvalues: k, D: N idiosyncratic, ridge).
/// O(N²k) instead of O(N³).
fn randomized_pca(
    sub_cov: &[f64],
    n: usize,
    k: usize,
    rng_seed: u64,
) -> FactorModel {
    use faer::Mat;

    let p = 5; // oversampling
    let l = (k + p).min(n);

    let mat = Mat::from_fn(n, n, |i, j| sub_cov[i * n + j]);

    // 1. Random Gaussian projection Ω (N×l)
    let mut rng_state = rng_seed;
    let omega = Mat::from_fn(n, l, |_, _| {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.max(1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    });

    // 2. Y = C × Ω
    let y = &mat * &omega;

    // 3. QR → thin Q
    let qr = y.col_piv_qr();
    let q = qr.compute_thin_q();

    // 4. Project: B_small = Q'CQ (l×l)
    let b_small = q.transpose() * &mat * &q;

    // 5. Eigendecompose B_small
    let eig = b_small.selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let u_small = eig.u();

    // 6. Recover: V = Q × U_small
    let v_full = &q * &u_small;

    let evals: Vec<f64> = (0..l).map(|i| s.column_vector().read(i)).collect();

    // Sort descending
    let mut idx: Vec<usize> = (0..l).collect();
    idx.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let k_use = k.min(l);

    // Build B (N×k row-major) and eigenvalues
    let mut b_flat = vec![0.0f64; n * k_use];
    let mut eigenvalues = vec![0.0f64; k_use];
    for fi in 0..k_use {
        eigenvalues[fi] = evals[idx[fi]].max(1e-10);
        for i in 0..n {
            b_flat[i * k_use + fi] = v_full.read(i, idx[fi]);
        }
    }

    // Compute BFB' diagonal for residuals
    let mut bfb_diag = vec![0.0f64; n];
    for i in 0..n {
        for fi in 0..k_use {
            bfb_diag[i] += eigenvalues[fi] * b_flat[i * k_use + fi].powi(2);
        }
    }

    // D = max(diag(cov) - diag(BFB'), 1e-10)
    let d: Vec<f64> = (0..n).map(|i| {
        (sub_cov[i * n + i] - bfb_diag[i]).max(1e-10)
    }).collect();

    // Ridge
    let trace: f64 = bfb_diag.iter().zip(d.iter()).map(|(b, d)| b + d).sum();
    let ridge = 1e-5 * trace / n as f64;

    FactorModel {
        b: b_flat,
        eigenvalues,
        d,
        ridge,
        n,
        k: k_use,
    }
}

// ---------------------------------------------------------------------------
// Main computation
// ---------------------------------------------------------------------------

/// Compute EWMA factor covariance across all timestamps.
///
/// Optimizations vs naive:
///   - Randomized PCA: O(N²k) instead of O(N³) per PCA call
///   - PCA skip: only recompute every `pca_interval` bars
///   - Factor storage: store (B,F,D) instead of full N×N
pub fn compute_ewma_factor_covariance(
    returns: &[f64],
    n_bars: usize,
    n_syms: usize,
    timestamps: &[i64],
    symbols: &[String],
    n_factors: usize,
    halflife: usize,
    min_history: usize,
    min_assets: usize,
) -> RollingCovariance {
    assert_eq!(returns.len(), n_bars * n_syms);
    assert_eq!(timestamps.len(), n_bars);
    assert_eq!(symbols.len(), n_syms);

    let alpha = 1.0 - (-2.0_f64.ln() / halflife as f64).exp();
    let pca_interval = 5; // recompute PCA every 5 bars

    let mut ewma_mean = vec![0.0f64; n_syms];
    let mut ewma_cov = vec![0.0f64; n_syms * n_syms];
    let mut obs_count = vec![0u32; n_syms];
    let mut initialized = false;

    let mut result = RollingCovariance::new();
    let mut last_model: Option<(Vec<usize>, FactorModel)> = None;
    let mut bars_since_pca = 0usize;

    let mut t_ewma_ns = 0u64;
    let mut t_pca_ns = 0u64;
    let mut n_pca_calls = 0usize;
    let mut n_reused = 0usize;

    for bar in 0..n_bars {
        let row = &returns[bar * n_syms..(bar + 1) * n_syms];

        let mut valid = vec![false; n_syms];
        for j in 0..n_syms {
            if row[j].is_finite() {
                valid[j] = true;
                obs_count[j] += 1;
            }
        }

        let mut active_idx: Vec<usize> = Vec::new();
        for j in 0..n_syms {
            if obs_count[j] >= min_history as u32 {
                active_idx.push(j);
            }
        }
        let n_active = active_idx.len();
        if n_active < min_assets {
            continue;
        }

        let r: Vec<f64> = row.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();

        if !initialized {
            ewma_mean = r;
            initialized = true;
            continue;
        }

        // EWMA update (O(N²) — unavoidable)
        let te = std::time::Instant::now();
        let mut diff = vec![0.0f64; n_syms];
        for j in 0..n_syms {
            diff[j] = r[j] - ewma_mean[j];
            if valid[j] {
                ewma_mean[j] += alpha * diff[j];
            }
        }
        // Symmetric rank-1 update (upper triangle + mirror)
        for j in 0..n_syms {
            if !valid[j] { continue; }
            let dj = diff[j];
            for kk in j..n_syms {
                if !valid[kk] { continue; }
                let update = alpha * dj * diff[kk];
                let idx = j * n_syms + kk;
                ewma_cov[idx] = (1.0 - alpha) * ewma_cov[idx] + update;
                if kk != j {
                    let idx2 = kk * n_syms + j;
                    ewma_cov[idx2] = ewma_cov[idx];
                }
            }
            // Decay non-valid pairs
            for kk in 0..n_syms {
                if !valid[kk] && kk != j {
                    let idx = j * n_syms + kk;
                    ewma_cov[idx] *= 1.0 - alpha;
                    ewma_cov[kk * n_syms + j] = ewma_cov[idx];
                }
            }
        }
        t_ewma_ns += te.elapsed().as_nanos() as u64;

        let active_syms: Vec<String> = active_idx.iter().map(|&i| symbols[i].clone()).collect();
        let k = n_factors.min(n_active - 1);

        if n_active < n_factors + 2 {
            // Too few for PCA — diagonal fallback
            let d: Vec<f64> = active_idx.iter()
                .map(|&i| ewma_cov[i * n_syms + i].max(1e-10))
                .collect();
            let model = FactorModel {
                b: vec![], eigenvalues: vec![], d, ridge: 0.0,
                n: n_active, k: 0,
            };
            result.insert_factor(timestamps[bar], active_syms, model);
            continue;
        }

        bars_since_pca += 1;

        // Decide: run PCA or reuse last model
        let need_pca = last_model.is_none()
            || bars_since_pca >= pca_interval
            || last_model.as_ref().map(|(idx, _)| idx != &active_idx).unwrap_or(true);

        if need_pca {
            let te = std::time::Instant::now();

            // Extract active submatrix
            let mut sub_cov = vec![0.0f64; n_active * n_active];
            for (ri, &i) in active_idx.iter().enumerate() {
                for (ci, &j) in active_idx.iter().enumerate() {
                    sub_cov[ri * n_active + ci] = ewma_cov[i * n_syms + j];
                }
            }

            let model = randomized_pca(&sub_cov, n_active, k, bar as u64 * 31337 + 42);

            t_pca_ns += te.elapsed().as_nanos() as u64;
            n_pca_calls += 1;
            bars_since_pca = 0;

            result.insert_factor(timestamps[bar], active_syms, model.clone());
            last_model = Some((active_idx.clone(), model));
        } else {
            // Reuse last PCA loadings, update F and D from current EWMA cov
            let te = std::time::Instant::now();

            let (_, ref prev_model) = last_model.as_ref().unwrap();
            let prev_k = prev_model.k;

            // Reproject: F_new = B' C_sub B, D_new = diag(C_sub) - diag(BF_newB')
            // First extract active submatrix diagonal and project
            let mut new_eigenvalues = vec![0.0f64; prev_k];
            let mut bfb_diag = vec![0.0f64; n_active];

            // F_new[fi,fj] = sum_i sum_j B[i,fi] * C[i,j] * B[j,fj]
            // Since F is diagonal in our model, we approximate: F_new[fi] = B[:,fi]' C B[:,fi]
            for fi in 0..prev_k {
                let mut val = 0.0;
                for (ri, &i) in active_idx.iter().enumerate() {
                    let bi = prev_model.b[ri * prev_k + fi];
                    for (ci, &j) in active_idx.iter().enumerate() {
                        let bj = prev_model.b[ci * prev_k + fi];
                        val += bi * ewma_cov[i * n_syms + j] * bj;
                    }
                }
                new_eigenvalues[fi] = val.max(1e-10);
            }

            // BFB' diagonal
            for ri in 0..n_active {
                for fi in 0..prev_k {
                    bfb_diag[ri] += new_eigenvalues[fi] * prev_model.b[ri * prev_k + fi].powi(2);
                }
            }

            let new_d: Vec<f64> = (0..n_active).map(|ri| {
                let i = active_idx[ri];
                (ewma_cov[i * n_syms + i] - bfb_diag[ri]).max(1e-10)
            }).collect();

            let trace: f64 = bfb_diag.iter().zip(new_d.iter()).map(|(b, d)| b + d).sum();
            let ridge = 1e-5 * trace / n_active as f64;

            let model = FactorModel {
                b: prev_model.b.clone(),
                eigenvalues: new_eigenvalues,
                d: new_d,
                ridge,
                n: n_active,
                k: prev_k,
            };

            t_pca_ns += te.elapsed().as_nanos() as u64;
            n_reused += 1;

            result.insert_factor(timestamps[bar], active_syms, model);
        }
    }

    println!("  cov profiling: ewma={:.2}s, pca={:.2}s ({} full + {} reused)",
        t_ewma_ns as f64 / 1e9, t_pca_ns as f64 / 1e9, n_pca_calls, n_reused);

    result
}

// ---------------------------------------------------------------------------
// Utility functions (unchanged API)
// ---------------------------------------------------------------------------

/// Compute log returns from close prices.
pub fn compute_log_returns(closes: &[f64], n_bars: usize, n_syms: usize) -> Vec<f64> {
    let mut returns = vec![f64::NAN; n_bars * n_syms];
    for bar in 1..n_bars {
        for j in 0..n_syms {
            let prev = closes[(bar - 1) * n_syms + j];
            let curr = closes[bar * n_syms + j];
            if prev.is_finite() && curr.is_finite() && prev > 0.0 {
                returns[bar * n_syms + j] = (curr / prev).ln();
            }
        }
    }
    returns
}

/// Winsorize per-asset returns at ±sigma × asset_std.
pub fn winsorize_per_asset(returns: &mut [f64], n_bars: usize, n_syms: usize, sigma: f64) {
    for j in 0..n_syms {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0u32;
        for bar in 0..n_bars {
            let v = returns[bar * n_syms + j];
            if v.is_finite() {
                sum += v;
                sum_sq += v * v;
                count += 1;
            }
        }
        if count < 2 { continue; }
        let mean = sum / count as f64;
        let var = sum_sq / count as f64 - mean * mean;
        let std = var.max(0.0).sqrt();
        let upper = sigma * std;
        let lower = -sigma * std;

        for bar in 0..n_bars {
            let v = returns[bar * n_syms + j];
            if v.is_finite() {
                returns[bar * n_syms + j] = v.clamp(lower, upper);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_factor_basic() {
        let n_bars = 100;
        let n_syms = 10;
        let mut rng_state: u64 = 42;
        let mut returns = vec![0.0f64; n_bars * n_syms];
        for v in returns.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.04;
        }

        let timestamps: Vec<i64> = (0..n_bars as i64).collect();
        let symbols: Vec<String> = (0..n_syms).map(|i| format!("SYM{i}")).collect();

        let cov = compute_ewma_factor_covariance(
            &returns, n_bars, n_syms, &timestamps, &symbols,
            3, 20, 10, 5,
        );

        assert!(cov.len() > 0);

        let last_ts = timestamps[n_bars - 1];
        let (sub, syms) = cov.get(last_ts, &symbols);
        assert!(!syms.is_empty());
        let m = syms.len();

        for i in 0..m {
            assert!(sub[i * m + i] > 0.0, "diagonal must be positive at [{i},{i}]");
        }

        for i in 0..m {
            for j in 0..m {
                let diff = (sub[i * m + j] - sub[j * m + i]).abs();
                assert!(diff < 1e-10, "covariance must be symmetric at [{i},{j}]");
            }
        }
    }

    #[test]
    fn test_winsorize() {
        let n_bars = 10;
        let n_syms = 1;
        let mut returns = vec![0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, 0.50];
        winsorize_per_asset(&mut returns, n_bars, n_syms, 3.0);
        assert!(returns[9] < 0.50);
        assert!((returns[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 200.0, 110.0, 190.0, 121.0, 180.0];
        let returns = compute_log_returns(&closes, 3, 2);
        assert!(returns[0].is_nan());
        assert!(returns[1].is_nan());
        assert!((returns[2] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[3] - (190.0_f64 / 200.0).ln()).abs() < 1e-10);
    }
}
