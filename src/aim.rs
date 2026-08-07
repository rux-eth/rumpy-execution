//! Aim portfolio: w_aim = Σ⁻¹ · α / γ, projected dollar-neutral, scaled to l_target.
//!
//! Ports `compute_aim_portfolio` from `aim.py`. The aim portfolio is the cost-free
//! Markowitz target — the QP uses it as the anchor in ||w - w_aim||² penalty.

use crate::covariance::RollingCovariance;
use crate::alpha::AlphaRow;

/// Per-bar aim weight.
#[derive(Debug, Clone)]
pub struct AimWeight {
    pub timestamp: i64,
    pub symbol: String,
    pub weight_aim: f64,
}

/// Compute aim portfolio for all bars.
/// w_aim(t) = Σ(t)⁻¹ · α_future(t), then project dollar-neutral and scale to l_target.
pub fn compute_aim_portfolio(
    alphas: &[AlphaRow],
    rolling_cov: &RollingCovariance,
    l_target: f64,
    ridge: f64,
) -> Vec<AimWeight> {
    use std::collections::HashMap;

    // Group alphas by timestamp
    let mut by_ts: HashMap<i64, Vec<(String, f64)>> = HashMap::new();
    for row in alphas {
        by_ts.entry(row.timestamp).or_default().push((row.symbol.clone(), row.alpha_future));
    }

    let mut result = Vec::new();

    for (&ts, assets) in &by_ts {
        let syms: Vec<String> = assets.iter().map(|(s, _)| s.clone()).collect();
        let alpha_vec: Vec<f64> = assets.iter().map(|(_, a)| *a).collect();

        let (cov, syms_present) = rolling_cov.get(ts, &syms);
        let n = syms_present.len();
        if n < 2 {
            continue;
        }

        // Align alpha to syms_present order
        let sym_to_alpha: HashMap<&str, f64> = syms.iter().zip(alpha_vec.iter()).map(|(s, a)| (s.as_str(), *a)).collect();
        let alpha_aligned: Vec<f64> = syms_present.iter().map(|s| *sym_to_alpha.get(s.as_str()).unwrap_or(&0.0)).collect();

        // Ridge for numerical stability
        let mut cov_reg = cov.clone();
        for i in 0..n {
            cov_reg[i * n + i] += ridge;
        }

        // Solve Σ⁻¹ · α via Cholesky or LU
        let w_raw = match solve_linear(&cov_reg, &alpha_aligned, n) {
            Some(w) => w,
            None => continue,
        };

        // Dollar-neutral projection: subtract mean
        let mean = w_raw.iter().sum::<f64>() / n as f64;
        let w_neutral: Vec<f64> = w_raw.iter().map(|w| w - mean).collect();

        // Scale to l_target gross
        let gross: f64 = w_neutral.iter().map(|w| w.abs()).sum();
        if gross < 1e-12 {
            continue;
        }
        let scale = l_target / gross;

        for (i, sym) in syms_present.iter().enumerate() {
            result.push(AimWeight {
                timestamp: ts,
                symbol: sym.clone(),
                weight_aim: w_neutral[i] * scale,
            });
        }
    }

    result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp).then(a.symbol.cmp(&b.symbol)));
    result
}

/// Solve Ax = b via Cholesky decomposition (A must be positive definite).
/// Falls back to None on failure.
fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    use faer::Mat;
    use faer::linalg::solvers::SpSolver;

    let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
    let rhs = Mat::from_fn(n, 1, |i, _| b[i]);

    // Try Cholesky first (fastest for PSD)
    match mat.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let x = chol.solve(&rhs);
            Some((0..n).map(|i| x.read(i, 0)).collect())
        }
        Err(_) => {
            // Fallback to LU
            let lu = mat.partial_piv_lu();
            let x = lu.solve(&rhs);
            Some((0..n).map(|i| x.read(i, 0)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::RollingCovariance;

    #[test]
    fn test_aim_portfolio_basic() {
        // 3 assets, 1 bar
        let alphas = vec![
            AlphaRow { timestamp: 100, symbol: "BTC".into(), alpha_future: 0.05 },
            AlphaRow { timestamp: 100, symbol: "ETH".into(), alpha_future: -0.03 },
            AlphaRow { timestamp: 100, symbol: "SOL".into(), alpha_future: 0.01 },
        ];

        let mut cov = RollingCovariance::new();
        // Simple diagonal covariance
        let n = 3;
        let mut cov_flat = vec![0.0; n * n];
        cov_flat[0] = 0.01; // BTC
        cov_flat[4] = 0.015; // ETH
        cov_flat[8] = 0.02; // SOL
        cov.insert(100, vec!["BTC".into(), "ETH".into(), "SOL".into()], cov_flat);

        let aim = compute_aim_portfolio(&alphas, &cov, 1.0, 1e-6);

        assert_eq!(aim.len(), 3);

        // Dollar neutral
        let sum: f64 = aim.iter().map(|w| w.weight_aim).sum();
        assert!(sum.abs() < 1e-6, "not dollar neutral: {sum}");

        // Gross = l_target = 1.0
        let gross: f64 = aim.iter().map(|w| w.weight_aim.abs()).sum();
        assert!((gross - 1.0).abs() < 1e-6, "gross not 1.0: {gross}");

        // BTC should be long (positive alpha), ETH short (negative alpha)
        let btc = aim.iter().find(|w| w.symbol == "BTC").unwrap();
        let eth = aim.iter().find(|w| w.symbol == "ETH").unwrap();
        assert!(btc.weight_aim > 0.0, "BTC should be long");
        assert!(eth.weight_aim < 0.0, "ETH should be short");
    }

    #[test]
    fn test_solve_linear() {
        // 2x2: [[2, 1], [1, 3]] * x = [5, 7] → x = [1.6, 1.8]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 7.0];
        let x = solve_linear(&a, &b, 2).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
}
