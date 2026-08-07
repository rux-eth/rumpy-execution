//! Per-stage diagnostics for the execution pipeline.
//!
//! Captures state after each pipeline stage so bugs can be traced
//! to the exact step where numbers go wrong.

use std::collections::HashMap;

use crate::pipeline::WeightOutput;

/// Per-stage aggregate snapshot.
#[derive(Debug, Clone)]
pub struct StageSnapshot {
    pub stage: String,
    pub n_bars: usize,
    pub n_positions: usize,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub max_abs_weight: f64,
    pub mean_abs_weight: f64,
    pub turnover_vs_prev: f64,
}

/// Per-bar aggregate diagnostics.
#[derive(Debug, Clone)]
pub struct BarDiagnostic {
    pub timestamp: i64,
    pub portfolio_return: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub n_long: usize,
    pub n_short: usize,
    pub turnover: f64,
    pub leverage: f64,
    pub running_nav: f64,
    pub drawdown: f64,
}

/// Compute a stage snapshot from weights.
pub fn snapshot(stage: &str, weights: &[WeightOutput], prev_weights: Option<&[WeightOutput]>) -> StageSnapshot {
    let mut by_ts: HashMap<i64, Vec<f64>> = HashMap::new();
    for w in weights {
        by_ts.entry(w.timestamp).or_default().push(w.weight_final);
    }

    let n_bars = by_ts.len();
    let n_positions = weights.iter().filter(|w| w.weight_final.abs() > 1e-9).count();

    let mut total_gross = 0.0;
    let mut total_net = 0.0;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0;

    for ws in by_ts.values() {
        let bar_gross: f64 = ws.iter().map(|w| w.abs()).sum();
        let bar_net: f64 = ws.iter().sum();
        total_gross += bar_gross;
        total_net += bar_net;
        for &w in ws {
            max_abs = max_abs.max(w.abs());
            sum_abs += w.abs();
        }
    }

    let avg_gross = if n_bars > 0 { total_gross / n_bars as f64 } else { 0.0 };
    let avg_net = if n_bars > 0 { total_net / n_bars as f64 } else { 0.0 };
    let mean_abs = if n_positions > 0 { sum_abs / n_positions as f64 } else { 0.0 };

    // Turnover vs previous stage
    let turnover = if let Some(prev) = prev_weights {
        let prev_map: HashMap<(i64, &str), f64> = prev.iter()
            .map(|w| ((w.timestamp, w.symbol.as_str()), w.weight_final))
            .collect();
        let mut total_delta = 0.0;
        for w in weights {
            let prev_w = prev_map.get(&(w.timestamp, w.symbol.as_str())).copied().unwrap_or(0.0);
            total_delta += (w.weight_final - prev_w).abs();
        }
        if n_bars > 0 { total_delta / n_bars as f64 } else { 0.0 }
    } else {
        0.0
    };

    StageSnapshot {
        stage: stage.to_string(),
        n_bars,
        n_positions,
        gross_exposure: avg_gross,
        net_exposure: avg_net,
        max_abs_weight: max_abs,
        mean_abs_weight: mean_abs,
        turnover_vs_prev: turnover,
    }
}

/// Print a stage snapshot as a one-line summary.
pub fn print_snapshot(s: &StageSnapshot) {
    println!(
        "    [{:>15}] bars={}, pos={}, gross={:.3}, net={:.4}, max|w|={:.4}, mean|w|={:.4}, Δ={:.4}",
        s.stage, s.n_bars, s.n_positions, s.gross_exposure, s.net_exposure,
        s.max_abs_weight, s.mean_abs_weight, s.turnover_vs_prev,
    );
}

/// Compute Sharpe from weights at current stage (for debugging pipeline stages).
pub fn stage_sharpe(
    weights: &[WeightOutput],
    forward_returns: &HashMap<(i64, String), f64>,
    bars_per_year: f64,
) -> f64 {
    let returns = crate::metrics::portfolio_return_per_bar(weights, forward_returns);
    let vals: Vec<f64> = returns.iter().map(|(_, r)| *r).collect();
    crate::metrics::sharpe_annualized(&vals, bars_per_year)
}

/// Print stage snapshot + Sharpe.
pub fn print_snapshot_with_sharpe(
    s: &StageSnapshot,
    weights: &[WeightOutput],
    forward_returns: &HashMap<(i64, String), f64>,
    bars_per_year: f64,
) {
    let sr = stage_sharpe(weights, forward_returns, bars_per_year);
    let returns = crate::metrics::portfolio_return_per_bar(weights, forward_returns);
    let vals: Vec<f64> = returns.iter().map(|(_, r)| *r).collect();
    let ann_ret = crate::metrics::annualized_return(&vals, bars_per_year);
    let mdd = crate::metrics::max_drawdown(&vals);
    println!(
        "    [{:>15}] gross={:.3}, max|w|={:.4}, Sharpe={:.2}, AnnRet={:.1}%, MDD={:.1}%",
        s.stage, s.gross_exposure, s.max_abs_weight, sr, ann_ret * 100.0, mdd * 100.0,
    );
}

/// Compute per-bar diagnostics from final weights + forward returns.
pub fn per_bar_diagnostics(
    weights: &[WeightOutput],
    forward_returns: &HashMap<(i64, String), f64>,
    nav_start: f64,
) -> Vec<BarDiagnostic> {
    // Group weights by timestamp
    let mut by_ts: HashMap<i64, Vec<(&WeightOutput, f64)>> = HashMap::new();
    for w in weights {
        let fwd = forward_returns.get(&(w.timestamp, w.symbol.clone())).copied().unwrap_or(0.0);
        by_ts.entry(w.timestamp).or_default().push((w, fwd));
    }

    let mut timestamps: Vec<i64> = by_ts.keys().copied().collect();
    timestamps.sort();

    // Track prev weights for turnover
    let mut prev_by_sym: HashMap<&str, f64> = HashMap::new();
    let mut nav = nav_start;
    let mut peak_nav = nav_start;
    let mut result = Vec::with_capacity(timestamps.len());

    for &ts in &timestamps {
        let bar = &by_ts[&ts];

        let mut port_ret = 0.0;
        let mut gross = 0.0;
        let mut net = 0.0;
        let mut n_long = 0usize;
        let mut n_short = 0usize;
        let mut turnover = 0.0;

        for (w, fwd) in bar {
            port_ret += w.weight_final * fwd;
            gross += w.weight_final.abs();
            net += w.weight_final;
            if w.weight_final > 1e-9 { n_long += 1; }
            if w.weight_final < -1e-9 { n_short += 1; }

            let prev = prev_by_sym.get(w.symbol.as_str()).copied().unwrap_or(0.0);
            turnover += (w.weight_final - prev).abs();
        }

        nav *= 1.0 + port_ret;
        if nav > peak_nav { peak_nav = nav; }
        let dd = if peak_nav > 0.0 { (nav - peak_nav) / peak_nav } else { 0.0 };

        result.push(BarDiagnostic {
            timestamp: ts,
            portfolio_return: port_ret,
            gross_exposure: gross,
            net_exposure: net,
            n_long,
            n_short,
            turnover,
            leverage: gross, // gross exposure = effective leverage
            running_nav: nav,
            drawdown: dd,
        });

        // Update prev weights
        prev_by_sym.clear();
        for (w, _) in bar {
            prev_by_sym.insert(&w.symbol, w.weight_final);
        }
    }

    result
}

/// Monthly portfolio snapshot.
#[derive(Debug, Clone)]
pub struct MonthlySnapshot {
    pub month: String, // "2024-01"
    pub n_bars: usize,
    pub n_long: usize,
    pub n_short: usize,
    pub gross_exposure: f64,
    pub long_exposure: f64,
    pub short_exposure: f64,
    pub portfolio_return: f64,
    pub mdd: f64,
    pub turnover: f64,
    pub total_cost_bps: f64,
    pub impact_cost_bps: f64,
    pub spread_cost_bps: f64,
    pub fee_cost_bps: f64,
    /// Top 5 most expensive assets this month (by total cost)
    pub top_cost_assets: Vec<(String, f64)>,
    /// Top 5 contributors (by PnL)
    pub top_contributors: Vec<(String, f64)>,
    /// Bottom 5 contributors (by PnL)
    pub bottom_contributors: Vec<(String, f64)>,
}

/// Compute monthly snapshots from weights + forward returns + cost data.
pub fn monthly_snapshots(
    weights: &[WeightOutput],
    forward_returns: &HashMap<(i64, String), f64>,
    spread_lookup: &HashMap<(i64, String), f64>,
    kappa_lookup: &HashMap<(i64, String), f64>,
    nav_usd: f64,
    taker_bps: f64,
) -> Vec<MonthlySnapshot> {
    use std::collections::BTreeMap;

    // Group weights by month
    let mut by_month: BTreeMap<String, Vec<&WeightOutput>> = BTreeMap::new();
    for w in weights {
        // Convert epoch seconds to YYYY-MM
        let days = w.timestamp / 86400;
        let _year = 1970 + (days / 365); // approximate
        // More accurate: use actual date computation
        let ts = w.timestamp;
        let (y, m, _) = epoch_to_ymd(ts);
        let key = format!("{y:04}-{m:02}");
        by_month.entry(key).or_default().push(w);
    }

    let mut snapshots = Vec::new();

    for (month, month_weights) in &by_month {
        // Per-bar stats within this month
        let mut by_ts: BTreeMap<i64, Vec<&WeightOutput>> = BTreeMap::new();
        for w in month_weights {
            by_ts.entry(w.timestamp).or_default().push(w);
        }

        let n_bars = by_ts.len();
        let mut month_return = 0.0;
        let mut month_turnover = 0.0;
        let mut total_gross = 0.0;
        let mut total_long = 0.0;
        let mut total_short = 0.0;
        let mut total_n_long = 0usize;
        let mut total_n_short = 0usize;
        let mut month_fee = 0.0;
        let mut month_spread = 0.0;
        let mut month_impact = 0.0;

        // Per-asset tracking
        let mut asset_pnl: HashMap<String, f64> = HashMap::new();
        let mut asset_cost: HashMap<String, f64> = HashMap::new();

        let mut prev_w: HashMap<&str, f64> = HashMap::new();
        let mut cum_return = 0.0;
        let mut peak_return = 0.0;
        let mut worst_dd = 0.0;

        for (_ts, bar_weights) in &by_ts {
            let mut bar_return = 0.0;
            let mut bar_gross = 0.0;
            let mut bar_long = 0.0;
            let mut bar_short = 0.0;
            let mut bar_n_long = 0usize;
            let mut bar_n_short = 0usize;
            let mut bar_turnover = 0.0;

            for w in bar_weights {
                let fwd = forward_returns.get(&(w.timestamp, w.symbol.clone())).copied().unwrap_or(0.0);
                let contrib = w.weight_final * fwd;
                bar_return += contrib;
                bar_gross += w.weight_final.abs();

                if w.weight_final > 1e-9 {
                    bar_long += w.weight_final;
                    bar_n_long += 1;
                } else if w.weight_final < -1e-9 {
                    bar_short += w.weight_final.abs();
                    bar_n_short += 1;
                }

                // Turnover
                let pw = prev_w.get(w.symbol.as_str()).copied().unwrap_or(0.0);
                let delta = (w.weight_final - pw).abs();
                bar_turnover += delta;

                // Cost decomposition
                let spread_bps = spread_lookup.get(&(w.timestamp, w.symbol.clone())).copied().unwrap_or(0.0);
                let kappa = kappa_lookup.get(&(w.timestamp, w.symbol.clone())).copied().unwrap_or(0.0);
                let trade_notional = delta * nav_usd;
                let fee = taker_bps / 10_000.0 * trade_notional;
                let spread = 0.5 * spread_bps / 10_000.0 * trade_notional;
                let impact = kappa * trade_notional.powf(1.5).max(0.0) * 0.001; // approximate

                month_fee += fee;
                month_spread += spread;
                month_impact += impact;

                *asset_pnl.entry(w.symbol.clone()).or_default() += contrib;
                *asset_cost.entry(w.symbol.clone()).or_default() += fee + spread + impact;
            }

            month_return += bar_return;
            month_turnover += bar_turnover;
            total_gross += bar_gross;
            total_long += bar_long;
            total_short += bar_short;
            total_n_long += bar_n_long;
            total_n_short += bar_n_short;

            // Drawdown tracking
            cum_return += bar_return;
            if cum_return > peak_return { peak_return = cum_return; }
            let dd = cum_return - peak_return;
            if dd < worst_dd { worst_dd = dd; }

            // Update prev weights
            prev_w.clear();
            for w in bar_weights {
                prev_w.insert(&w.symbol, w.weight_final);
            }
        }

        let avg_gross = if n_bars > 0 { total_gross / n_bars as f64 } else { 0.0 };
        let avg_long = if n_bars > 0 { total_long / n_bars as f64 } else { 0.0 };
        let avg_short = if n_bars > 0 { total_short / n_bars as f64 } else { 0.0 };
        let avg_n_long = if n_bars > 0 { total_n_long / n_bars } else { 0 };
        let avg_n_short = if n_bars > 0 { total_n_short / n_bars } else { 0 };
        let avg_turnover = if n_bars > 0 { month_turnover / n_bars as f64 } else { 0.0 };

        let total_cost = month_fee + month_spread + month_impact;
        let cost_bps = if avg_gross > 1e-12 && n_bars > 0 {
            total_cost / (avg_gross * nav_usd * n_bars as f64) * 10_000.0
        } else { 0.0 };

        // Top cost assets
        let mut cost_vec: Vec<(String, f64)> = asset_cost.into_iter().collect();
        cost_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_cost = cost_vec.into_iter().take(5).collect();

        // Top/bottom contributors
        let mut pnl_vec: Vec<(String, f64)> = asset_pnl.into_iter().collect();
        pnl_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_contributors = pnl_vec.iter().take(5).cloned().collect();
        let bottom_contributors = pnl_vec.iter().rev().take(5).cloned().collect();

        snapshots.push(MonthlySnapshot {
            month: month.clone(),
            n_bars,
            n_long: avg_n_long,
            n_short: avg_n_short,
            gross_exposure: avg_gross,
            long_exposure: avg_long,
            short_exposure: avg_short,
            portfolio_return: month_return,
            mdd: worst_dd,
            turnover: avg_turnover,
            total_cost_bps: cost_bps,
            impact_cost_bps: 0.0, // TODO: decompose properly
            spread_cost_bps: 0.0,
            fee_cost_bps: 0.0,
            top_cost_assets: top_cost,
            top_contributors,
            bottom_contributors,
        });
    }

    snapshots
}

/// Convert epoch seconds to (year, month, day).
pub fn epoch_to_ymd(ts: i64) -> (i32, u32, u32) {
    let days = ts / 86400;
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

/// Print monthly snapshots.
pub fn print_monthly_snapshots(snapshots: &[MonthlySnapshot]) {
    if snapshots.is_empty() { return; }
    println!("\n  === Monthly Snapshots ===");
    println!("  {:>7} | {:>5} | {:>3}L {:>3}S | {:>5} | {:>7} | {:>6} | {:>5} | Top Contrib          | Worst Contrib",
        "Month", "Gross", "", "", "Ret%", "MDD%", "TO", "Cost");
    println!("  {}", "-".repeat(110));
    for s in snapshots {
        let top = if !s.top_contributors.is_empty() {
            format!("{} {:.2}%", s.top_contributors[0].0, s.top_contributors[0].1 * 100.0)
        } else { "-".to_string() };
        let bot = if !s.bottom_contributors.is_empty() {
            format!("{} {:.2}%", s.bottom_contributors[0].0, s.bottom_contributors[0].1 * 100.0)
        } else { "-".to_string() };
        println!(
            "  {:>7} | {:>5.2} | {:>3}L {:>3}S | {:>+6.1}% | {:>+6.1}% | {:>.4} | {:>5.1}bp | {:<20} | {}",
            s.month, s.gross_exposure, s.n_long, s.n_short,
            s.portfolio_return * 100.0, s.mdd * 100.0,
            s.turnover, s.total_cost_bps,
            top, bot,
        );
    }
}

/// Print summary stats from per-bar diagnostics.
pub fn print_bar_summary(diags: &[BarDiagnostic]) {
    if diags.is_empty() { return; }
    let n = diags.len();

    let avg_gross: f64 = diags.iter().map(|d| d.gross_exposure).sum::<f64>() / n as f64;
    let max_gross: f64 = diags.iter().map(|d| d.gross_exposure).fold(0.0, f64::max);
    let avg_net: f64 = diags.iter().map(|d| d.net_exposure.abs()).sum::<f64>() / n as f64;
    let avg_positions: f64 = diags.iter().map(|d| (d.n_long + d.n_short) as f64).sum::<f64>() / n as f64;
    let avg_turnover: f64 = diags.iter().map(|d| d.turnover).sum::<f64>() / n as f64;
    let worst_dd: f64 = diags.iter().map(|d| d.drawdown).fold(0.0, f64::min);
    let final_nav = diags.last().unwrap().running_nav;

    println!("\n  === Per-Bar Diagnostics ===");
    println!("  Avg gross exposure:  {:.3}", avg_gross);
    println!("  Max gross exposure:  {:.3}", max_gross);
    println!("  Avg |net exposure|:  {:.4}", avg_net);
    println!("  Avg positions:       {:.0}", avg_positions);
    println!("  Avg turnover/bar:    {:.4}", avg_turnover);
    println!("  Worst drawdown:      {:.1}%", worst_dd * 100.0);
    println!("  Final NAV:           {:.2} (start: 1000)", final_nav);
}
