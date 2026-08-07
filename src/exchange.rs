//! Exchange-agnostic adapter for fees, funding, and margin mechanics.
//!
//! Implemented as a config struct with methods (not a trait). Will convert
//! to a trait when a second exchange is added. Avoids premature abstraction.
//!
//! Default configuration matches Hyperliquid perpetual futures:
//!   - Volume-tiered fees (Tier 0: 4.5 bps taker, 1.5 bps maker)
//!   - Hourly funding
//!   - Cross-margin, 5% maintenance margin rate
//!   - Weighted average cost basis
//!
//! Fee tiers adapt to 14-day trailing volume, mirroring live trading.

use std::collections::VecDeque;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Fee tiers
// ---------------------------------------------------------------------------

/// A single fee tier based on 14-day trailing volume.
#[derive(Debug, Clone, Deserialize)]
pub struct FeeTier {
    /// 14-day trailing volume threshold in USD.
    /// Tier applies when trailing volume >= this value.
    pub volume_threshold: f64,
    /// Taker fee in basis points.
    pub taker_bps: f64,
    /// Maker fee in basis points.
    pub maker_bps: f64,
}

/// Default HL perp fee tiers (as of 2026-04).
fn default_fee_tiers() -> Vec<FeeTier> {
    vec![
        FeeTier { volume_threshold: 0.0,             taker_bps: 4.5, maker_bps: 1.5 },
        FeeTier { volume_threshold: 5_000_000.0,     taker_bps: 4.0, maker_bps: 1.2 },
        FeeTier { volume_threshold: 25_000_000.0,    taker_bps: 3.5, maker_bps: 0.8 },
        FeeTier { volume_threshold: 100_000_000.0,   taker_bps: 3.0, maker_bps: 0.4 },
        FeeTier { volume_threshold: 500_000_000.0,   taker_bps: 2.8, maker_bps: 0.0 },
        FeeTier { volume_threshold: 2_000_000_000.0, taker_bps: 2.6, maker_bps: 0.0 },
        FeeTier { volume_threshold: 7_000_000_000.0, taker_bps: 2.4, maker_bps: 0.0 },
    ]
}

// ---------------------------------------------------------------------------
// Volume tracker
// ---------------------------------------------------------------------------

/// Tracks 14-day trailing volume for fee tier lookup.
///
/// Aggregates trade notionals per day, maintains a sliding window,
/// and provides the current trailing volume for tier determination.
#[derive(Debug, Clone)]
pub struct VolumeTracker {
    /// Window size in seconds (default: 14 * 86400 = 1,209,600).
    window_seconds: i64,
    /// (day_start_timestamp, daily_volume_usd) pairs.
    daily_volumes: VecDeque<(i64, f64)>,
    /// Current day being accumulated.
    current_day: i64,
    /// Volume accumulated for current day.
    current_day_volume: f64,
}

impl VolumeTracker {
    /// Create a new tracker with 14-day window.
    pub fn new() -> Self {
        Self {
            window_seconds: 14 * 86400,
            daily_volumes: VecDeque::new(),
            current_day: 0,
            current_day_volume: 0.0,
        }
    }

    /// Record a trade. `timestamp` is epoch seconds, `notional` is absolute USD.
    pub fn record_trade(&mut self, timestamp: i64, notional: f64) {
        let day = (timestamp / 86400) * 86400;

        if day != self.current_day {
            // Flush previous day
            if self.current_day > 0 && self.current_day_volume > 0.0 {
                self.daily_volumes.push_back((self.current_day, self.current_day_volume));
            }
            self.current_day = day;
            self.current_day_volume = 0.0;

            // Expire old days
            let cutoff = day - self.window_seconds;
            while let Some(&(d, _)) = self.daily_volumes.front() {
                if d < cutoff {
                    self.daily_volumes.pop_front();
                } else {
                    break;
                }
            }
        }

        self.current_day_volume += notional.abs();
    }

    /// Get 14-day trailing volume (including current partial day).
    pub fn trailing_volume(&self) -> f64 {
        let historical: f64 = self.daily_volumes.iter().map(|(_, v)| v).sum();
        historical + self.current_day_volume
    }
}

impl Default for VolumeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How cost basis is computed for positions.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub enum CostBasisMethod {
    /// Weighted average (used by HL, Binance, all major perp exchanges).
    #[default]
    WeightedAverage,
}

/// Exchange-specific configuration for simulation.
///
/// Encapsulates fee schedules, funding mechanics, and margin rules.
/// Defaults to Hyperliquid perpetual futures parameters.
///
/// Fee tiers adapt to 14-day trailing volume — the caller tracks volume
/// via [`VolumeTracker`] and passes trailing volume to [`compute_fee`].
#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeConfig {
    /// Exchange name (e.g. "hyperliquid", "binance").
    #[serde(default = "default_name")]
    pub name: String,

    /// Volume-based fee tiers. Sorted by volume_threshold ascending.
    /// The highest tier where trailing_14d_volume >= threshold applies.
    #[serde(default = "default_fee_tiers")]
    pub fee_tiers: Vec<FeeTier>,

    /// Funding interval in hours. HL=1 (hourly), Binance=8 (8-hourly).
    #[serde(default = "default_funding_interval_hours")]
    pub funding_interval_hours: usize,

    /// Maintenance margin rate (fraction of notional).
    /// HL cross-margin: ~3-5% depending on position size.
    /// Using 5% as conservative default.
    #[serde(default = "default_maintenance_margin_rate")]
    pub maintenance_margin_rate: f64,

    /// Margin ratio at which liquidation triggers (equity / maintenance).
    /// HL: 1.0 (equity < maintenance → market orders to close).
    /// Binance: also ~1.0 but with different maintenance tiers.
    /// Configurable per exchange.
    #[serde(default = "default_liquidation_ratio")]
    pub liquidation_ratio: f64,

    /// Margin ratio below which the simulation scales down new trades to avoid
    /// approaching liquidation. Acts as a soft circuit breaker.
    /// Default 1.5 = start reducing when NAV < 1.5× maintenance.
    #[serde(default = "default_margin_halt_ratio")]
    pub margin_halt_ratio: f64,

    /// Minimum order size in USD. Per HL spec (2026-04-27 verified):
    /// orders that **open or increase** a position must be ≥ this size.
    /// Orders that **close or reduce** a position have NO minimum (any size).
    /// Position flips count as "increasing" (the new exposure side > 0).
    /// For TWAP fills, the rule applies per child order (per bar).
    /// Default 10.0 = HL minimum order notional.
    ///
    /// Backtest enforces this in `simulation.rs` per-bar trade-execution loop.
    /// Pre-2026-04-27 the simulation used a hardcoded $1 dust filter regardless
    /// of direction — that destroyed alpha at low NAV (91% skip at $1k) and
    /// failed to enforce HL's actual minimum. See CALIBRATION-CHECKLIST.md
    /// Tier 0 Part A for full rationale.
    #[serde(default = "default_min_order_size_usd")]
    pub min_order_size_usd: f64,

    /// Funding rate cap per interval (absolute value).
    /// None = no cap (default for backtesting — historical data already reflects
    /// actual exchange caps). Set explicitly for forward simulation.
    #[serde(default)]
    pub funding_cap_per_interval: Option<f64>,

    /// Cost basis method for position tracking.
    #[serde(default)]
    pub cost_basis_method: CostBasisMethod,
}

// HL defaults
fn default_name() -> String { "hyperliquid".to_string() }
fn default_funding_interval_hours() -> usize { 1 }
fn default_maintenance_margin_rate() -> f64 { 0.05 }
fn default_liquidation_ratio() -> f64 { 1.0 } // HL: equity < maintenance
fn default_margin_halt_ratio() -> f64 { 1.5 }
fn default_min_order_size_usd() -> f64 { 10.0 }

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            name: default_name(),
            fee_tiers: default_fee_tiers(),
            funding_interval_hours: default_funding_interval_hours(),
            maintenance_margin_rate: default_maintenance_margin_rate(),
            liquidation_ratio: default_liquidation_ratio(),
            margin_halt_ratio: default_margin_halt_ratio(),
            min_order_size_usd: default_min_order_size_usd(),
            funding_cap_per_interval: None,
            cost_basis_method: CostBasisMethod::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

impl ExchangeConfig {
    /// Look up fee bps for a given 14-day trailing volume.
    ///
    /// Returns (taker_bps, maker_bps) for the highest qualifying tier.
    pub fn fee_bps_for_volume(&self, trailing_14d_volume: f64) -> (f64, f64) {
        let mut taker = self.fee_tiers[0].taker_bps;
        let mut maker = self.fee_tiers[0].maker_bps;
        for tier in &self.fee_tiers {
            if trailing_14d_volume >= tier.volume_threshold {
                taker = tier.taker_bps;
                maker = tier.maker_bps;
            }
        }
        (taker, maker)
    }

    /// Current taker fee in bps for a given trailing volume.
    pub fn current_taker_bps(&self, trailing_14d_volume: f64) -> f64 {
        self.fee_bps_for_volume(trailing_14d_volume).0
    }

    /// Compute trading fee for a given trade notional and trailing volume.
    ///
    /// `trade_notional`: USD value of the trade (sign doesn't matter).
    /// `is_maker`: whether this is a maker (passive) or taker (aggressive) order.
    /// `trailing_14d_volume`: 14-day trailing volume for tier lookup.
    ///
    /// Returns the fee in USD (always non-negative for taker).
    pub fn compute_fee(&self, trade_notional: f64, is_maker: bool, trailing_14d_volume: f64) -> f64 {
        let (taker_bps, maker_bps) = self.fee_bps_for_volume(trailing_14d_volume);
        let fee_bps = if is_maker { maker_bps } else { taker_bps };
        fee_bps / 10_000.0 * trade_notional.abs()
    }

    /// Compute funding payment for a position.
    ///
    /// Convention: positive `funding_rate` means longs pay shorts.
    /// Returns the cash flow from the position holder's perspective:
    ///   - Negative = holder pays
    ///   - Positive = holder receives
    ///
    /// `qty`: signed position quantity (positive=long, negative=short).
    /// `oracle_price`: oracle/index price used for funding (not mark price on HL).
    /// `funding_rate`: the funding rate for this interval.
    pub fn compute_funding_payment(&self, qty: f64, oracle_price: f64, funding_rate: f64) -> f64 {
        if qty.abs() < 1e-12 {
            return 0.0;
        }

        // Apply cap if configured
        let rate = match self.funding_cap_per_interval {
            Some(cap) => funding_rate.clamp(-cap, cap),
            None => funding_rate,
        };

        // Payment = |qty| × oracle_price × |rate|
        let payment = qty.abs() * oracle_price * rate.abs();

        // Direction: positive rate → longs pay, shorts receive
        if rate > 0.0 {
            if qty > 0.0 { -payment } else { payment }
        } else if rate < 0.0 {
            if qty > 0.0 { payment } else { -payment }
        } else {
            0.0
        }
    }

    /// Compute maintenance margin requirement for a given notional exposure.
    ///
    /// For cross-margin (HL default): maintenance = rate × |notional|.
    pub fn maintenance_margin(&self, notional: f64) -> f64 {
        self.maintenance_margin_rate * notional.abs()
    }

    /// Check if liquidation triggers.
    ///
    /// Liquidation when margin_ratio < liquidation_ratio.
    /// HL default: liquidation_ratio = 1.0 (equity < maintenance).
    /// Other exchanges may use different thresholds.
    pub fn is_liquidatable(&self, nav: f64, total_maintenance_margin: f64) -> bool {
        total_maintenance_margin > 1e-12 && nav < self.liquidation_ratio * total_maintenance_margin
    }

    /// Check if margin is under pressure (soft halt).
    /// Returns true when margin_ratio < margin_halt_ratio.
    pub fn is_margin_stressed(&self, nav: f64, total_maintenance_margin: f64) -> bool {
        if total_maintenance_margin < 1e-12 { return false; }
        nav / total_maintenance_margin < self.margin_halt_ratio
    }

    /// Distance to liquidation as a fraction: 0.0 = at liquidation threshold, 1.0+ = safe.
    /// Computed as (nav - liq_threshold) / nav. Negative = underwater.
    pub fn liquidation_distance(&self, nav: f64, total_maintenance_margin: f64) -> f64 {
        if nav < 1e-12 { return 0.0; }
        let threshold = self.liquidation_ratio * total_maintenance_margin;
        (nav - threshold) / nav
    }

    /// Margin ratio: NAV / total_maintenance_margin.
    /// >1 = safe, <1 = liquidated. Higher = more buffer.
    pub fn margin_ratio(&self, nav: f64, total_maintenance_margin: f64) -> f64 {
        if total_maintenance_margin < 1e-12 {
            return f64::INFINITY;
        }
        nav / total_maintenance_margin
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hl_config() -> ExchangeConfig {
        ExchangeConfig::default()
    }

    // -----------------------------------------------------------------------
    // Defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_defaults_are_hyperliquid() {
        let cfg = hl_config();
        assert_eq!(cfg.name, "hyperliquid");
        assert_eq!(cfg.fee_tiers.len(), 7);
        // Tier 0: 4.5 taker, 1.5 maker
        assert!((cfg.fee_tiers[0].taker_bps - 4.5).abs() < 1e-10);
        assert!((cfg.fee_tiers[0].maker_bps - 1.5).abs() < 1e-10);
        assert_eq!(cfg.funding_interval_hours, 1);
        assert!((cfg.maintenance_margin_rate - 0.05).abs() < 1e-10);
        assert_eq!(cfg.funding_cap_per_interval, None);
        assert_eq!(cfg.cost_basis_method, CostBasisMethod::WeightedAverage);
    }

    // -----------------------------------------------------------------------
    // Fee tier lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier0_low_volume() {
        let cfg = hl_config();
        // $1K trailing volume → Tier 0
        let (taker, maker) = cfg.fee_bps_for_volume(1_000.0);
        assert!((taker - 4.5).abs() < 1e-10);
        assert!((maker - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_tier1_5m_volume() {
        let cfg = hl_config();
        // $5M trailing volume → Tier 1
        let (taker, maker) = cfg.fee_bps_for_volume(5_000_000.0);
        assert!((taker - 4.0).abs() < 1e-10);
        assert!((maker - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_tier2_25m_volume() {
        let cfg = hl_config();
        let (taker, _) = cfg.fee_bps_for_volume(25_000_000.0);
        assert!((taker - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_tier6_max_volume() {
        let cfg = hl_config();
        let (taker, maker) = cfg.fee_bps_for_volume(10_000_000_000.0);
        assert!((taker - 2.4).abs() < 1e-10);
        assert!((maker - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tier_boundary_exact() {
        let cfg = hl_config();
        // Exactly at $100M → Tier 3
        let (taker, _) = cfg.fee_bps_for_volume(100_000_000.0);
        assert!((taker - 3.0).abs() < 1e-10);
        // Just below → Tier 2
        let (taker, _) = cfg.fee_bps_for_volume(99_999_999.0);
        assert!((taker - 3.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Fee computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_taker_fee_tier0() {
        let cfg = hl_config();
        // $100K taker at Tier 0 (4.5 bps) = $45
        let fee = cfg.compute_fee(100_000.0, false, 0.0);
        assert!((fee - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_taker_fee_tier2() {
        let cfg = hl_config();
        // $100K taker at Tier 2 ($25M volume, 3.5 bps) = $35
        let fee = cfg.compute_fee(100_000.0, false, 30_000_000.0);
        assert!((fee - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_maker_fee_tier0() {
        let cfg = hl_config();
        // $100K maker at Tier 0 (1.5 bps) = $15
        let fee = cfg.compute_fee(100_000.0, true, 0.0);
        assert!((fee - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_fee_uses_absolute_notional() {
        let cfg = hl_config();
        let fee_pos = cfg.compute_fee(50_000.0, false, 0.0);
        let fee_neg = cfg.compute_fee(-50_000.0, false, 0.0);
        assert!((fee_pos - fee_neg).abs() < 1e-10);
    }

    #[test]
    fn test_zero_notional_zero_fee() {
        let cfg = hl_config();
        assert!((cfg.compute_fee(0.0, false, 0.0)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Volume tracker
    // -----------------------------------------------------------------------

    #[test]
    fn test_volume_tracker_single_day() {
        let mut vt = VolumeTracker::new();
        vt.record_trade(86400, 1_000.0); // day 1
        vt.record_trade(86400, 2_000.0); // same day
        assert!((vt.trailing_volume() - 3_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_tracker_multi_day() {
        let mut vt = VolumeTracker::new();
        vt.record_trade(86400 * 1, 1_000.0);
        vt.record_trade(86400 * 2, 2_000.0);
        vt.record_trade(86400 * 3, 3_000.0);
        assert!((vt.trailing_volume() - 6_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_tracker_expires_old() {
        let mut vt = VolumeTracker::new();
        // Day 1: $1000
        vt.record_trade(86400 * 1, 1_000.0);
        // Day 15: $500 — day 1 is now 14 days ago, should be expired
        vt.record_trade(86400 * 16, 500.0);
        // Only day 16's volume should remain
        assert!((vt.trailing_volume() - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_tracker_14_day_boundary() {
        let mut vt = VolumeTracker::new();
        // Day 1: $1000
        vt.record_trade(86400, 1_000.0);
        // Day 14: $500 — day 1 is still within window (14 days ago = not expired)
        vt.record_trade(86400 * 14, 500.0);
        // Wait, let me check: window_seconds = 14 * 86400
        // cutoff = day14_start - 14*86400 = 86400*14 - 86400*14 = 0
        // day 1 start = 86400, which is >= 0, so NOT expired
        assert!((vt.trailing_volume() - 1500.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Funding computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_funding_long_pays_positive_rate() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, 0.0001);
        assert!((cf - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_funding_short_receives_positive_rate() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(-1.0, 50_000.0, 0.0001);
        assert!((cf - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_funding_long_receives_negative_rate() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, -0.0001);
        assert!((cf - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_funding_short_pays_negative_rate() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(-1.0, 50_000.0, -0.0001);
        assert!((cf - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_funding_cap_applied() {
        let mut cfg = hl_config();
        cfg.funding_cap_per_interval = Some(0.01);
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, 0.05);
        assert!((cf - (-500.0)).abs() < 1e-10);
    }

    #[test]
    fn test_funding_cap_negative_rate() {
        let mut cfg = hl_config();
        cfg.funding_cap_per_interval = Some(0.01);
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, -0.05);
        assert!((cf - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_funding_no_cap_default() {
        let cfg = hl_config();
        assert!(cfg.funding_cap_per_interval.is_none());
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, 0.10);
        assert!((cf - (-5000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_funding_flat_position() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(0.0, 50_000.0, 0.0001);
        assert!((cf).abs() < 1e-10);
    }

    #[test]
    fn test_funding_zero_rate() {
        let cfg = hl_config();
        let cf = cfg.compute_funding_payment(1.0, 50_000.0, 0.0);
        assert!((cf).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Margin
    // -----------------------------------------------------------------------

    #[test]
    fn test_maintenance_margin() {
        let cfg = hl_config();
        let mm = cfg.maintenance_margin(100_000.0);
        assert!((mm - 5_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_maintenance_margin_negative_notional() {
        let cfg = hl_config();
        let mm = cfg.maintenance_margin(-100_000.0);
        assert!((mm - 5_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_not_liquidatable() {
        let cfg = hl_config();
        // NAV $10K > maintenance $5K → safe
        assert!(!cfg.is_liquidatable(10_000.0, 5_000.0));
    }

    #[test]
    fn test_above_maintenance_safe() {
        let cfg = hl_config();
        // NAV $6K > maintenance $5K → safe
        assert!(!cfg.is_liquidatable(6_000.0, 5_000.0));
    }

    #[test]
    fn test_liquidatable() {
        let cfg = hl_config();
        // NAV $4K < maintenance $5K → liquidated
        assert!(cfg.is_liquidatable(4_000.0, 5_000.0));
    }

    #[test]
    fn test_margin_stressed() {
        let cfg = hl_config();
        // halt_ratio = 1.5. NAV $6K, maintenance $5K → ratio 1.2 < 1.5 → stressed
        assert!(cfg.is_margin_stressed(6_000.0, 5_000.0));
        // NAV $8K → ratio 1.6 > 1.5 → not stressed
        assert!(!cfg.is_margin_stressed(8_000.0, 5_000.0));
    }

    #[test]
    fn test_liquidation_distance() {
        let cfg = hl_config();
        // HL default: liquidation_ratio=1.0
        // NAV $10K, maintenance $5K, threshold = 1.0 × $5K
        // dist = (10K - 5K) / 10K = 0.5
        let dist = cfg.liquidation_distance(10_000.0, 5_000.0);
        assert!((dist - 0.5).abs() < 0.01);
        // At threshold: dist = 0
        let dist_at = cfg.liquidation_distance(5_000.0, 5_000.0);
        assert!(dist_at.abs() < 0.01);
    }

    #[test]
    fn test_configurable_liquidation_ratio() {
        let mut cfg = hl_config();
        cfg.liquidation_ratio = 0.5; // hypothetical exchange: liq at 50% maintenance
        // NAV $3K, maintenance $5K → threshold = 0.5 × $5K = $2.5K → NOT liquidated
        assert!(!cfg.is_liquidatable(3_000.0, 5_000.0));
        // NAV $2K < $2.5K → liquidated
        assert!(cfg.is_liquidatable(2_000.0, 5_000.0));
    }

    #[test]
    fn test_margin_ratio() {
        let cfg = hl_config();
        let ratio = cfg.margin_ratio(10_000.0, 5_000.0);
        assert!((ratio - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_margin_ratio_zero_margin() {
        let cfg = hl_config();
        let ratio = cfg.margin_ratio(10_000.0, 0.0);
        assert!(ratio.is_infinite());
    }

    // -----------------------------------------------------------------------
    // Serde deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_with_defaults() {
        let yaml = "name: binance\n";
        let cfg: ExchangeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.name, "binance");
        // Gets HL default fee tiers
        assert_eq!(cfg.fee_tiers.len(), 7);
    }

    #[test]
    fn test_deserialize_custom_tiers() {
        let yaml = r#"
name: binance
fee_tiers:
  - volume_threshold: 0
    taker_bps: 5.0
    maker_bps: 2.0
  - volume_threshold: 10000000
    taker_bps: 4.0
    maker_bps: 1.0
funding_interval_hours: 8
maintenance_margin_rate: 0.04
"#;
        let cfg: ExchangeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.name, "binance");
        assert_eq!(cfg.fee_tiers.len(), 2);
        assert_eq!(cfg.funding_interval_hours, 8);
        assert!((cfg.fee_tiers[0].taker_bps - 5.0).abs() < 1e-10);
        assert!((cfg.fee_tiers[1].taker_bps - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_deserialize_empty_uses_all_defaults() {
        let yaml = "{}";
        let cfg: ExchangeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.name, "hyperliquid");
        assert_eq!(cfg.fee_tiers.len(), 7);
    }

    // -----------------------------------------------------------------------
    // Integration: volume tracker + fee lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_volume_drives_tier() {
        let cfg = hl_config();
        let mut vt = VolumeTracker::new();

        // Initially Tier 0
        let fee = cfg.compute_fee(100_000.0, false, vt.trailing_volume());
        assert!((fee - 45.0).abs() < 1e-10); // 4.5 bps

        // Trade enough to reach Tier 1 (>$5M)
        vt.record_trade(86400, 6_000_000.0);
        let fee = cfg.compute_fee(100_000.0, false, vt.trailing_volume());
        assert!((fee - 40.0).abs() < 1e-10); // 4.0 bps

        // Trade enough to reach Tier 2 (>$25M)
        vt.record_trade(86400, 20_000_000.0);
        let fee = cfg.compute_fee(100_000.0, false, vt.trailing_volume());
        assert!((fee - 35.0).abs() < 1e-10); // 3.5 bps
    }
}
