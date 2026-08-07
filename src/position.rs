//! Position state and portfolio book for holdings-based simulation.
//!
//! Production quant systems track holdings in dollar terms, not weights.
//! Weights are derived: w = h / NAV. The optimizer receives actual weights
//! (marked-to-market), not target weights from last period.
//!
//! References:
//!   - Boyd et al. (2017) "Multi-Period Trading via Convex Optimization" — cvxportfolio
//!   - Gârleanu & Pedersen (2013) — aim portfolio framework
//!   - Hyperliquid docs — weighted average cost basis, cross-margin

use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// PositionState
// ---------------------------------------------------------------------------

/// Per-symbol position in a perpetual futures portfolio.
///
/// Uses weighted average cost basis (what HL and all major exchanges use).
/// Tracks PnL decomposition: trading + funding + commission.
#[derive(Debug, Clone)]
pub struct PositionState {
    pub symbol: String,
    /// Signed quantity: positive = long, negative = short.
    pub quantity: f64,
    /// Weighted average entry price.
    pub avg_entry_price: f64,
    /// Current mark price (updated each bar).
    pub mark_price: f64,
    /// Unrealized PnL: qty × (mark - entry).
    pub unrealized_pnl: f64,
    /// Accumulated realized PnL from closed trades.
    pub realized_trading_pnl: f64,
    /// Accumulated funding payments (positive = received).
    pub realized_funding_pnl: f64,
    /// Accumulated trading commissions.
    pub total_commission: f64,
    /// Epoch seconds when position was first opened.
    pub opened_at: i64,
    /// Epoch seconds of last fill.
    pub last_fill_at: i64,
}

impl PositionState {
    /// Create a new empty position for a symbol.
    pub fn new(symbol: String, timestamp: i64) -> Self {
        Self {
            symbol,
            quantity: 0.0,
            avg_entry_price: 0.0,
            mark_price: 0.0,
            unrealized_pnl: 0.0,
            realized_trading_pnl: 0.0,
            realized_funding_pnl: 0.0,
            total_commission: 0.0,
            opened_at: timestamp,
            last_fill_at: timestamp,
        }
    }

    /// Position side: 1 (long), -1 (short), 0 (flat).
    pub fn side(&self) -> i8 {
        if self.quantity > 1e-12 { 1 }
        else if self.quantity < -1e-12 { -1 }
        else { 0 }
    }

    /// Absolute notional value: |qty| × mark_price.
    pub fn notional(&self) -> f64 {
        self.quantity.abs() * self.mark_price
    }

    /// Signed dollar value: qty × mark_price.
    pub fn dollar_value(&self) -> f64 {
        self.quantity * self.mark_price
    }

    /// Total PnL (realized + unrealized + funding - commission).
    pub fn total_pnl(&self) -> f64 {
        self.unrealized_pnl + self.realized_trading_pnl
            + self.realized_funding_pnl - self.total_commission
    }

    /// Is this position flat (no holdings)?
    pub fn is_flat(&self) -> bool {
        self.quantity.abs() < 1e-12
    }

    /// Update mark price and recompute unrealized PnL.
    pub fn update_mark(&mut self, price: f64) {
        self.mark_price = price;
        if !self.is_flat() {
            self.unrealized_pnl = self.quantity * (price - self.avg_entry_price);
        } else {
            self.unrealized_pnl = 0.0;
        }
    }

    /// Apply a fill. Weighted average cost on opens, realize PnL on closes.
    ///
    /// `fill_qty`: signed quantity (positive=buy, negative=sell).
    /// `fill_price`: execution price.
    /// `commission`: fee for this fill (always positive).
    ///
    /// Returns the realized PnL from this fill (0 for opens, nonzero for closes/flips).
    /// The caller should add this to `book.cash` to settle the realized PnL.
    pub fn apply_fill(&mut self, fill_qty: f64, fill_price: f64, commission: f64, timestamp: i64) -> f64 {
        self.total_commission += commission;
        self.last_fill_at = timestamp;

        if self.is_flat() {
            // New position
            self.quantity = fill_qty;
            self.avg_entry_price = fill_price;
            self.mark_price = fill_price;
            self.unrealized_pnl = 0.0;
            self.opened_at = timestamp;
            self.update_mark(fill_price);
            return 0.0;
        }

        let same_direction = (self.quantity > 0.0 && fill_qty > 0.0)
            || (self.quantity < 0.0 && fill_qty < 0.0);

        let realized_pnl;

        if same_direction {
            // Adding to position: weighted average entry price
            let total_qty = self.quantity + fill_qty;
            self.avg_entry_price = (self.avg_entry_price * self.quantity.abs()
                + fill_price * fill_qty.abs())
                / total_qty.abs();
            self.quantity = total_qty;
            realized_pnl = 0.0;
        } else {
            // Reducing or flipping position
            let close_qty = fill_qty.abs().min(self.quantity.abs());
            let side = if self.quantity > 0.0 { 1.0 } else { -1.0 };

            // Realize PnL on closed portion
            realized_pnl = side * (fill_price - self.avg_entry_price) * close_qty;
            self.realized_trading_pnl += realized_pnl;

            let remaining = self.quantity.abs() - close_qty;
            if fill_qty.abs() > self.quantity.abs() {
                // Flip: close old position, open new in opposite direction
                let flip_qty = fill_qty.abs() - self.quantity.abs();
                self.quantity = -side * flip_qty;
                self.avg_entry_price = fill_price;
                self.opened_at = timestamp;
            } else {
                // Partial or full close
                self.quantity = side * remaining;
                // Entry price unchanged on partial close (weighted avg stays same)
            }
        }

        // Update mark and unrealized PnL
        self.update_mark(fill_price);
        realized_pnl
    }

    /// Accrue funding payment. Returns the cash flow (positive = received by holder).
    ///
    /// Convention: positive funding_rate means longs pay shorts.
    /// `oracle_price`: price used for funding computation (on HL, this is oracle not mark).
    pub fn apply_funding(&mut self, funding_rate: f64, oracle_price: f64) -> f64 {
        if self.is_flat() {
            return 0.0;
        }
        // Payment amount (unsigned)
        let payment = self.quantity.abs() * oracle_price * funding_rate;
        // Cash flow: longs pay when rate > 0 (negative cash flow for longs)
        let cash_flow = if self.quantity > 0.0 {
            -payment // long pays
        } else {
            payment  // short receives
        };
        self.realized_funding_pnl += cash_flow;
        cash_flow
    }
}

// ---------------------------------------------------------------------------
// PortfolioBook
// ---------------------------------------------------------------------------

/// Simulated portfolio: cash + positions.
///
/// NAV = cash + Σ(qty × mark_price). Weights derived from holdings.
#[derive(Debug, Clone)]
pub struct PortfolioBook {
    /// USD cash balance (includes margin, funding flows, commission deductions).
    pub cash: f64,
    /// Per-symbol position state (BTreeMap for deterministic iteration order).
    pub positions: BTreeMap<String, PositionState>,
    /// Initial NAV for reference.
    pub initial_nav: f64,
    /// Lifetime accumulator of `realized_trading_pnl` from positions that
    /// have been purged from the map by `remove_flat_positions`.
    /// Without this, `total_realized_trading_pnl()` would be lossy across
    /// position lifecycles, breaking the A1 bookkeeping invariant.
    /// See `position::tests::diag_remove_flat_destroys_realized_pnl` for
    /// regression coverage.
    pub purged_realized_trading_pnl: f64,
}

impl PortfolioBook {
    /// Create a new portfolio with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            positions: BTreeMap::new(),
            initial_nav: initial_cash,
            purged_realized_trading_pnl: 0.0,
        }
    }

    /// Net asset value for perpetual futures: cash + unrealized PnL.
    ///
    /// Unlike spot, opening a perp position doesn't spend cash — only margin is locked.
    /// NAV = cash_balance + Σ(unrealized_pnl across all positions).
    /// Cash changes from: commissions, funding payments, and realized PnL.
    pub fn nav(&self) -> f64 {
        self.cash + self.positions.values().map(|p| p.unrealized_pnl).sum::<f64>()
    }

    /// Compute actual weights from marked-to-market holdings.
    ///
    /// For perps: weight = (signed_qty × mark_price) / NAV.
    /// This is the effective portfolio exposure per dollar of equity.
    /// Returns empty map if NAV <= 0.
    /// Per-symbol portfolio weights (notional / NAV).
    ///
    /// Returns BTreeMap (not HashMap) so iteration order is sorted by symbol —
    /// required for cross-platform FP determinism. Callers that build
    /// downstream Vec/sum/accumulator structures via `.into_iter()` on this
    /// map depend on the deterministic order.
    /// See docs/plans/cross-platform-divergence.md (Patch 1).
    pub fn actual_weights(&self) -> BTreeMap<String, f64> {
        let nav = self.nav();
        if nav <= 1e-12 {
            return BTreeMap::new();
        }
        self.positions
            .iter()
            .filter(|(_, p)| !p.is_flat())
            .map(|(sym, p)| (sym.clone(), p.quantity * p.mark_price / nav))
            .collect()
    }

    /// Gross exposure: sum of absolute notional values.
    pub fn gross_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.notional()).sum()
    }

    /// Gross exposure as fraction of NAV.
    pub fn gross_leverage(&self) -> f64 {
        let nav = self.nav();
        if nav <= 1e-12 { return 0.0; }
        self.gross_exposure() / nav
    }

    /// Net exposure: sum of signed dollar values.
    pub fn net_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.dollar_value()).sum()
    }

    /// Number of long positions.
    pub fn n_long(&self) -> usize {
        self.positions.values().filter(|p| p.side() == 1).count()
    }

    /// Number of short positions.
    pub fn n_short(&self) -> usize {
        self.positions.values().filter(|p| p.side() == -1).count()
    }

    /// Update all position mark prices from a price map.
    pub fn mark_all(&mut self, prices: &HashMap<String, f64>) {
        for (sym, pos) in &mut self.positions {
            if let Some(&price) = prices.get(sym) {
                pos.update_mark(price);
            }
        }
    }

    /// Get or create a position for a symbol.
    pub fn get_or_create_position(&mut self, symbol: &str, timestamp: i64) -> &mut PositionState {
        self.positions
            .entry(symbol.to_string())
            .or_insert_with(|| PositionState::new(symbol.to_string(), timestamp))
    }

    /// Remove flat positions (cleanup). Accumulates their realized_trading_pnl
    /// into `purged_realized_trading_pnl` BEFORE dropping so the book's
    /// lifetime PnL accumulator survives position lifecycle.
    pub fn remove_flat_positions(&mut self) {
        for pos in self.positions.values() {
            if pos.is_flat() {
                self.purged_realized_trading_pnl += pos.realized_trading_pnl;
            }
        }
        self.positions.retain(|_, p| !p.is_flat());
    }

    /// Total unrealized PnL across all positions.
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    /// Total realized trading PnL across all positions, including positions
    /// that have been purged from the map. The `purged_realized_trading_pnl`
    /// term ensures this is a true lifetime accumulator — required by the
    /// A1 bookkeeping discrepancy check in `simulation::run`.
    pub fn total_realized_trading_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.realized_trading_pnl).sum::<f64>()
            + self.purged_realized_trading_pnl
    }

    /// Total realized funding PnL across all positions.
    pub fn total_realized_funding_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.realized_funding_pnl).sum()
    }

    /// Total commissions paid.
    pub fn total_commission(&self) -> f64 {
        self.positions.values().map(|p| p.total_commission).sum()
    }

    /// Total maintenance margin requirement (cross-margin: sum across all positions).
    pub fn total_maintenance_margin(&self, margin_rate: f64) -> f64 {
        self.positions.values().map(|p| margin_rate * p.notional()).sum()
    }

    /// Margin ratio: NAV / total_maintenance.
    /// Returns infinity if no positions.
    pub fn margin_ratio(&self, margin_rate: f64) -> f64 {
        let mm = self.total_maintenance_margin(margin_rate);
        if mm < 1e-12 { return f64::INFINITY; }
        self.nav() / mm
    }

    /// Available margin: NAV - total_maintenance.
    pub fn available_margin(&self, margin_rate: f64) -> f64 {
        self.nav() - self.total_maintenance_margin(margin_rate)
    }

    /// Close all positions at their current mark price (liquidation).
    /// Returns total realized PnL and total commission.
    pub fn liquidate_all(&mut self, timestamp: i64, fee_rate: f64) -> (f64, f64) {
        let mut total_rpnl = 0.0f64;
        let mut total_commission = 0.0f64;
        let syms: Vec<String> = self.positions.keys().cloned().collect();
        for sym in syms {
            let pos = self.positions.get_mut(&sym).unwrap();
            if pos.is_flat() { continue; }
            let price = pos.mark_price;
            let qty = -pos.quantity;
            let notional = (qty * price).abs();
            let commission = fee_rate * notional;
            let rpnl = pos.apply_fill(qty, price, commission, timestamp);
            self.cash += rpnl;
            self.cash -= commission;
            total_rpnl += rpnl;
            total_commission += commission;
        }
        self.remove_flat_positions();
        (total_rpnl, total_commission)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position_flat() {
        let pos = PositionState::new("BTC".into(), 100);
        assert!(pos.is_flat());
        assert_eq!(pos.side(), 0);
        assert_eq!(pos.notional(), 0.0);
    }

    #[test]
    fn test_open_long() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(1.0, 50000.0, 5.0, 100); // buy 1 BTC at $50K

        assert_eq!(pos.side(), 1);
        assert!((pos.quantity - 1.0).abs() < 1e-10);
        assert!((pos.avg_entry_price - 50000.0).abs() < 1e-10);
        assert!((pos.total_commission - 5.0).abs() < 1e-10);
        assert!((pos.notional() - 50000.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_to_long_weighted_avg() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(1.0, 50000.0, 0.0, 100);
        pos.apply_fill(1.0, 60000.0, 0.0, 200);

        assert!((pos.quantity - 2.0).abs() < 1e-10);
        // Weighted avg: (50000 * 1 + 60000 * 1) / 2 = 55000
        assert!((pos.avg_entry_price - 55000.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_close_long() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(2.0, 50000.0, 0.0, 100); // buy 2 BTC at $50K
        pos.apply_fill(-1.0, 60000.0, 0.0, 200); // sell 1 BTC at $60K

        assert!((pos.quantity - 1.0).abs() < 1e-10);
        // Realized PnL: 1 * (60000 - 50000) = 10000
        assert!((pos.realized_trading_pnl - 10000.0).abs() < 1e-10);
        // Entry price unchanged on partial close
        assert!((pos.avg_entry_price - 50000.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_close() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(1.0, 50000.0, 0.0, 100);
        pos.apply_fill(-1.0, 55000.0, 0.0, 200);

        assert!(pos.is_flat());
        assert!((pos.realized_trading_pnl - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_flip_long_to_short() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(1.0, 50000.0, 0.0, 100);  // long 1 BTC at $50K
        pos.apply_fill(-3.0, 60000.0, 0.0, 200);  // sell 3 BTC at $60K → short 2

        assert_eq!(pos.side(), -1);
        assert!((pos.quantity - (-2.0)).abs() < 1e-10);
        // Realized: 1 * (60000 - 50000) = 10000 (closing the long)
        assert!((pos.realized_trading_pnl - 10000.0).abs() < 1e-10);
        // New entry price is the flip price
        assert!((pos.avg_entry_price - 60000.0).abs() < 1e-10);
    }

    #[test]
    fn test_short_position_pnl() {
        let mut pos = PositionState::new("ETH".into(), 100);
        pos.apply_fill(-10.0, 3000.0, 0.0, 100); // short 10 ETH at $3K
        pos.update_mark(2500.0); // price drops

        // Short profits when price drops: -10 * (2500 - 3000) = 5000
        assert!((pos.unrealized_pnl - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_funding_long_pays() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(1.0, 50000.0, 0.0, 100);

        // Positive rate → longs pay
        let cash_flow = pos.apply_funding(0.0001, 50000.0);
        // Payment = 1.0 * 50000 * 0.0001 = 5.0
        // Long pays → cash_flow = -5.0
        assert!((cash_flow - (-5.0)).abs() < 1e-10);
        assert!((pos.realized_funding_pnl - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_funding_short_receives() {
        let mut pos = PositionState::new("BTC".into(), 100);
        pos.apply_fill(-1.0, 50000.0, 0.0, 100);

        // Positive rate → shorts receive
        let cash_flow = pos.apply_funding(0.0001, 50000.0);
        assert!((cash_flow - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_book_nav_perps() {
        let mut book = PortfolioBook::new(100_000.0);

        // Open long 1 BTC at $50K (perp: no cash spent, just margin)
        let pos = book.get_or_create_position("BTC", 100);
        let rpnl = pos.apply_fill(1.0, 50000.0, 10.0, 100);
        book.cash -= 10.0; // commission
        book.cash += rpnl;  // realized PnL (0 for open)
        assert!((rpnl).abs() < 1e-10);

        // At entry, unrealized PnL = 0, so NAV = cash = 99990
        assert!((book.nav() - 99990.0).abs() < 1e-10);

        // BTC goes to $60K
        let mut prices = HashMap::new();
        prices.insert("BTC".into(), 60000.0);
        book.mark_all(&prices);

        // Unrealized PnL = 1 * (60000 - 50000) = 10000
        // NAV = 99990 + 10000 = 109990
        assert!((book.nav() - 109990.0).abs() < 1e-10);

        // Close position at $60K
        let pos = book.get_or_create_position("BTC", 200);
        let rpnl = pos.apply_fill(-1.0, 60000.0, 10.0, 200);
        book.cash -= 10.0; // commission
        book.cash += rpnl;  // realized PnL

        // Realized PnL = 10000, unrealized = 0, cash = 99990 - 10 + 10000 = 109980
        assert!((rpnl - 10000.0).abs() < 1e-10);
        assert!((book.nav() - 109980.0).abs() < 1e-10);
    }

    /// Empirical test for the A1 bookkeeping discrepancy hypothesis:
    /// remove_flat_positions() destroys accumulated realized_trading_pnl
    /// from positions that have been closed. If true, this breaks the
    /// PortfolioBook::total_realized_trading_pnl() invariant that the
    /// A1 fix relies on (delta-realized between day boundaries should
    /// equal sum of rpnl from apply_fill calls during the day).
    #[test]
    fn diag_remove_flat_destroys_realized_pnl() {
        let mut book = PortfolioBook::new(1000.0);

        // Open + close 1 BTC at +$100 profit → realized $100
        let pos = book.get_or_create_position("BTC", 100);
        let _ = pos.apply_fill(1.0, 50000.0, 0.0, 100);
        let rpnl = pos.apply_fill(-1.0, 50100.0, 0.0, 200);
        assert!((rpnl - 100.0).abs() < 1e-10, "rpnl on close = $100");

        // Position is flat but still in map; total_realized = $100
        let total_before_purge = book.total_realized_trading_pnl();
        assert!(
            (total_before_purge - 100.0).abs() < 1e-10,
            "before purge: total_realized = $100, got ${total_before_purge}"
        );

        // Open + close another at -$50 → cumulative realized should be $50
        let pos = book.get_or_create_position("ETH", 300);
        let _ = pos.apply_fill(1.0, 3000.0, 0.0, 300);
        let rpnl = pos.apply_fill(-1.0, 2950.0, 0.0, 400);
        assert!((rpnl - (-50.0)).abs() < 1e-10);

        // Both positions flat, BTC + ETH cumulative = $100 - $50 = $50
        let total_with_flats = book.total_realized_trading_pnl();
        assert!(
            (total_with_flats - 50.0).abs() < 1e-10,
            "with flats in map: total_realized = $50, got ${total_with_flats}"
        );

        // Now purge flat positions (this is what liquidate_all + the
        // periodic cleanup call). HYPOTHESIS: this destroys $50 of realized.
        book.remove_flat_positions();
        let total_after_purge = book.total_realized_trading_pnl();

        // INVARIANT we want: total_realized must SURVIVE position removal.
        // If this assertion FAILS, the A1 bookkeeping discrepancy is
        // confirmed to be caused by remove_flat_positions wiping rpnl.
        assert!(
            (total_after_purge - 50.0).abs() < 1e-10,
            "BUG CONFIRMED: total_realized_trading_pnl() returned ${total_after_purge} \
             after remove_flat_positions(), but should be $50. Lost ${} of realized PnL.",
            50.0 - total_after_purge
        );
    }

    #[test]
    fn test_actual_weights_with_drift() {
        let mut book = PortfolioBook::new(100_000.0);

        // Open positions at equal weights
        let btc = book.get_or_create_position("BTC", 100);
        btc.apply_fill(1.0, 50000.0, 0.0, 100);

        let eth = book.get_or_create_position("ETH", 100);
        eth.apply_fill(-10.0, 3000.0, 0.0, 100);

        // Mark at entry prices → unrealized PnL = 0
        let mut prices = HashMap::new();
        prices.insert("BTC".into(), 50000.0);
        prices.insert("ETH".into(), 3000.0);
        book.mark_all(&prices);

        // Now BTC doubles
        prices.insert("BTC".into(), 100000.0);
        book.mark_all(&prices);

        // BTC unrealized = 1 * (100000 - 50000) = 50000
        // ETH unrealized = -10 * (3000 - 3000) = 0
        // NAV = 100000 + 50000 + 0 = 150000 (for perps: cash + unrealized)
        let perp_nav = book.cash + book.total_unrealized_pnl();
        assert!((perp_nav - 150000.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_flat_positions() {
        let mut book = PortfolioBook::new(100_000.0);

        let pos = book.get_or_create_position("BTC", 100);
        pos.apply_fill(1.0, 50000.0, 0.0, 100);
        pos.apply_fill(-1.0, 55000.0, 0.0, 200); // fully closed

        assert_eq!(book.positions.len(), 1);
        book.remove_flat_positions();
        assert_eq!(book.positions.len(), 0);
    }
}
