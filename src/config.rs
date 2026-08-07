use serde::Deserialize;

// ---------------------------------------------------------------------------
// Cost model configuration
// ---------------------------------------------------------------------------
//
// Phase 3 cleanup (2026-05-18): the `CostSource` enum was removed. It was
// only ever populated by spec deserialization and never read at runtime —
// cost data flows through the unified execution_h1.parquet's baked columns
// (spread_bps, kappa), not through a CostSource variant. Existing spec yamls
// still carry a decorative `cost_model.source:` block; serde silently ignores
// unknown fields, so they continue to load. Phase 1's `expected_sha256` +
// `scores_parquet` fields are the actual artifact-lineage source of truth.

/// Cost model configuration.
///
/// Taker fee comes from `ExchangeConfig.fee_tiers` (volume-adaptive).
/// This config holds spread/impact knobs and the cost-scores artifact pin.
/// EWMA halflives (in days) for sigma and ADV used in the impact-cost
/// formula `kappa·sigma·Q^(1+δ) / ADV^δ`, where δ is `simulation::IMPACT_DELTA`.
#[derive(Debug, Clone, Deserialize)]
pub struct CostModelConfig {
    /// Multiplier on spread component of c_lin (Optuna-tunable).
    #[serde(default = "default_c_lin_multiplier")]
    pub c_lin_multiplier: f64,
    /// EWMA halflife in DAYS for realized volatility (used in cost formula).
    /// Calibration target: tune against realised fill data once available.
    #[serde(default = "default_sigma_halflife_days")]
    pub sigma_halflife_days: usize,
    /// EWMA halflife in DAYS for average daily volume (used in cost formula).
    /// Calibration target: tune against realised fill data once available.
    #[serde(default = "default_adv_halflife_days")]
    pub adv_halflife_days: usize,
    /// Power-law exponent δ in impact formula `κ·σ·|Q|^(1+δ)/ADV^δ`.
    /// Fitted from cell oracle (Phase 2 of A2 fix). Replaces the legacy
    /// `simulation::IMPACT_DELTA` constant. See docs/plans/cost-model-A2-fix.md.
    /// Range: [0.1, 0.7] per literature (Tóth-Bouchaud-Bucci).
    #[serde(default = "default_impact_delta")]
    pub impact_delta: f64,
    /// Reference Q (USD) at which the residual-layer impact contribution is
    /// zero. R9 predicts cell-median cost which already includes implicit
    /// impact at the cell's typical Q ≈ Q_ref. Charging additional impact
    /// only above Q_ref preserves R9's calibration regime.
    /// Default = 0 means "do not subtract reference" (legacy behavior).
    #[serde(default = "default_q_ref_usd")]
    pub q_ref_usd: f64,
    /// A4 fix: tradeability gate, expressed as bps of |notional|. When
    /// predicted impact for a trade exceeds this threshold, the simulator
    /// skips the fill entirely (no cash debit, no position change). This is
    /// a structural prevention rather than a cost cap — we don't charge a
    /// clipped cost on a trade that happened, we refuse to execute the
    /// trade at all. Maps to real-market behavior: orders that would move
    /// price beyond the strategy's threshold simply don't fill.
    /// Empirical justification: P99.9 of realized impact in the production
    /// trade tape is 1.27 bps; gate at 100 bps deems trades non-tradeable
    /// only when their predicted impact is > 79× the natural P99.9.
    /// Default = 0 means "no gate" (all trades fill).
    #[serde(default = "default_impact_gate_bps")]
    pub impact_gate_bps: f64,
    /// A4 Layer 2 fallback: DLOM ("Discount for Lack of Marketability")
    /// applied as a flat fraction of |notional| when computing the
    /// liquidation haircut for a position whose σ/κ/ADV lookups are
    /// missing AND the last-known fallback is unavailable or too stale.
    /// Range typically 0.10-0.30 from restricted-stock studies (Stout 2013;
    /// SEC Rule 144); 0.50 is a conservative anchor for "we genuinely
    /// cannot price the exit." Per IFRS 13 / ASC 820 Level 3 convention.
    /// Default 0.50 — only fires for stuck-out-of-universe positions.
    #[serde(default = "default_illiquid_dlom_pct")]
    pub illiquid_dlom_pct: f64,
    /// A4 Layer 2: max age (in days) for "last-known" σ/κ/ADV fallback.
    /// When current-bar lookup misses but the symbol has data within this
    /// window, use the most recent value. Beyond this, degrade to DLOM.
    /// Default 90 days — stale enough that liquidity can have changed
    /// materially, but recent enough that real data still beats DLOM.
    #[serde(default = "default_last_known_max_age_days")]
    pub last_known_max_age_days: usize,
    /// Volume column unit semantics for ohlcv_filtered.parquet inputs.
    ///
    /// `false` (default, LEGACY): treats `volume` column as coin units and
    /// computes `dollar_volume = volume × close` in `extract_d1_data`. Matches
    /// historical buggy convention; the bug is consistent across cost_v2
    /// trainer + engine, so κ-cancellation preserves prediction correctness.
    /// Used by prod_v4 + prod_v5 (legacy v3/v5 scores trained with same bug).
    ///
    /// `true` (CORRECTED): treats `volume` column as USD directly (matches
    /// actual CG-sourced data per `coingecko/client.rs` vs_currency=usd).
    /// Used by prod_v6+ specs trained against cost_v2 with same flag set.
    /// κ values become physically meaningful.
    ///
    /// MUST match the flag used when training the spec's cost-scores parquet.
    /// Mixing flag values across spec + scores artifact silently mispredicts
    /// by a per-asset `close^δ` factor.
    #[serde(default = "default_volume_is_usd")]
    pub volume_is_usd: bool,

    /// Fallback spread (bps) for (timestamp, symbol) pairs missing from the
    /// cost-scores artifact. Used by the per-bar `c_lin` build when a row
    /// has no spread prediction. Default 5.0 bps. Set wider (e.g. 20–50) for
    /// universes with thin coverage; set tighter (0–2) when running a fully
    /// scored universe and you'd rather refuse than impute.
    #[serde(default = "default_spread_bps_fallback")]
    pub default_spread_bps: f64,

    /// Calibration multiplier applied to predicted spread_bps at parquet
    /// load time. Used to bias the cost model conservatively above realized
    /// cost (e.g., 1.20 = +20% safety margin) per L2 book-walk validation.
    /// Spec-pinned, NOT tunable — distinct from `c_lin_multiplier` (which is
    /// the QP optimizer's knob). Default 1.0 = no adjustment.
    #[serde(default = "default_calibration_multiplier")]
    pub spread_bps_calibration_multiplier: f64,

    /// Calibration multiplier applied to predicted kappa at parquet load
    /// time. Used to bias liquidation haircut + QP impact-cost conservative
    /// above realized impact. Spec-pinned, NOT tunable. Default 1.0.
    #[serde(default = "default_calibration_multiplier")]
    pub kappa_calibration_multiplier: f64,

    // -----------------------------------------------------------------------
    // Artifact lineage (2026-05-18, Phase 1 of cost-consolidation plan).
    //
    // These three fields pin a spec to a specific cost-scores parquet so any
    // run validates the inputs match what the spec was tuned against.
    // All Option because legacy specs lack the metadata; when None, no check
    // is performed (current behavior).
    //
    // See `train/scripts/build_cost_scores.py` for the producer side: the
    // SHA-256 here is over the BYTES of the scores parquet file (not the
    // input cost.parquet); use `shasum -a 256 PATH` to recover.
    // -----------------------------------------------------------------------
    /// Path the spec was tuned against. Used by `execution solve`/`execution
    /// backtest` to look up `expected_sha256` when only a spec is provided.
    /// Informational when a `--cost-scores` CLI override is in play.
    #[serde(default)]
    pub scores_parquet: Option<std::path::PathBuf>,
    /// SHA-256 of the cost-scores parquet this spec was tuned against. When
    /// set, the execution loader hashes the actual file and refuses on
    /// mismatch. Override with `--cost-scores-sha-check skip` (TODO Phase 1d).
    #[serde(default)]
    pub expected_sha256: Option<String>,
}

fn default_c_lin_multiplier() -> f64 { 1.0 }
fn default_sigma_halflife_days() -> usize { 60 }
fn default_adv_halflife_days() -> usize { 60 }
fn default_impact_delta() -> f64 { 0.10 }
fn default_q_ref_usd() -> f64 { 0.0 }
fn default_impact_gate_bps() -> f64 { 0.0 }
fn default_illiquid_dlom_pct() -> f64 { 0.50 }
fn default_last_known_max_age_days() -> usize { 90 }
fn default_volume_is_usd() -> bool { false }
fn default_spread_bps_fallback() -> f64 { 5.0 }
fn default_calibration_multiplier() -> f64 { 1.0 }

impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            c_lin_multiplier: 1.0,
            sigma_halflife_days: 60,
            adv_halflife_days: 60,
            impact_delta: 0.10,
            q_ref_usd: 0.0,
            impact_gate_bps: 0.0,
            illiquid_dlom_pct: 0.50,
            last_known_max_age_days: 90,
            volume_is_usd: false,
            default_spread_bps: 5.0,
            spread_bps_calibration_multiplier: 1.0,
            kappa_calibration_multiplier: 1.0,
            scores_parquet: None,
            expected_sha256: None,
        }
    }
}

// ---------------------------------------------------------------------------
// AlphaUnits — declares what units the alpha_future column is in
// ---------------------------------------------------------------------------

/// Declares the units of the `alpha_future` column in the unified parquet.
///
/// The QP solves `min γw'Σw − α'w + ...` and treats α as **dollar-PnL per
/// unit of weight**. If α is in any other units (vol-normalized, rank, etc.),
/// the QP will systematically misweight the cross-section. To prevent that,
/// the loader applies the appropriate per-target adapter at preload time.
///
/// Reference: train/rumpy_train/targets/excess_volnorm.py — the comment
/// "At execution time, multiply predictions by sigma_i to recover dollar
/// alphas" was the contract; this enum makes it enforceable.
///
/// Calibration source: each alpha target in train/rumpy_train/targets/ should
/// declare its units; the execution config matches that declaration.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AlphaUnits {
    /// Direct dollar-PnL prediction. No adapter applied. Default for
    /// backward-compat with parquets predating the units contract.
    DollarAlpha,
    /// Vol-normalized excess return: prediction = (fwd_ret - mkt_ret) / σ_i.
    /// Loader multiplies by per-asset σ_i (D1 EWMA) at preload.
    /// Used by `excess_volnorm` target.
    VolNormalizedExcess,
}

impl Default for AlphaUnits {
    fn default() -> Self { AlphaUnits::DollarAlpha }
}

// ---------------------------------------------------------------------------
// FillMode — execution timeline
// ---------------------------------------------------------------------------

/// How daily target weights are turned into fills.
///
/// Strategic decisions (alpha, QP, vol target, dynamic gamma) are always daily.
/// `FillMode` only changes the execution-bar timeline:
///   - `Market`: one fill per day at the day's price (D1 execution loop).
///   - `Twap`:  24 hourly fills walking from current weights toward the daily
///             target (H1 execution loop). Reduces impact at large NAV.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FillMode {
    Market,
    Twap,
}

impl Default for FillMode {
    fn default() -> Self { FillMode::Market }
}

// ---------------------------------------------------------------------------
// Phase 2B-2E placeholder configs (filled in later phases)
// ---------------------------------------------------------------------------

/// Walk-forward CV configuration (Phase 2B).
#[derive(Debug, Clone, Deserialize)]
pub struct WalkForwardConfig {
    #[serde(default = "default_n_folds")]
    pub n_folds: usize,
    #[serde(default = "default_warmup_bars")]
    pub warmup_bars: usize,
    #[serde(default = "default_purge_bars")]
    pub purge_bars: usize,
    #[serde(default = "default_stability_penalty")]
    pub stability_penalty: f64,
    #[serde(default = "default_min_fold_bars")]
    pub min_fold_bars: usize,
}

fn default_n_folds() -> usize { 5 }
fn default_warmup_bars() -> usize { 200 }
fn default_purge_bars() -> usize { 50 }
fn default_stability_penalty() -> f64 { 0.5 }
fn default_min_fold_bars() -> usize { 10 }

fn default_per_name_cap_floor_safety_mult() -> f64 { 1.5 }
fn default_per_name_cap_vol_scale_ceiling_mult() -> f64 { 5.0 }
fn default_per_name_cap_fallback() -> f64 { 0.05 }

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            warmup_bars: 200,
            purge_bars: 50,
            stability_penalty: 0.5,
            min_fold_bars: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Core execution configs (existing)
// ---------------------------------------------------------------------------

/// QP solver configuration.
/// Defaults: execution_v10_cdar_l3 champion (250 trials, CDaR λ=3 objective).
#[derive(Debug, Clone, Deserialize)]
pub struct QPConfig {
    pub gamma: f64,
    pub lambda_aim: f64,
    pub l_max: f64,
    pub per_name_cap: Option<f64>,
    pub ridge: f64,
    /// 2026-05-16 Path 1 (vol-scaled per-name cap, Richard-Roncalli 2019).
    /// When true, replace the scalar `per_name_cap` constraint `|w_i| ≤ c`
    /// with a per-asset vector `|w_i| ≤ c × median(σ)/σ_i`. High-vol names
    /// get tighter caps automatically; low-vol names get looser caps. The
    /// scalar `per_name_cap` is reinterpreted as B_max (the cap at the
    /// median-vol asset). The cross-section median σ is computed each bar
    /// from the diagonal of the bar's Σ (factor model output).
    /// Default false. See memory/project_per_name_cap_research_2026_05_16.md.
    #[serde(default)]
    pub per_name_cap_vol_scale: bool,
    /// 2026-05-16 Path 2 (Anis-Kwon 2022 top-K concentration cap via convex
    /// LP reformulation). When Some, enforces `Σ_{i∈top-K} |w_i| ≤ θ` using
    /// the standard sum-of-largest-K LP trick. Adds n+1 auxiliary variables
    /// (τ scalar + n nonneg slacks) and 2n+1 NonnegCone rows to the QP.
    /// Targets crisis-correlation regimes where pairwise correlations spike
    /// and a "diversified" portfolio becomes 1 concentrated bet.
    /// See memory/project_per_name_cap_research_2026_05_16.md.
    #[serde(default)]
    pub topk_cap: Option<TopKCapConfig>,
    /// Probe knob (2026-05-05): if set, restrict the QP to the top-K assets
    /// by |alpha| each bar. Outside-top-K assets get alpha=0 and w_aim=0,
    /// so the only forces on them are the cone movement cost and γ·w'Σw —
    /// existing positions bleed off rather than rotate. None ⇒ disabled.
    #[serde(default)]
    pub prune_top_k: Option<usize>,

    /// Vol-target constraint inside the QP (L1 of leverage refactor 2026-05-13).
    /// When Some(σ), the QP enforces `w'Σw ≤ σ²` as an SOC constraint via
    /// Cholesky factorization: `||L'w|| ≤ σ` where Σ = LL'. When None, no
    /// SOC is added — solver behavior matches pre-refactor exactly.
    /// Units: daily vol (e.g. 0.010 = 1% daily ≈ 16% annualized).
    /// Replaces the post-QP `vol_target.vol_target_daily` rescale.
    #[serde(default)]
    pub sigma_target_daily: Option<f64>,

    /// NAV-aware per-name-cap floor: safety multiplier on
    /// `exchange.min_order_size_usd` when computing the floor
    /// `(min_order_size · safety_mult) / (l_max · NAV)`. At low NAV the
    /// configured per_name_cap × l_max × NAV may fall below the exchange
    /// minimum order — every per-asset opening trade gets silently rejected.
    /// The floor lifts the cap so the smallest position is still tradeable.
    /// 1.5 ensures partial-fill TWAP bars (trade_rate < 1) clear the gate
    /// at the last hour. Default 1.5.
    #[serde(default = "default_per_name_cap_floor_safety_mult")]
    pub per_name_cap_floor_safety_mult: f64,

    /// Per-asset cap ceiling for the vol-scaled per-name-cap path (Path 1,
    /// `per_name_cap_vol_scale=true`): keeps ultra-low-vol names from
    /// receiving unboundedly large caps via `c_base · σ_med / σ_i`.
    /// Final per-asset cap = `min(c_base · σ_med / σ_i, ceiling_mult · c_base)`.
    /// Only fires when `per_name_cap_vol_scale=true`. Default 5.0.
    #[serde(default = "default_per_name_cap_vol_scale_ceiling_mult")]
    pub per_name_cap_vol_scale_ceiling_mult: f64,

    /// Fallback used when `per_name_cap = None` and either
    /// `per_name_cap_vol_scale=true` or `blend_rho > 0` fires (both paths
    /// need a numeric base cap to build per-asset caps from). In production
    /// `per_name_cap` is always Some, so this fallback is dead at the prod
    /// path; keeping it tunable surfaces the value for test fixtures and
    /// future configurations. Default 0.05.
    #[serde(default = "default_per_name_cap_fallback")]
    pub per_name_cap_default: f64,
}

/// Top-K concentration cap config (Anis-Kwon 2022).
#[derive(Debug, Clone, Deserialize)]
pub struct TopKCapConfig {
    /// K — number of largest |w_i| names whose sum is constrained.
    pub k: usize,
    /// θ — upper bound on `Σ_{i∈top-K} |w_i|` (fraction of NAV).
    pub theta: f64,
}

// Cost tiers removed — ML model provides per-asset spread/kappa predictions.
// See ml_cost.rs for the CostProvider resolution chain (L2 > ML > error).

// L2 (2026-05-13): VolTargetConfig + DynamicGammaConfig removed.
// Post-QP vol-target rescale and γ-dispersion overlay both eliminated; the
// QP's SOC constraint (qp.sigma_target_daily) is now the leverage knob and
// γ is a static smoothness regularizer (Boyd cvxportfolio 2024 canonical
// form). See docs/sessions/2026-05-13-pre-phase2-leverage-refactor-plan.md.

/// Covariance estimation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct CovarianceConfig {
    pub n_factors: usize,
    pub ewma_halflife: usize,
    pub min_history: usize,
    pub min_assets: usize,
    pub winsorize_sigma: f64,
    pub ridge: f64,
    /// Probe A (2026-05-09 — Markowitz regime-fragility fix): per-bar shrinkage
    /// of the covariance matrix toward its diagonal:
    ///   Σ_shrunk[i,j] = (1−ρ)·Σ[i,j]   for i ≠ j
    ///   Σ_shrunk[i,i] = Σ[i,i]
    ///
    /// 0.0 (default) = no shrinkage, current production behavior.
    /// 1.0           = pure diagonal (vol-normalized only; off-diagonals zeroed).
    ///
    /// Diagonal target is preferred over identity (Ledoit-Wolf canonical form):
    /// keeps σ_i² on the diagonal so vol-normalization is preserved across
    /// the cross-section. Empirically motivated by the Markowitz regime
    /// fragility — Σ⁻¹·α corrupts mid-rank alpha when off-diagonal cov
    /// estimates are noisy. Memory ref: project_markowitz_fix_research.md.
    ///
    /// LW-optimal closed form deferred (requires access to the raw returns
    /// matrix not currently plumbed through `RollingCovariance.get`). Static
    /// ρ first; LW-adaptive can follow if the static sweep finds a clear
    /// regime-conditional optimum.
    #[serde(default)]
    pub shrinkage_rho: f64,
}

/// Alpha blending configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AlphaConfig {
    pub horizons: Vec<usize>,
    pub winsorize_sigma: f64,
    pub l_target: f64,
    /// Probe G (2026-05-09 — Markowitz regime-fragility fix): per-bar soft
    /// threshold (Donoho-Johnstone 1994) on the cross-section's z-scored
    /// alpha BEFORE the QP. For each asset, with z = (α − mean) / σ:
    ///   z_clean = sign(z) · max(0, |z| − threshold)
    ///   α_clean = mean + z_clean · σ
    ///
    /// 0.0 (default) = no thresholding, current production behavior.
    /// 0.5/1.0/1.5  = increasingly aggressive denoising of mid-rank alpha.
    ///
    /// Mid-rank alpha is what Σ⁻¹ amplifies into wrong-side bets in regime-
    /// shift periods (memory ref: project_markowitz_regime_fragility.md;
    /// the IC-drop in Period B was concentrated in the MIDDLE of the
    /// cross-section, not the tails). Soft-threshold flattens the middle
    /// to mean while preserving the rank structure of the tails.
    /// Memory ref: project_markowitz_fix_research.md (candidate G).
    #[serde(default)]
    pub soft_threshold_z: f64,
    /// 2026-05-16 Path 4 (MOP 2012 vol-scaling): divide per-asset alpha by σ_i
    /// before the QP / rank-vector. Targets risk-parity-style sizing where
    /// high-vol names get LESS weight pressure for the same conviction signal.
    /// Default false. Cheap falsifier: if Markowitz Σ⁻¹·α already captures
    /// this effect via the gamma term, the lever will be near no-op.
    /// See memory/project_per_name_cap_research_2026_05_16.md.
    #[serde(default)]
    pub vol_scale: bool,
}

/// Funding cost model configuration.
///
/// Calibration note (2026-04-27): the EWMA halflife was changed 24 → 5 hours
/// based on empirical fit of `funding_1h.parquet` (229 HL assets, 2023-05 →
/// 2026-04, 3.84M obs). Lag-1 hourly autocorrelation has median ρ=0.849
/// (pooled 0.871), implying empirical halflife ~4-5 hours per
/// `−ln(2)/ln(ρ)`. We picked 5h to match the pooled value.
///
/// **The choice is operationally moot for current dollar-neutral L/S**:
/// sensitivity test (v7 post-fix-cdar12 champion at hl ∈ {5, 24, 72}) showed
/// ΔSharpe = 0.001 across the three. Reasons: 71.7% of funding bars use
/// actual rates (halflife only matters for the 28.3% model-estimated
/// portion); long/short funding nets to ~$0 regardless.
///
/// The prior 24h default and the `research_funding_rate_modeling.md` memory's
/// "halflife=3d" recommendation describe the persistence of *daily* funding
/// means (smoothed across the funding window). Our code's EWMA operates on
/// *raw hourly* observations — the noise floor in hourly rates makes the
/// empirical decay much faster.
///
/// `default_rho = 0.89` is consistent with empirical pooled ρ = 0.871 and
/// not changed here.
#[derive(Debug, Clone, Deserialize)]
pub struct FundingConfig {
    pub enabled: bool,
    /// EWMA halflife in hours for current rate estimate (f₀).
    /// Default 5h, calibrated against real funding history — see struct doc.
    pub ewma_halflife_hours: usize,
    /// Rolling window in hours for long-run mean (μ) estimation.
    pub long_run_mean_window_hours: usize,
    /// Default hourly autocorrelation (ρ). Used when per-asset estimate unavailable.
    pub default_rho: f64,
    /// Expected holding period in hours. Used for AR(1) cumulative cost.
    pub holding_hours: usize,
    /// 2026-05-15: subtract `FundingModel::expected_alpha_adjustment()` from
    /// the alpha vector before the QP solves (Koijen et al. 2018 *Carry*).
    /// Default `false` to preserve historical backtest comparability.
    #[serde(default)]
    pub use_in_alpha: bool,
    /// When `use_in_alpha = true`, cross-sectionally demean the adjustment
    /// before subtraction. Defends against asymmetric-gross leakage: a
    /// non-zero cross-sectional mean tilts gross when L/S notionals are not
    /// exactly balanced (current strategy is ~70–85% L:S).
    #[serde(default = "default_funding_alpha_demean")]
    pub alpha_demean: bool,
    /// 2026-05-15: scalar multiplier applied to the funding adjustment before
    /// subtraction from alpha. Default 1.0 — but at our z-score-blended alpha
    /// magnitudes (~0.5) vs raw 24h funding cost (~1e-4), 1.0 is effectively
    /// a no-op. Tunable surface to bring the carry-tilt into a magnitude that
    /// actually influences QP trade selection. Empirical sweep recommended;
    /// see backtest in /tmp/funding_aware/ (2026-05-15).
    #[serde(default = "default_funding_alpha_scale")]
    pub alpha_scale: f64,
}

fn default_funding_alpha_demean() -> bool { true }
fn default_funding_alpha_scale() -> f64 { 1.0 }

/// Regime-conditional execution (Phase 2 — 2026-05-05).
///
// L2 (2026-05-13): RegimeConfig removed. Regime-conditional per_name_cap
// overlay had the same overfit-stacking pattern as dynamic_gamma; dropped
// per leverage-refactor external research (Lopez de Prado AFML Ch. 11).

/// Top-level execution config (loaded from execution.yaml).
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionConfig {
    pub qp: QPConfig,
    pub covariance: CovarianceConfig,
    pub alpha: AlphaConfig,
    pub funding: FundingConfig,
    #[serde(default)]
    pub cost_model: CostModelConfig,
    /// Exchange-specific parameters (fees, funding, margin).
    #[serde(default)]
    pub exchange: crate::exchange::ExchangeConfig,
    pub nav_usd: f64,
    /// Execution timeline: `Market` (one fill/day) or `Twap` (24 fills/day).
    /// Strategic decisions (alpha, QP, vol target) are always daily.
    #[serde(default)]
    pub fill_mode: FillMode,
    /// Units of `alpha_future` in the unified parquet. The loader applies a
    /// per-target adapter to recover dollar-alpha before feeding to the QP.
    /// Production today uses `excess_volnorm` → set to `vol_normalized_excess`.
    #[serde(default)]
    pub alpha_units: AlphaUnits,
    /// Constant-NAV mode for tuning. Resets portfolio to nav_usd each bar so
    /// impact costs are consistent throughout. Returns are still %-based.
    /// Use for Optuna tuning; disable for reporting/diagnostics.
    #[serde(default)]
    pub constant_nav: bool,
    /// One-sided NAV cap for deployment-realistic tuning. When `Some(cap)`:
    /// after each bar's PnL update, if `nav > cap`, the engine "skims" the
    /// excess by scaling positions back to `cap` (same mechanic as
    /// constant_nav reset, but conditional on overshooting only). Loss days
    /// are NOT touched — `nav` falls naturally and the strategy trades the
    /// reduced capital next bar. Skimmed amounts are accumulated in
    /// `cumulative_skimmed_usd` for diagnostics. Models the deployment
    /// pattern "compound up to cap, skim above cap" without the free-money
    /// pathology of the legacy constant_nav mode (which resets both
    /// directions, magically refilling losses). Takes precedence over
    /// `constant_nav` when both are set.
    #[serde(default)]
    pub nav_cap_usd: Option<f64>,
    /// Walk-forward NAV reset boundaries. After the LAST H1 bar of each day
    /// listed here completes (record pushed), the engine performs a
    /// constant_nav-style scale-and-refill (positions scaled, cash refilled
    /// to `nav_usd`). Values are interpreted as DAY-FLOOR unix seconds
    /// (`(ts / 86400) * 86400`); any value within a target reset day works
    /// (the engine floors during set construction). Used by the MOO walk-
    /// forward tuner to make per-fold metrics directly comparable: each
    /// fold starts at the same effective NAV, so impact-cost regime is the
    /// same across folds.
    ///
    /// Distinct from `constant_nav` (resets every bar) and `nav_cap_usd`
    /// (one-sided skim above cap). Composes additively when others are off:
    /// fold-reset only fires at the listed days. When `nav_cap_usd` or
    /// `constant_nav` is set, those take precedence per the existing rules.
    ///
    /// The list is converted to a HashSet for O(1) lookup at simulation
    /// start; order doesn't matter. Empty (default) = no fold resets.
    /// Memory ref: project_execution_moo_design.md (2026-05-09).
    #[serde(default)]
    pub nav_reset_timestamps: Vec<i64>,
    // L2 (2026-05-13): `regime: Option<RegimeConfig>` removed with the
    // per_name_cap overlay. Probe F's `blend_rho` (next field) is the
    // separate Markowitz-fragility line of research and is kept.

    /// Probe F (2026-05-09 — Markowitz regime-fragility fix): hybrid blend
    /// between the Markowitz-optimized weights and a naive rank-weighted
    /// vector built from `alpha_aligned`:
    ///   w_final = (1 − ρ) · w_qp + ρ · w_rank
    ///
    /// 0.0 (default) = pure Markowitz, current production behavior.
    /// 1.0           = pure rank-weighted (ignore Σ⁻¹·α; use winsorized
    ///                 z-score weights normalized to ±per_name_cap).
    ///
    /// Empirical motivation: in regime-shift periods (post 2025-12-01) the
    /// Markowitz Σ⁻¹·α product corrupts mid-rank alpha into wrong-side bets
    /// even when per-asset alpha edge is preserved. Naive rank-weighted
    /// captures the alpha edge directly (memory cross-ref:
    /// project_markowitz_regime_fragility.md, project_markowitz_fix_research.md).
    /// Both inputs are dollar-neutral and per_name_cap-bounded, so the blend
    /// inherits both properties (re-clipped + re-demeaned after blend to
    /// guarantee).
    #[serde(default)]
    pub blend_rho: f64,
}

/// Defaults from execution_v10_cdar_l3 champion (#211, 250 trials, CDaR λ=3).
impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            qp: QPConfig {
                gamma: 4.795,
                lambda_aim: 0.054,
                l_max: 1.5,
                per_name_cap: Some(0.052),
                ridge: 1e-6,
                prune_top_k: None,
                sigma_target_daily: None,
                per_name_cap_vol_scale: false,
                topk_cap: None,
                per_name_cap_floor_safety_mult: 1.5,
                per_name_cap_vol_scale_ceiling_mult: 5.0,
                per_name_cap_default: 0.05,
            },
            covariance: CovarianceConfig {
                n_factors: 8,
                // 200 day halflife: empirical optimum from 10-point CLI
                // sensitivity test on v10 #402 params (2026-04-27).
                // Sharpe peaks at hl=220 (2.852), with rolloff on both sides
                // (2.69 at hl=60 / 2.69 at hl=430). 200 captures ~99% of
                // available lift (+0.18 Sharpe vs prior default of 45).
                // Pattern is a real basin (not monotone), arguing for
                // genuine signal — crypto factor cov is slow-moving,
                // 6-month memory captures BTC/altcoin cycle structure.
                ewma_halflife: 200,
                min_history: 60,
                min_assets: 10,
                winsorize_sigma: 5.0,
                ridge: 1e-6,
                shrinkage_rho: 0.0,
            },
            alpha: AlphaConfig {
                horizons: vec![3, 6, 12, 24, 50],
                winsorize_sigma: 3.0,
                l_target: 1.0,
                soft_threshold_z: 0.0,
                vol_scale: false,
            },
            funding: FundingConfig {
                enabled: true,
                // 5h: empirical pooled halflife from HL funding history (2026-04-27 calibration)
                ewma_halflife_hours: 5,
                long_run_mean_window_hours: 720, // 30 days
                default_rho: 0.89,
                holding_hours: 24,
                use_in_alpha: false,
                alpha_demean: true,
                alpha_scale: 1.0,
            },
            cost_model: CostModelConfig {
                c_lin_multiplier: 1.0,
                sigma_halflife_days: 60,
                adv_halflife_days: 60,
                impact_delta: 0.10,
                q_ref_usd: 0.0,
                impact_gate_bps: 0.0,
                illiquid_dlom_pct: 0.50,
                last_known_max_age_days: 90,
                volume_is_usd: false,
                default_spread_bps: 5.0,
                spread_bps_calibration_multiplier: 1.0,
                kappa_calibration_multiplier: 1.0,
                scores_parquet: None,
                expected_sha256: None,
            },
            exchange: crate::exchange::ExchangeConfig::default(),
            nav_usd: 1000.0,
            fill_mode: FillMode::Market,
            alpha_units: AlphaUnits::DollarAlpha,
            constant_nav: false,
            nav_cap_usd: None,
            nav_reset_timestamps: Vec::new(),
            blend_rho: 0.0,
        }
    }
}
