# execution/ — CLAUDE.md

QP-based portfolio optimizer for crypto L/S execution. Holds the unified solver (post-Phase-2 consolidation): both `rumpy execution backtest` and `rumpy execution solve` flow through `SimulationEngine` via the same code path.

## Architecture

```
unified execution_h1.parquet (baked by features/src/execution.rs)
   ├── alpha_future (D1, broadcast to H1)
   ├── ohlcv (H1)
   ├── spread_bps, kappa (from cost-scores parquet, exp-transformed)
   └── funding_rate (H1, from HL fundingHistory)
       ↓
  backtest::PreloadedData (one-time mat load, ~2 GB RSS @ full universe)
       ↓
  preproc.rs: rolling cov → factor PCA → aim weights → forward returns
       ↓
  SimulationEngine (bar-by-bar holdings-based loop)
       ├── alpha alignment + funding-aware adjustment (Koijen 2018, optional)
       ├── covariance shrinkage (ρ-blended, optional)
       ├── soft-threshold alpha denoise (optional)
       ├── QP solve (Clarabel SOC + Anis-Kwon top-K LP + ρ-blend rank-vector)
       ├── post-rescale (vol-target, NAV-cap, fold reset)
       └── trade settlement (TWAP or Market fill, impact gate, illiq haircut)
       ↓
  weights_final.parquet + per_bar_returns.parquet + metrics
```

## Entry points

### `rumpy execution backtest --execution-h1 PATH --config SPEC`

Full walk-forward backtest. Consumes pre-baked unified parquet directly.

Used by the tuner (`train/scripts/tune_execution.py`), validation runs, and ad-hoc research.

### `rumpy execution solve --alpha-dir ... --ohlcv ... --cost-scores ... --funding ... --config SPEC`

**Post-Phase-2 (2026-05-18)**: rewritten as `build_execution + run_backtest`. Composes:
1. `rumpy_features::execution::build_execution(...)` to a temp unified parquet
2. `crates::execution::backtest::run_backtest(...)` on that temp parquet

Live path (maestro) shells out to this command. Shares the `SimulationEngine` code path with `execution backtest` by construction — parity is bit-equivalent up to FP noise.

History: pre-Phase-2, `solve` ran a separate `pipeline::run_qp_impact` loop that diverged from `simulation::SimulationEngine` (Pearson 0.75, 7% sign-flip rate, 37% gross-leverage gap on identical inputs). The divergent solver was deleted; see `pipeline.rs` for the residual type definitions still consumed by `gates.rs` and adjacent modules.

## Module map

| Module | Role |
|---|---|
| `lib.rs` | Public surface; wires modules |
| `backtest.rs` | `PreloadedData`, `run_backtest`, `BacktestPaths`, summary printer |
| `simulation.rs` | `SimulationEngine` — bar-by-bar holdings-based loop, ~2000 LOC. Both Market and TWAP fill modes |
| `solver.rs` | `CachedOSQP` / `CachedClarabel` wrappers; QP construction (Σ, c_lin, A, b) |
| `aim.rs` | `AimWeight` — rolling aim portfolio with movement cost (Gârleanu-Pedersen 2013) |
| `alpha.rs` | `AlphaRow`, alpha winsorization, soft-threshold denoise, vol-scale (Path 4) |
| `covariance.rs` | `RollingCovariance` factor model with EWMA + ρ-shrinkage |
| `cost.rs` | `FundingModel`, `CostMatrix` — runtime cost-cone construction |
| `ml_cost.rs` | `CostProvider` chain: L2 > ML > error (legacy spec hook; unused after Phase 2) |
| `preproc.rs` | Sidecar (`*.preproc.bin`) computation: cov factors, aim, sigma, ADV, fwd returns |
| `config.rs` | `ExecutionConfig`, `CostModelConfig`, `QPConfig`, `FundingConfig`, etc. Phase 1 lineage fields live here |
| `io.rs` | Parquet readers/writers; `sha256_file`, `read_parquet_kv`, `validate_artifact_sha` (Phase 1d helpers) |
| `position.rs` | `PortfolioBook` — actual holdings, marked-to-market |
| `exchange.rs` | `ExchangeConfig`, fee tiers, `VolumeTracker` |
| `gates.rs` | Tradeability gate (`passes_impact_gate`) — refuses fills above impact threshold |
| `universe.rs` | Symbol interning, alpha-aligned universe filtering |
| `walkforward.rs` | Per-fold metrics aggregation (5-fold default; returns empty on short datasets) |
| `metrics.rs` | Sharpe, MDD, CDaR, Calmar, DSR, block-bootstrap CI |
| `diagnostics.rs` | Per-bar / monthly diagnostic tables printed at run end |
| `pipeline.rs` | Type definitions only (`WeightOutput`, `SolveStats`) — legacy `run_qp_impact` deleted 2026-05-18 |
| `python.rs` | PyO3 bindings (feature-gated `python`) |

## QP knobs (`QPConfig`)

The QP is `min ½γ·w'Σw − α'w + λ·c_lin(w − w_prev) + …` subject to:

- **per_name_cap** — scalar `|w_i| ≤ c` or vol-scaled `|w_i| ≤ c·σ_target/σ_i` (Richard-Roncalli 2019, Path 1 — falsified empirically)
- **topk_cap** — Anis-Kwon 2022 sum-of-largest-K LP via auxiliary `τ + Σ slacks`. Deployed in prod_v4 (`K=10, θ=0.186`)
- **sigma_target_daily** — SOC constraint `||L'w|| ≤ σ` where Σ = LL'; replaces post-rescale (L2 leverage refactor 2026-05-13)
- **prune_top_k** — restrict QP to top-K assets by |alpha|; existing positions bleed off
- **ridge** — small Σ diagonal additive

See `config.rs` for the full struct + per-field rationale.

## Cost-model artifact lineage (Phase 1)

Two optional fields on `CostModelConfig` pin a spec to a specific cost-scores parquet:

```yaml
cost_model:
  scores_parquet: data/scores/spread_kappa_scores_h1_v3.parquet
  expected_sha256: 9e7bd0962a0893fbdbf3f3263377d7c125bf0c0fbb959e7d3c62c9de208c68ba
```

When set, the runtime:

1. `features build-execution` stamps `cost_scores_sha256` into the unified parquet's `kv_metadata` at bake time.
2. `execution backtest` reads the stamp and compares to `cfg.cost_model.expected_sha256`. Mismatch → refuses to run.
3. `execution solve` hashes the `--cost-scores PATH` directly and compares to the spec.

Both fields are `Option` — legacy specs without lineage continue to load (warn-only when no stamp present in parquet).

## Phase status (2026-05-18)

| Phase | Status | Reference |
|---|---|---|
| 0 — decisions | ✅ | session_2026_05_17_cost_consolidation_plan |
| 1 — artifact lineage | ✅ PR #48 | session_2026_05_18_phase1_artifact_lineage |
| 2 — unify solve+backtest | ✅ PR #48 | session_2026_05_17_cost_consolidation_plan |
| 3 — code cleanup | PR #49 | session_2026_05_18_phase1_artifact_lineage |
| 4 — orchestrator | PR #50 | `scripts/cost-rebuild.sh` |
| 5 — docs | PR #51 (this) | this CLAUDE.md + cost/ + retune procedure |

Pre-Phase-2 deletions:
- `pipeline::run_qp_impact` — the divergent live-path solver
- `crates/execution/src/spread_features.rs` — the orphan spread feature module
- `CostSource` enum and `CostModelConfig.source` field — dead config

C2 deletions (2026-05-21, PR #60 — execution model consolidation per `docs/audits/execution-state-2026-05-21.md`):
- `crates/execution/src/risk.rs` (477 LOC) — `IcScaler` + `DdDampener` post-QP leverage adjustments. Header docstring claimed "QP → vol target → IC scaler × DD dampener → sizing → final weights" but `apply_leverage_adjustments` was only ever called from `risk.rs`'s own `#[cfg(test)]`. `RiskConfig` field on `ExecutionConfig` deserialized but was never read.
- `crates/execution/src/sizing.rs` (377 LOC) — benchmark-port functions (`apply_per_name_cap`, `apply_participation_cap`, `apply_min_notional_filter`, `apply_sizing_pipeline`, `SizingConfig`, `SizingDiagnostic`). Same story — only invoked by own tests. NAV-management (`apply_nav_cap_skim`, `apply_nav_reset_to_target`) lives in `simulation.rs`, not here.
- 5 dead/decorative fields on live structs: `QPConfig.n_iter`, `CostModelConfig.qp_liquidity_typical_w`, `CostModelConfig.model_git_sha`, `ExchangeConfig.max_leverage`, `ExchangeConfig.backstop_fraction`.
- `crates/execution/examples/probe_layer1.rs` — diagnostic probe for the now-deleted `qp_liquidity_typical_w` Layer 1 mechanic.

## Testing & gates

- Unit tests: `cargo test -p rumpy-execution --lib` (174 passing post-C2; was 191 pre-C2, -17 = 10 risk + 6 sizing + 1 dead-field no-op test)
- Gold-standard parity test: ensures `execution solve` and `execution backtest` produce bit-equivalent weights on identical inputs. Threshold: Pearson > 0.99999, sign match > 99.99%, max |Δw| < 1e-3. Outputs land at `/tmp/parity_phase{N}/` on desktop.

After any change to `simulation.rs`, `solver.rs`, `pipeline.rs`, `features/src/execution.rs`, or `cli/src/execution.rs`, run the parity test to detect drift. The current gold standard is `/tmp/parity_phase3/` (Pearson 0.9999999946, sign 100%, max |Δw| 1.33e-4).

## Related docs

- [`crates/features/src/cost/CLAUDE.md`](../../features/src/cost/CLAUDE.md) — bar-level cost-feature build (the producer side)
- [`docs/procedures/PROCEDURE-cost-model-retune.md`](../../../docs/procedures/PROCEDURE-cost-model-retune.md) — gated retune workflow
- [`scripts/cost-rebuild.sh`](../../../scripts/cost-rebuild.sh) — single orchestrator for the artifact chain

## References

- Boyd et al. 2017 — "Multi-Period Trading via Convex Optimization"
- Gârleanu-Pedersen 2013 — aim portfolio framework (movement cost)
- Anis-Kwon 2022 — top-K concentration LP reformulation
- Richard-Roncalli 2019 — box-approximation for vol-scaled position caps
- Koijen et al. 2018 — *Carry* (funding-aware alpha)
- Donoho-Johnstone 1994 — soft-threshold denoise
