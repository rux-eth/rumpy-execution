//! Preprocessing sidecar for the unified execution parquet.
//!
//! The per-trial `PreloadedData::load` previously computed five derived
//! lookups every time the backtest engine started:
//!   - `rolling_cov`        — D1 EWMA factor covariance via faer randomized PCA
//!   - `aim_weights`        — D1 inv(cov)·alpha aim portfolio via faer Cholesky
//!   - `sigma_lookup`       — D1 EWMA realized vol per asset
//!   - `adv_lookup`         — D1 EWMA dollar volume per asset
//!   - `forward_returns`    — D1 next-day log returns per asset
//!
//! All five depend only on FIXED inputs (parquet content + a small set of
//! config fields that aren't tuned across Optuna trials). Computing them
//! every preload was wasted work; worse, the computations include
//! cross-platform-divergent primitives (faer SIMD, libm `ln()`) which made
//! Mac and Desktop produce ULP-different lookup tables — the residual cause
//! of the cross-platform divergence we chased.
//!
//! Solution: compute these once, persist them as a binary sidecar next to
//! the parquet, and reuse on every subsequent load. Provenance fields are
//! stamped into the sidecar; if any relevant config or parquet header
//! changes, the sidecar is auto-rebuilt.
//!
//! See docs/plans/cross-platform-divergence.md.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::aim::AimWeight;
use crate::config::ExecutionConfig;
use crate::covariance::RollingCovariance;

/// File magic + format version. Bump version on any layout change.
const MAGIC: &[u8; 8] = b"RUMPYPRP";
const VERSION: u32 = 2; // v2: added arch_tag to provenance (2026-04-27)

/// Current build's arch tag (e.g. "aarch64-macos", "x86_64-linux"). Stamped
/// into every sidecar written; checked at load time to enforce that sidecars
/// are not silently consumed across architectures.
///
/// We can't make sidecar bytes bit-identical across platforms because faer's
/// linear algebra (eigendecomposition, Cholesky) calls platform libm
/// internally. Without forking faer, the safest thing is to LOUDLY refuse a
/// cross-arch sidecar so the user explicitly chooses (rsync from canonical
/// machine OR rebuild locally with `RUMPY_ALLOW_CROSS_ARCH_SIDECAR=1`).
fn current_arch_tag() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// Provenance of the preproc sidecar — drives invalidation. If any of these
/// fields differ between the saved sidecar and the current load context, the
/// sidecar is rebuilt automatically.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocProvenance {
    /// Config knobs that affect cov computation.
    pub cov_n_factors: u32,
    pub cov_ewma_halflife: u32,
    pub cov_min_history: u32,
    pub cov_min_assets: u32,
    pub cov_winsorize_sigma: f64,
    pub cov_ridge: f64,
    /// Sigma / ADV halflives.
    pub sigma_halflife_days: u32,
    pub adv_halflife_days: u32,
    /// Alpha config (l_target affects aim weights).
    pub alpha_l_target: f64,
    pub alpha_units: String,
    /// Parquet identity — invalidates if input data changed.
    pub parquet_n_rows: u64,
    pub parquet_first_ts: i64,
    pub parquet_last_ts: i64,
    pub parquet_n_symbols: u64,
    /// FNV-style content hash over D1 close + dollar-volume values.
    /// Detects same-shape but content-different parquets (e.g. alpha rebuild).
    pub content_hash: u64,
    /// Build-arch tag of the machine that wrote this sidecar (e.g.
    /// "aarch64-macos", "x86_64-linux"). Cross-arch loads are refused unless
    /// `RUMPY_ALLOW_CROSS_ARCH_SIDECAR=1`. See `current_arch_tag`.
    pub arch_tag: String,
}

impl PreprocProvenance {
    /// Build provenance from the current execution config + parquet header info.
    pub fn from_config(
        config: &ExecutionConfig,
        parquet_n_rows: u64,
        parquet_first_ts: i64,
        parquet_last_ts: i64,
        parquet_n_symbols: u64,
        content_hash: u64,
    ) -> Self {
        Self {
            cov_n_factors: config.covariance.n_factors as u32,
            cov_ewma_halflife: config.covariance.ewma_halflife as u32,
            cov_min_history: config.covariance.min_history as u32,
            cov_min_assets: config.covariance.min_assets as u32,
            cov_winsorize_sigma: config.covariance.winsorize_sigma,
            cov_ridge: config.covariance.ridge,
            sigma_halflife_days: config.cost_model.sigma_halflife_days as u32,
            adv_halflife_days: config.cost_model.adv_halflife_days as u32,
            alpha_l_target: config.alpha.l_target,
            alpha_units: match config.alpha_units {
                crate::config::AlphaUnits::DollarAlpha => "dollar_alpha".to_string(),
                crate::config::AlphaUnits::VolNormalizedExcess => "vol_normalized_excess".to_string(),
            },
            parquet_n_rows,
            parquet_first_ts,
            parquet_last_ts,
            parquet_n_symbols,
            content_hash,
            arch_tag: current_arch_tag(),
        }
    }

    fn write<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.cov_n_factors.to_le_bytes())?;
        w.write_all(&self.cov_ewma_halflife.to_le_bytes())?;
        w.write_all(&self.cov_min_history.to_le_bytes())?;
        w.write_all(&self.cov_min_assets.to_le_bytes())?;
        w.write_all(&self.cov_winsorize_sigma.to_le_bytes())?;
        w.write_all(&self.cov_ridge.to_le_bytes())?;
        w.write_all(&self.sigma_halflife_days.to_le_bytes())?;
        w.write_all(&self.adv_halflife_days.to_le_bytes())?;
        w.write_all(&self.alpha_l_target.to_le_bytes())?;
        let units_bytes = self.alpha_units.as_bytes();
        w.write_all(&(units_bytes.len() as u32).to_le_bytes())?;
        w.write_all(units_bytes)?;
        w.write_all(&self.parquet_n_rows.to_le_bytes())?;
        w.write_all(&self.parquet_first_ts.to_le_bytes())?;
        w.write_all(&self.parquet_last_ts.to_le_bytes())?;
        w.write_all(&self.parquet_n_symbols.to_le_bytes())?;
        w.write_all(&self.content_hash.to_le_bytes())?;
        let arch_bytes = self.arch_tag.as_bytes();
        w.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        w.write_all(arch_bytes)?;
        Ok(())
    }

    fn read<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf4)?;
        let cov_n_factors = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let cov_ewma_halflife = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let cov_min_history = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let cov_min_assets = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf8)?;
        let cov_winsorize_sigma = f64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let cov_ridge = f64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?;
        let sigma_halflife_days = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let adv_halflife_days = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf8)?;
        let alpha_l_target = f64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?;
        let units_len = u32::from_le_bytes(buf4) as usize;
        let mut units_bytes = vec![0u8; units_len];
        r.read_exact(&mut units_bytes)?;
        let alpha_units = String::from_utf8(units_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        r.read_exact(&mut buf8)?;
        let parquet_n_rows = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let parquet_first_ts = i64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let parquet_last_ts = i64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let parquet_n_symbols = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let content_hash = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        let mut arch_bytes = vec![0u8; arch_len];
        r.read_exact(&mut arch_bytes)?;
        let arch_tag = String::from_utf8(arch_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            cov_n_factors,
            cov_ewma_halflife,
            cov_min_history,
            cov_min_assets,
            cov_winsorize_sigma,
            cov_ridge,
            sigma_halflife_days,
            adv_halflife_days,
            alpha_l_target,
            alpha_units,
            parquet_n_rows,
            parquet_first_ts,
            parquet_last_ts,
            parquet_n_symbols,
            content_hash,
            arch_tag,
        })
    }
}

/// Compute a content hash over a D1 array (close, dollar_vol, etc).
/// Used in `PreprocProvenance.content_hash` to detect parquet content changes
/// even when shape (rows, ts range, sym count) is unchanged.
pub fn fnv_f64s(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in xs {
        h = h.wrapping_mul(0x100000001b3).wrapping_add(x.to_bits());
    }
    h
}


/// All five preprocessed lookups bundled into one struct for serialization.
pub struct Preproc {
    pub provenance: PreprocProvenance,
    pub rolling_cov: RollingCovariance,
    pub aim_weights: Vec<AimWeight>,
    pub sigma_lookup: HashMap<(i64, u32), f64>,
    pub adv_lookup: HashMap<(i64, u32), f64>,
    pub forward_returns: HashMap<(i64, u32), f64>,
}

impl Preproc {
    /// Default sidecar path next to a parquet: `<parquet>.preproc.bin`.
    pub fn sidecar_path(parquet_path: &Path) -> std::path::PathBuf {
        let mut p = parquet_path.as_os_str().to_owned();
        p.push(".preproc.bin");
        std::path::PathBuf::from(p)
    }

    /// Save to disk. Layout:
    ///   magic[8] | version[4] | provenance | cov | aim | sigma | adv | fwd
    pub fn save(&self, path: &Path) -> Result<()> {
        let f = std::fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        self.provenance.write(&mut w)?;
        write_cov(&mut w, &self.rolling_cov)?;
        write_aim(&mut w, &self.aim_weights)?;
        write_u32_lookup(&mut w, &self.sigma_lookup)?;
        write_u32_lookup(&mut w, &self.adv_lookup)?;
        write_u32_lookup(&mut w, &self.forward_returns)?;
        w.flush()?;
        Ok(())
    }

    /// Load and validate provenance against `expected`. Returns Err if the
    /// sidecar is invalid or doesn't match `expected` — caller should fall
    /// back to recomputing.
    pub fn load(path: &Path, expected: &PreprocProvenance) -> Result<Self> {
        let f = std::fs::File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut r = std::io::BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            anyhow::bail!("bad magic");
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            anyhow::bail!("version mismatch (file={}, expected={VERSION})", version);
        }
        let provenance = PreprocProvenance::read(&mut r)?;

        // Arch check: faer's eigendecomposition / Cholesky / SVD use platform
        // libm internally (we can't easily replace), so cross-arch sidecars are
        // not bit-identical. Refuse to load loudly so the user explicitly
        // chooses to either rsync from the canonical build machine or
        // override with `RUMPY_ALLOW_CROSS_ARCH_SIDECAR=1`.
        let mut prov_for_compare = provenance.clone();
        if provenance.arch_tag != expected.arch_tag {
            let allow = std::env::var("RUMPY_ALLOW_CROSS_ARCH_SIDECAR").is_ok();
            if !allow {
                anyhow::bail!(
                    "sidecar arch mismatch: file built on '{}', current machine is '{}'.\n\
                     Cross-arch sidecars diverge from faer/libm internals. Either:\n\
                       1. rsync the canonical sidecar from the build machine\n\
                       2. set RUMPY_ALLOW_CROSS_ARCH_SIDECAR=1 to load anyway (NOT for production)\n\
                       3. set RUMPY_PREPROC_REBUILD=1 to rebuild locally (results will differ slightly)",
                    provenance.arch_tag, expected.arch_tag
                );
            }
            // Override active — compare everything else as if arch matched.
            prov_for_compare.arch_tag = expected.arch_tag.clone();
        }
        if prov_for_compare != *expected {
            anyhow::bail!("provenance mismatch (config or parquet changed)");
        }
        let rolling_cov = read_cov(&mut r)?;
        let aim_weights = read_aim(&mut r)?;
        let sigma_lookup = read_u32_lookup(&mut r)?;
        let adv_lookup = read_u32_lookup(&mut r)?;
        let forward_returns = read_u32_lookup(&mut r)?;
        Ok(Self {
            provenance,
            rolling_cov,
            aim_weights,
            sigma_lookup,
            adv_lookup,
            forward_returns,
        })
    }
}

// ---------------------------------------------------------------------------
// Section serializers
// ---------------------------------------------------------------------------

fn write_cov<W: Write>(w: &mut W, cov: &RollingCovariance) -> std::io::Result<()> {
    cov.write_to(w)
}

fn read_cov<R: Read>(r: &mut R) -> std::io::Result<RollingCovariance> {
    RollingCovariance::read_from(r)
}

fn write_aim<W: Write>(w: &mut W, aim: &[AimWeight]) -> std::io::Result<()> {
    w.write_all(&(aim.len() as u64).to_le_bytes())?;
    for a in aim {
        w.write_all(&a.timestamp.to_le_bytes())?;
        let sb = a.symbol.as_bytes();
        w.write_all(&(sb.len() as u32).to_le_bytes())?;
        w.write_all(sb)?;
        w.write_all(&a.weight_aim.to_le_bytes())?;
    }
    Ok(())
}

fn read_aim<R: Read>(r: &mut R) -> std::io::Result<Vec<AimWeight>> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut buf8)?;
        let timestamp = i64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?;
        let len = u32::from_le_bytes(buf4) as usize;
        let mut sb = vec![0u8; len];
        r.read_exact(&mut sb)?;
        let symbol = String::from_utf8(sb)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        r.read_exact(&mut buf8)?;
        let weight_aim = f64::from_le_bytes(buf8);
        out.push(AimWeight { timestamp, symbol, weight_aim });
    }
    Ok(out)
}

fn write_u32_lookup<W: Write>(w: &mut W, map: &HashMap<(i64, u32), f64>) -> std::io::Result<()> {
    let mut keys: Vec<(i64, u32)> = map.keys().copied().collect();
    keys.sort();
    w.write_all(&(keys.len() as u64).to_le_bytes())?;
    for (ts, sid) in keys {
        w.write_all(&ts.to_le_bytes())?;
        w.write_all(&sid.to_le_bytes())?;
        w.write_all(&map[&(ts, sid)].to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_lookup<R: Read>(r: &mut R) -> std::io::Result<HashMap<(i64, u32), f64>> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    let mut map = HashMap::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut buf8)?;
        let ts = i64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?;
        let sid = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf8)?;
        let val = f64::from_le_bytes(buf8);
        map.insert((ts, sid), val);
    }
    Ok(map)
}
