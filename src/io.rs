//! Parquet I/O for the execution pipeline.
//!
//! Reads: alpha score parquets, OHLCV, funding, previous weights, config YAML.
//! Writes: weights_final.parquet, weights_qp_raw.parquet.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray, LargeStringArray, Array, ArrayRef};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::pipeline::WeightOutput;

/// Extract string value from a column that may be Utf8 or LargeUtf8.
/// Polars writes LargeUtf8, standard Arrow writes Utf8.
fn get_string_value(col: &dyn Array, idx: usize) -> String {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return arr.value(idx).to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return arr.value(idx).to_string();
    }
    panic!("symbol column is neither Utf8 nor LargeUtf8");
}

/// Extract f64 value from a column that may be Float32 or Float64.
fn get_f64_value(col: &dyn Array, idx: usize) -> f64 {
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return arr.value(idx);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return arr.value(idx) as f64;
    }
    panic!("numeric column is neither Float32 nor Float64");
}

/// Load alpha scores from a single horizon parquet.
/// Returns Vec<(timestamp, symbol, score)>.
pub fn load_alpha_scores(path: &Path, horizon: usize) -> Result<Vec<(i64, String, f64)>> {
    let file = File::open(path)
        .with_context(|| format!("opening alpha scores: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(100_000).build()?;

    // Find the score column (alpha_{target}_h{N})
    let score_col_idx = schema.fields().iter().position(|f| {
        let name = f.name();
        name.starts_with("alpha_") && name.ends_with(&format!("_h{horizon}"))
    }).with_context(|| format!("no alpha score column for h{horizon} in {}", path.display()))?;

    let ts_idx = schema.fields().iter().position(|f| f.name() == "timestamp")
        .context("missing timestamp column")?;
    let sym_idx = schema.fields().iter().position(|f| f.name() == "symbol")
        .context("missing symbol column")?;

    let mut result = Vec::new();
    for batch in reader {
        let batch = batch?;
        let timestamps = batch.column(ts_idx).as_any().downcast_ref::<Int64Array>().unwrap();
        let sym_col = batch.column(sym_idx);
        let scores = batch.column(score_col_idx).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..batch.num_rows() {
            let score = scores.value(i);
            if score.is_finite() {
                result.push((timestamps.value(i), get_string_value(sym_col.as_ref(), i), score));
            }
        }
    }
    Ok(result)
}

/// Load OHLCV parquet. Returns (timestamps, symbols, closes, volumes, market_caps)
/// as parallel vecs grouped by symbol.
pub fn load_ohlcv(path: &Path) -> Result<OhlcvData> {
    let file = File::open(path)
        .with_context(|| format!("opening OHLCV: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(200_000).build()?;

    let col_idx = |name: &str| -> Option<usize> {
        schema.fields().iter().position(|f| f.name() == name)
    };
    let ts_idx = col_idx("timestamp").context("missing timestamp")?;
    let sym_idx = col_idx("symbol").context("missing symbol")?;
    let close_idx = col_idx("close").context("missing close")?;
    let vol_idx = col_idx("volume").context("missing volume")?;
    let mcap_idx = col_idx("market_cap").context("missing market_cap")?;

    let mut data: HashMap<String, Vec<(i64, f64, f64, f64)>> = HashMap::new();

    for batch in reader {
        let batch = batch?;
        let timestamps = batch.column(ts_idx).as_any().downcast_ref::<Int64Array>().unwrap();
        let sym_col = batch.column(sym_idx);
        let closes = batch.column(close_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        let volumes = batch.column(vol_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        let mcaps = batch.column(mcap_idx).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..batch.num_rows() {
            let sym = get_string_value(sym_col.as_ref(), i);
            data.entry(sym).or_default().push((
                timestamps.value(i), closes.value(i), volumes.value(i), mcaps.value(i),
            ));
        }
    }

    // Sort each asset by timestamp
    for rows in data.values_mut() {
        rows.sort_by_key(|(ts, _, _, _)| *ts);
    }

    Ok(OhlcvData { by_symbol: data })
}

/// OHLCV data grouped by symbol.
pub struct OhlcvData {
    pub by_symbol: HashMap<String, Vec<(i64, f64, f64, f64)>>, // ts, close, volume, mcap
}

impl OhlcvData {
    /// Get all unique timestamps across all symbols, sorted.
    pub fn timestamps(&self) -> Vec<i64> {
        let mut ts_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for rows in self.by_symbol.values() {
            for (ts, _, _, _) in rows {
                ts_set.insert(*ts);
            }
        }
        ts_set.into_iter().collect()
    }

    /// Get all symbols.
    pub fn symbols(&self) -> Vec<String> {
        let mut syms: Vec<String> = self.by_symbol.keys().cloned().collect();
        syms.sort();
        syms
    }

    /// Build wide-format close matrix (n_bars × n_syms) row-major.
    /// Returns (data, timestamps, symbols). Missing = NaN.
    pub fn close_matrix(&self) -> (Vec<f64>, Vec<i64>, Vec<String>) {
        let timestamps = self.timestamps();
        let symbols = self.symbols();
        let n_bars = timestamps.len();
        let n_syms = symbols.len();

        let ts_to_idx: HashMap<i64, usize> = timestamps.iter().enumerate().map(|(i, &ts)| (ts, i)).collect();
        let sym_to_idx: HashMap<&str, usize> = symbols.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

        let mut matrix = vec![f64::NAN; n_bars * n_syms];
        for (sym, rows) in &self.by_symbol {
            if let Some(&si) = sym_to_idx.get(sym.as_str()) {
                for (ts, close, _, _) in rows {
                    if let Some(&ti) = ts_to_idx.get(ts) {
                        matrix[ti * n_syms + si] = *close;
                    }
                }
            }
        }

        (matrix, timestamps, symbols)
    }
}

/// Load previous weights parquet. Returns map of (symbol → weight).
pub fn load_prev_weights(path: &Path) -> Result<HashMap<String, f64>> {
    let file = File::open(path)
        .with_context(|| format!("opening prev weights: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(10_000).build()?;

    let sym_idx = schema.fields().iter().position(|f| f.name() == "symbol")
        .context("missing symbol column")?;
    let w_idx = schema.fields().iter().position(|f| f.name() == "weight_qp")
        .context("missing weight_qp column")?;

    let mut weights = HashMap::new();
    for batch in reader {
        let batch = batch?;
        let sym_col = batch.column(sym_idx);
        let ws = batch.column(w_idx).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..batch.num_rows() {
            weights.insert(get_string_value(sym_col.as_ref(), i), ws.value(i));
        }
    }
    Ok(weights)
}

/// Write weights to parquet.
pub fn write_weights(
    path: &Path,
    weights: &[WeightOutput],
    weight_col: &str, // "weight_final" or "weight_qp"
) -> Result<()> {
    let timestamps: Vec<i64> = weights.iter().map(|w| w.timestamp).collect();
    let symbols: Vec<&str> = weights.iter().map(|w| w.symbol.as_str()).collect();
    let values: Vec<f64> = weights.iter().map(|w| {
        if weight_col == "weight_qp" { w.weight_qp } else { w.weight_final }
    }).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new(weight_col, DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(timestamps)) as ArrayRef,
            Arc::new(StringArray::from(symbols)) as ArrayRef,
            Arc::new(Float64Array::from(values)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Write per-bar (per-day) simulation records to parquet for downstream
/// window-sliced analysis (e.g. holdout-only Sharpe / CDaR computation).
///
/// Columns:
///   timestamp (i64 UTC sec) — day-end ts matching `weights_final.timestamp`
///   portfolio_return (f64) — NET return after funding + fees + impact (mark NAV)
///   portfolio_return_liquid (f64) — Layer 2 net-of-haircut return (Frazzini-Israel-Moskowitz)
///   nav (f64) — end-of-day mark NAV
///   nav_liquid (f64) — end-of-day liquid NAV (mark − liquidation haircut)
///   commission (f64) — taker fee dollars, positive = deducted
///   spread (f64) — half-spread cost dollars, positive = deducted
///   impact (f64) — power-law impact dollars, positive = deducted
///   funding (f64) — funding pnl, signed (positive = received)
///   turnover (f64) — gross-notional turnover for the bar
pub fn write_per_bar_returns(
    path: &Path,
    per_bar_returns: &[(i64, f64)],
    per_bar_returns_liquid: &[(i64, f64)],
    cost_breakdown: &[crate::backtest::DailyCosts],
    per_bar_nav_liquid: &[f64],
) -> Result<()> {
    let n = per_bar_returns.len();
    anyhow::ensure!(per_bar_returns_liquid.len() == n,
        "per_bar_returns_liquid length mismatch: {} vs {}", per_bar_returns_liquid.len(), n);
    anyhow::ensure!(cost_breakdown.len() == n,
        "cost_breakdown length mismatch: {} vs {}", cost_breakdown.len(), n);
    anyhow::ensure!(per_bar_nav_liquid.len() == n,
        "per_bar_nav_liquid length mismatch: {} vs {}", per_bar_nav_liquid.len(), n);

    let timestamps: Vec<i64> = per_bar_returns.iter().map(|(ts, _)| *ts).collect();
    let returns: Vec<f64> = per_bar_returns.iter().map(|(_, r)| *r).collect();
    let returns_liq: Vec<f64> = per_bar_returns_liquid.iter().map(|(_, r)| *r).collect();
    let navs: Vec<f64> = cost_breakdown.iter().map(|c| c.nav).collect();
    let navs_liq: Vec<f64> = per_bar_nav_liquid.to_vec();
    let commissions: Vec<f64> = cost_breakdown.iter().map(|c| c.commission).collect();
    let spreads: Vec<f64> = cost_breakdown.iter().map(|c| c.spread).collect();
    let impacts: Vec<f64> = cost_breakdown.iter().map(|c| c.impact).collect();
    let fundings: Vec<f64> = cost_breakdown.iter().map(|c| c.funding).collect();
    let turnovers: Vec<f64> = cost_breakdown.iter().map(|c| c.turnover).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("portfolio_return", DataType::Float64, false),
        Field::new("portfolio_return_liquid", DataType::Float64, false),
        Field::new("nav", DataType::Float64, false),
        Field::new("nav_liquid", DataType::Float64, false),
        Field::new("commission", DataType::Float64, false),
        Field::new("spread", DataType::Float64, false),
        Field::new("impact", DataType::Float64, false),
        Field::new("funding", DataType::Float64, false),
        Field::new("turnover", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(timestamps)) as ArrayRef,
            Arc::new(Float64Array::from(returns)) as ArrayRef,
            Arc::new(Float64Array::from(returns_liq)) as ArrayRef,
            Arc::new(Float64Array::from(navs)) as ArrayRef,
            Arc::new(Float64Array::from(navs_liq)) as ArrayRef,
            Arc::new(Float64Array::from(commissions)) as ArrayRef,
            Arc::new(Float64Array::from(spreads)) as ArrayRef,
            Arc::new(Float64Array::from(impacts)) as ArrayRef,
            Arc::new(Float64Array::from(fundings)) as ArrayRef,
            Arc::new(Float64Array::from(turnovers)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Load ML cost scores parquet (H1 spread + kappa predictions).
///
/// Expected columns: timestamp (i64), symbol (Utf8), predicted_log_spread_bps (f64), predicted_log_kappa (f64).
/// Returns raw (timestamp, symbol, log_spread, log_kappa) tuples — caller exp-transforms via `build_ml_scores()`.
pub fn load_cost_scores(path: &Path) -> Result<Vec<(i64, String, f64, f64)>> {
    let file = File::open(path)
        .with_context(|| format!("opening cost scores: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(200_000).build()?;

    let col_idx = |name: &str| -> Option<usize> {
        schema.fields().iter().position(|f| f.name() == name)
    };
    let ts_idx = col_idx("timestamp").context("missing timestamp column in cost scores")?;
    let sym_idx = col_idx("symbol").context("missing symbol column in cost scores")?;
    let spread_idx = col_idx("predicted_log_spread_bps")
        .context("missing predicted_log_spread_bps column in cost scores")?;
    let kappa_idx = col_idx("predicted_log_kappa")
        .context("missing predicted_log_kappa column in cost scores")?;

    let mut result = Vec::new();
    for batch in reader {
        let batch = batch?;
        let timestamps = batch.column(ts_idx).as_any().downcast_ref::<Int64Array>().unwrap();
        let sym_col = batch.column(sym_idx);
        let spread_col = batch.column(spread_idx);
        let kappa_col = batch.column(kappa_idx);

        for i in 0..batch.num_rows() {
            let s = get_f64_value(spread_col.as_ref(), i);
            let k = get_f64_value(kappa_col.as_ref(), i);
            if s.is_finite() && k.is_finite() {
                result.push((
                    timestamps.value(i),
                    get_string_value(sym_col.as_ref(), i),
                    s,
                    k,
                ));
            }
        }
    }
    Ok(result)
}

/// Load execution config from YAML.
pub fn load_config(path: &Path) -> Result<crate::config::ExecutionConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config: {}", path.display()))?;
    let config: crate::config::ExecutionConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("parsing config: {}", path.display()))?;
    Ok(config)
}

// ---------------------------------------------------------------------------
// Artifact lineage helpers (Phase 1, 2026-05-18)
// ---------------------------------------------------------------------------

/// Streaming SHA-256 of a file. 1 MiB chunks; suitable for >GB parquets.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = File::open(path)
        .with_context(|| format!("opening for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)
            .with_context(|| format!("reading for hash: {}", path.display()))?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Read a single key from a parquet file's kv_metadata.
/// Returns `Ok(None)` if the file has no kv_metadata or the key is absent.
pub fn read_parquet_kv(path: &Path, key: &str) -> Result<Option<String>> {
    let file = File::open(path)
        .with_context(|| format!("opening parquet for metadata: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let md = builder.metadata().file_metadata().key_value_metadata();
    if let Some(kvs) = md {
        for entry in kvs {
            if entry.key == key {
                return Ok(entry.value.clone());
            }
        }
    }
    Ok(None)
}

/// Validate `path`'s SHA-256 against `expected`. When `expected` is `None`,
/// no check is performed (legacy specs that lack the pin). When set and the
/// actual SHA differs, returns a descriptive error so the run is refused at
/// load time rather than silently producing weights against an unintended
/// cost surface. `label` is folded into the error message for clarity.
pub fn validate_artifact_sha(
    path: &Path,
    expected: Option<&str>,
    label: &str,
) -> Result<()> {
    let Some(expected) = expected else { return Ok(()); };
    let actual = sha256_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        anyhow::bail!(
            "{label} SHA-256 mismatch\n  spec expected: {}\n  file actual:   {}\n  path: {}\nRefusing to run. Re-tune against the current artifact, or update the spec's expected_sha256 if the swap is intentional.",
            expected, actual, path.display(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_write_read_roundtrip() {
        let weights = vec![
            WeightOutput { timestamp: 100, symbol: "BTC".into(), weight_qp: 0.5, weight_final: 0.4 },
            WeightOutput { timestamp: 100, symbol: "ETH".into(), weight_qp: -0.3, weight_final: -0.25 },
            WeightOutput { timestamp: 200, symbol: "BTC".into(), weight_qp: 0.6, weight_final: 0.5 },
        ];

        let dir = std::env::temp_dir().join("rumpy_test_io");
        std::fs::create_dir_all(&dir).unwrap();

        // Write QP raw
        let qp_path = dir.join("test_qp.parquet");
        write_weights(&qp_path, &weights, "weight_qp").unwrap();

        // Write final
        let final_path = dir.join("test_final.parquet");
        write_weights(&final_path, &weights, "weight_final").unwrap();

        // Read back QP raw as prev weights
        let prev = load_prev_weights(&qp_path).unwrap();
        assert_eq!(prev.len(), 2); // BTC overwritten by bar 200
        assert!((prev["BTC"] - 0.6).abs() < 1e-10);
        assert!((prev["ETH"] - (-0.3)).abs() < 1e-10);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sha256_file_and_validate_artifact_sha() {
        // Hash a known-content file, then exercise validate_artifact_sha
        // accept / reject / skip-when-None paths.
        let dir = std::env::temp_dir().join("rumpy_test_io_sha");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("known.bin");
        std::fs::write(&p, b"hello world\n").unwrap();
        // Expected sha256 of "hello world\n" (with the trailing newline).
        let expected = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447";

        let actual = sha256_file(&p).unwrap();
        assert_eq!(actual, expected);

        // None → no check, returns Ok.
        validate_artifact_sha(&p, None, "test").unwrap();
        // Matching → Ok.
        validate_artifact_sha(&p, Some(expected), "test").unwrap();
        // Mismatched → Err.
        let bad = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = validate_artifact_sha(&p, Some(bad), "test").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("mismatch"), "got: {msg}");
        assert!(msg.contains(expected), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
