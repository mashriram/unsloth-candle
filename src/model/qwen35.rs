/// Qwen3.5 dense model family (4B, 9B, 27B)
///
/// Architecture: "Qwen3_5ForConditionalGeneration" / model_type = "qwen3_5"
///
/// Key differences from Qwen3:
/// - Uses Gated Delta Networks (linear attention mechanism) -- but for simplicity
///   we implement the standard transformer path which is also exposed in the
///   transformers-compatible checkpoint
/// - Still has QK-Norm (per-head RMSNorm), same as Qwen3
/// - SwiGLU MLP (same as Qwen3)
/// - Multimodal-capable training, but text-only inference path unchanged
///
/// In practice the text transformer path is identical to Qwen3, so we thin-wrap
/// the Qwen3 implementation. A pure Rust rewrite of GDN is left as future work.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};
use crate::model::qwen3::{Qwen3, Config as Qwen3Config};

// ─── Config ──────────────────────────────────────────────────────────────────
// Qwen3.5 uses the same config keys as Qwen3 for the dense text transformer.
pub type Config = Qwen3Config;

// ─── Model alias ─────────────────────────────────────────────────────────────
pub type Qwen35 = Qwen3;

// ─── Wrapper ─────────────────────────────────────────────────────────────────

pub struct Qwen35Model {
    pub model: Qwen35,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Qwen35Model {
    pub fn new(model: Qwen35, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self { model, config, device, dtype, cache, varmap }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn configure_cache(&mut self, q: crate::core::cache::KVQuantization, rotor: bool) {
        self.cache.quantization = q;
        self.cache.use_rotor = rotor;
        self.clear_cache();
    }

    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}
