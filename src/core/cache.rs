use candle_core::{Tensor, Result, DType};

use crate::kernels::turbo_quant::{
    apply_planar_rotation, apply_inverse_planar_rotation,
    quantize_4bit, dequantize_4bit, quantize_8bit, dequantize_8bit,
};

#[derive(Clone, Debug, PartialEq)]
pub enum KVQuantization {
    None,
    Q4_0,
    Q8_0,
}

#[derive(Clone)]
pub enum CacheState {
    Full(Tensor, Tensor),
    Quantized {
        k_data: Tensor,
        k_scale: Tensor,
        v_data: Tensor,
        v_scale: Tensor,
    },
}

#[derive(Clone)]
pub struct Cache {
    pub kvs: Vec<Option<CacheState>>,
    pub use_kv_cache: bool,
    pub quantization: KVQuantization,
    pub use_rotor: bool,
}

impl Cache {
    pub fn new(use_kv_cache: bool, num_layers: usize) -> Self {
        Self {
            kvs: vec![None; num_layers],
            use_kv_cache,
            quantization: KVQuantization::None,
            use_rotor: false,
        }
    }

    pub fn with_quantization(mut self, q: KVQuantization) -> Self {
        self.quantization = q;
        self
    }
    
    pub fn with_rotor(mut self, use_rotor: bool) -> Self {
         self.use_rotor = use_rotor;
         self
    }

    pub fn append_and_fetch(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if !self.use_kv_cache {
            return Ok((k.clone(), v.clone()));
        }

        let dtype = k.dtype(); // We'll restore to this type

        // Apply rotor to the new segment
        let (k_new, v_new) = if self.use_rotor {
            (apply_planar_rotation(k)?, apply_planar_rotation(v)?)
        } else {
            (k.clone(), v.clone())
        };

        let (k_full, v_full) = match &self.kvs[layer_idx] {
            Some(state) => match state {
                CacheState::Full(prev_k, prev_v) => {
                    let k_f = Tensor::cat(&[prev_k, &k_new], 2)?;
                    let v_f = Tensor::cat(&[prev_v, &v_new], 2)?;
                    (k_f, v_f)
                }
                CacheState::Quantized { k_data, k_scale, v_data, v_scale } => {
                    let prev_k = match self.quantization {
                        KVQuantization::Q4_0 => dequantize_4bit(k_data, k_scale, dtype)?,
                        KVQuantization::Q8_0 => dequantize_8bit(k_data, k_scale, dtype)?,
                        KVQuantization::None => unreachable!(),
                    };
                    let prev_v = match self.quantization {
                        KVQuantization::Q4_0 => dequantize_4bit(v_data, v_scale, dtype)?,
                        KVQuantization::Q8_0 => dequantize_8bit(v_data, v_scale, dtype)?,
                        KVQuantization::None => unreachable!(),
                    };
                    
                    let k_f = Tensor::cat(&[&prev_k, &k_new], 2)?;
                    let v_f = Tensor::cat(&[&prev_v, &v_new], 2)?;
                    (k_f, v_f)
                }
            },
            None => (k_new, v_new),
        };

        // Quantize and Store for next turn
        let next_state = match self.quantization {
            KVQuantization::None => CacheState::Full(k_full.clone(), v_full.clone()),
            KVQuantization::Q4_0 => {
                let (k_data, k_scale) = quantize_4bit(&k_full)?;
                let (v_data, v_scale) = quantize_4bit(&v_full)?;
                CacheState::Quantized { k_data, k_scale, v_data, v_scale }
            }
            KVQuantization::Q8_0 => {
                let (k_data, k_scale) = quantize_8bit(&k_full)?;
                let (v_data, v_scale) = quantize_8bit(&v_full)?;
                CacheState::Quantized { k_data, k_scale, v_data, v_scale }
            }
        };

        self.kvs[layer_idx] = Some(next_state);

        // Before returning for Attention, we MUST reverse the planar rotation.
        // It has been applied to k_full and v_full.
        if self.use_rotor {
            Ok((apply_inverse_planar_rotation(&k_full)?, apply_inverse_planar_rotation(&v_full)?))
        } else {
            Ok((k_full, v_full))
        }
    }
}
