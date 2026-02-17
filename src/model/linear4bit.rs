use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;

// Local kernel import
use crate::kernels::quantization::dequantize_nf4;

/// Linear4bit Layer
///
/// Stores weights in 4-bit NF4 format (packed u8).
/// Dequantizes on-the-fly during forward pass.
#[derive(Clone)]
pub struct Linear4bit {
    pub weight_packed: Tensor, // [out_features, in_features / 2] (u8)
    pub scales: Tensor,        // [out_features, in_features / block_size] (f32/bf16)
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub block_size: usize,
}

impl Linear4bit {
    pub fn new(
        weight_packed: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
        block_size: usize,
    ) -> Self {
        Self {
            weight_packed,
            scales,
            bias,
            in_features,
            out_features,
            block_size,
        }
    }
    
    pub fn dequantize(&self) -> Result<Tensor> {
        dequantize_nf4(
            &self.weight_packed,
            &self.scales,
            self.out_features,
            self.in_features,
            self.block_size
        )
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Dequantize weights
        let weight = self.dequantize()?;
        
        // 2. Matmul
        // x: [batch, seq, in_features]
        // weight: [out_features, in_features]
        // output: [batch, seq, out_features]
        
        let x = x.matmul(&weight.t()?)?;
        
        // 3. Bias
        match &self.bias {
            Some(bias) => x.broadcast_add(bias),
            None => Ok(x),
        }
    }

    pub fn from_tensor(weight: &Tensor, block_size: usize) -> Result<Self> {
        // Naive CPU quantization (F32 -> NF4)
        // 1. Get data as Vec<f32>
        // 2. Process in blocks
        // 3. Compute scales
        // 4. Quantize to u8
        
        let device = weight.device();
        let (out_features, in_features) = weight.dims2()?;
        
        // Move to CPU for processing (unless we write a GPU kernel for this later)
        let weight_cpu = weight.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?;
        let weight_data = weight_cpu.flatten_all()?.to_vec1::<f32>()?;
        
        let num_blocks = (out_features * in_features).div_ceil(block_size);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut packed = Vec::with_capacity((out_features * in_features) / 2);
        
        // NF4 values sorted
        let nf4_values = [
            -1.00000000f32, -0.69619280, -0.52507305, -0.39491749, 
            -0.28444138, -0.18477343, -0.09105004, 0.00000000, 
            0.07958030, 0.16093020, 0.24611230, 0.33791524, 
            0.44070983, 0.56261700, 0.72295684, 1.00000000
        ];
        
        for chunk in weight_data.chunks(block_size) {
            // 1. Find absmax
            let mut max_val = 0.0f32;
            for &x in chunk {
                let abs = x.abs();
                if abs > max_val { max_val = abs; }
            }
            scales.push(max_val);
            
            // 2. Quantize
            let packed_chunk_capacity = chunk.len() / 2;
            // Iterate pairs
            for i in 0..packed_chunk_capacity {
                let v0 = chunk[2*i];
                let v1 = chunk[2*i + 1];
                
                let q0 = quantize_closest(v0, max_val, &nf4_values);
                let q1 = quantize_closest(v1, max_val, &nf4_values);
                
                // Pack: usually [q1 << 4 | q0] or similar.
                // Bitsandbytes: (v0 & 0x0F) | ((v1 & 0x0F) << 4)
                let b = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                packed.push(b);
            }
        }
        
        let packed_tensor = Tensor::from_vec(packed, (out_features, in_features / 2), device)?;
        let scales_tensor = Tensor::from_vec(scales, (out_features, in_features / block_size), device)?; 
        
        Ok(Self::new(
            packed_tensor,
            scales_tensor,
            None, // Bias usually separate
            in_features,
            out_features,
            block_size
        ))
    }
}

fn quantize_closest(v: f32, absmax: f32, nf4_values: &[f32]) -> u8 {
    if absmax == 0.0 { return 7; } // map to 0.0
    let normalized = v / absmax;
    // Find index of nearest value in nf4_values
    let mut min_err = f32::MAX;
    let mut best_idx = 0;
    for (i, &nf4) in nf4_values.iter().enumerate() {
        let err = (normalized - nf4).abs();
        if err < min_err {
            min_err = err;
            best_idx = i;
        }
    }
    best_idx as u8
}

// Ensure AdapterLayer can hold Linear4bit?
// AdapterLayer enum is in `layers.rs`. We need to update it.
