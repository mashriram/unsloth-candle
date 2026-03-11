// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Quantization kernels (NF4, etc.)

use crate::error::{Result as UnslothResult, UnslothError};
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;
#[cfg(feature = "cuda")]
use cubecl_cuda::CudaRuntime;

/// NF4 Dequantization Kernel
///
/// Dequantizes packed 4-bit NormalFloat (NF4) weights into floating point (F16/BF16/F32).
///
/// # Arguments
/// * `weight_packed`: [N, K/2] Packed u8 tensor (2 x 4-bit weights per u8)
/// * `scales`: [N, K/block_size] Scaling factors (blocks)
/// * `output`: [N, K] Dequantized weights
/// * `absmax`: Absolute maximum value for NF4 normalization (usually 1.0 or derived)
/// * `block_size`: Quantization block size (e.g., 64)
#[cfg(feature = "cuda")]
#[cube(launch)]
pub fn dequantize_nf4_kernel<F: Float + CubeElement>(
    weight_packed: &Array<u8>,
    scales: &Array<F>,
    output: &mut Array<F>,
    num_rows: u32,
    num_cols: u32,
    block_size: u32,
) {
    // Grid: (num_rows, num_cols / block_size(?), 1) or similar.
    // Let's assume 1 thread per output element or 1 thread per packed byte (2 elements).
    // Easiest: 1 thread per 2 elements (1 byte).
    
    let global_idx = CUBE_POS_X * CUBE_BLOCK_SIZE_X + UNIT_POS_X;
    let total_elements = num_rows * num_cols;
    let total_packed = total_elements / 2;
    
    if global_idx < total_packed {
        // Each thread processes 1 packed byte -> 2 output weights.
        let packed_val = weight_packed[global_idx];
        
        // Extract 4-bit values
        // Lower 4 bits = weight 0? Or upper?
        // Usually: [w1, w0] or [w0, w1].
        // bitsandbytes uses: val & 0x0F is first, val >> 4 is second.
        let w0_idx = packed_val & 0x0F;
        let w1_idx = (packed_val >> 4) & 0x0F;
        
        // Lookup NF4 value from LUT (constant array in kernel?)
        // CubeCL doesn't easily support const arrays yet inside cube function unless passed or hardcoded match.
        // Hardcoded match is faster for small LUT.
        let w0_val = get_nf4_value(w0_idx);
        let w1_val = get_nf4_value(w1_idx);
        
        // Determine output indices
        let out_idx0 = global_idx * 2;
        let out_idx1 = out_idx0 + 1;
        
        // Determine scale index
        // Scale is per block.
        // Block index = out_idx / block_size
        let block_idx0 = out_idx0 / block_size;
        let block_idx1 = out_idx1 / block_size;
        
        let scale0 = scales[block_idx0];
        let scale1 = scales[block_idx1]; // Likely same as scale0 if block_size >= 2
        
        // Dequantize: value * scale
        output[out_idx0] = w0_val * scale0;
        output[out_idx1] = w1_val * scale1;
    }
}

/// Helper to map 4-bit index to NF4 value
#[cfg(feature = "cuda")]
#[cube]
fn get_nf4_value<F: Float + CubeElement>(index: u8) -> F {
    // NF4 values (normalized to [-1, 1] range approximately)
    // These are standard bitsandbytes NF4 values.
    // offsets:
    // 0: -1.0
    // 1: -0.6961928009986877
    // 2: -0.5250730514526367
    // ...
    // writing manual match:
    
    let val = if index == 0 { -1.00000000f32 }
    else if index == 1 { -0.69619280f32 }
    else if index == 2 { -0.52507305f32 }
    else if index == 3 { -0.39491749f32 }
    else if index == 4 { -0.28444138f32 }
    else if index == 5 { -0.18477343f32 }
    else if index == 6 { -0.09105004f32 }
    else if index == 7 {  0.00000000f32 }
    else if index == 8 {  0.07958030f32 }
    else if index == 9 {  0.16093020f32 }
    else if index == 10 { 0.24611230f32 }
    else if index == 11 { 0.33791524f32 }
    else if index == 12 { 0.44070983f32 }
    else if index == 13 { 0.56261700f32 }
    else if index == 14 { 0.72295684f32 }
    else {                1.00000000f32 }; // index 15
    
    F::cast_from(val)
}

/// CPU dispatch for dequantize_nf4
#[cfg(not(feature = "cuda"))]
pub fn dequantize_nf4(
    weight_packed: &Tensor,
    scales: &Tensor,
    num_rows: usize,
    num_cols: usize,
    block_size: usize,
) -> UnslothResult<Tensor> {
    // Fallback CPU implementation or error
    // For now, implementing a slow CPU loop is better than crashing, but 'unsloth-rs' focuses on GPU.
    // But let's verify if we want to support CPU qlora.
    // bitsandbytes is CUDA only largely.
    // candle-core has a from_slice?
    // Let's return error for now as this is "Advanced Fused Kernels" mostly.
    Err(UnslothError::CpuFallback("dequantize_nf4".to_string()))
}

#[cfg(feature = "cuda")]
pub fn dequantize_nf4(
    weight_packed: &Tensor,
    scales: &Tensor,
    num_rows: usize,
    num_cols: usize,
    block_size: usize,
) -> UnslothResult<Tensor> {
    let dev = weight_packed.device();
    match dev {
        candle_core::Device::Cuda(_) => {
            // Launch CubeCL kernel
            // 1. Prepare inputs (get GPU slices) - this part is tricky cleanly with Candle <-> CubeCL without `cudarc`.
            // However, `unsloth-rs` seems to assume it can launch via `cubecl-cuda` runtime.
            // We need to construct CubeCL handles.
            // Refer to fused_rmsnorm_rope.rs for the dispatch pattern.
            
            // For now, stub to indicate structure.
            // We need to implement the actual launch glue code which is usually boilerplate.
            // Assuming we ignore the glue for a second and just focus on logic correctness.
            
            // To properly launch, we need:
            // let client = CudaRuntime::client(dev.ordinal()?);
            // let w_handle = ...
            // let out_handle = ...
            // dequantize_nf4_kernel::launch(...)
            
            // Since I cannot easily replicate the full runtime binding here without seeing `rope.rs` glue,
            // I will look at `fused_rmsnorm_rope.rs` launch code again.
            Err(UnslothError::Custom("Not yet wired to runtime".to_string()))
        }
        _ => Err(UnslothError::CpuFallback("dequantize_nf4 matches cuda feature but device is not cuda".to_string()))
    }
}
