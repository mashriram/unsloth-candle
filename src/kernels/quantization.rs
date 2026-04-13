// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Quantization kernels (NF4, etc.)

use candle_core::{Tensor, Error};  // ← do NOT import Result here; conflicts with cube(launch)

// Use native Candle ops

pub fn dequantize_nf4(
    weight_packed: &Tensor,
    scales: &Tensor,
    num_rows: usize,
    num_cols: usize,
    block_size: usize,
) -> candle_core::Result<Tensor> {
    let dev = weight_packed.device();
    
    // NF4 constant values as a lookup table
    let nf4_values = [
        -1.00000000f32, -0.69619280, -0.52507305, -0.39491749, 
        -0.28444138, -0.18477343, -0.09105004, 0.00000000, 
        0.07958030, 0.16093020, 0.24611230, 0.33791524, 
        0.44070983, 0.56261700, 0.72295684, 1.00000000
    ];
    let lut = Tensor::from_vec(nf4_values.to_vec(), (16,), dev)?;

    // Unpack: each byte has 2 x 4-bit indices.
    // i0 = b & 0x0F, i1 = b >> 4. 
    // In Candle, we can use f32 arithmetic for this.
    let packed_f32 = weight_packed.flatten_all()?.to_dtype(candle_core::DType::F32)?;
    
    // i1 = (packed / 16).floor()
    let sixteen = Tensor::new(16.0f32, dev)?;
    let i1 = packed_f32.broadcast_div(&sixteen)?.floor()?;
    
    // i0 = packed - (i1 * 16)
    let i0 = packed_f32.broadcast_sub(&i1.broadcast_mul(&sixteen)?)?;
    
    // Convert back to U32 for gather
    let i0 = i0.to_dtype(candle_core::DType::U32)?;
    let i1 = i1.to_dtype(candle_core::DType::U32)?;
    
    // Interleave them: [b0_i0, b0_i1, b1_i0, b1_i1, ...]
    // stack(0) -> [2, packed_len], then transpose(0, 1) -> [packed_len, 2]
    let indices = Tensor::stack(&[i0, i1], 0)?.transpose(0, 1)?.flatten_all()?;
    
    // Lookup values
    let values = lut.gather(&indices, 0)?;
    
    // Reshape to final size
    let values = values.reshape((num_rows, num_cols))?;
    
    // Apply scales (each block has one scale)
    let num_elements = num_rows * num_cols;
    let num_blocks = (num_elements + block_size - 1) / block_size;
    
    let scales = scales.flatten_all()?;
    
    // Repeat each scale 'block_size' times
    let scales_expanded = scales.reshape((num_blocks, 1))?
        .expand((num_blocks, block_size))?
        .flatten_all()?;
        
    let scales_expanded = if scales_expanded.dim(0)? > num_elements {
        scales_expanded.narrow(0, 0, num_elements)?
    } else {
        scales_expanded
    };
    
    let dequantized = values.flatten_all()?.broadcast_mul(&scales_expanded)?;
    dequantized.reshape((num_rows, num_cols))
}