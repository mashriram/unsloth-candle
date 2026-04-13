// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Quantization kernels (NF4, etc.)

use crate::error::{Result as UnslothResult, UnslothError};
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

#[cfg(feature = "cuda")]
#[cube]
pub fn dequantize_nf4_kernel(
    weight_packed: &Array<u8>,
    scales: &Array<f32>,
    output: &mut Array<f32>,
    num_rows: u32,
    num_cols: u32,
    block_size: u32,
) {
    let global_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let total_elements = num_rows * num_cols;
    let total_packed = total_elements / 2u32;

    if global_idx < total_packed {
        let packed_val = weight_packed[global_idx as usize];  // fix 1

        let w0_idx = (packed_val & 0x0Fu8) as u32;
        let w1_idx = ((packed_val >> 4u8) & 0x0Fu8) as u32;

        let w0_val = get_nf4_value(w0_idx);
        let w1_val = get_nf4_value(w1_idx);

        let out_idx0 = global_idx * 2u32;
        let out_idx1 = out_idx0 + 1u32;

        let block_idx0 = out_idx0 / block_size;
        let block_idx1 = out_idx1 / block_size;

        let scale0 = scales[block_idx0 as usize];  // fix 1
        let scale1 = scales[block_idx1 as usize];  // fix 1

        output[out_idx0 as usize] = w0_val * scale0;  // fix 1
        output[out_idx1 as usize] = w1_val * scale1;  // fix 1
    }
}

#[cfg(feature = "cuda")]
#[cube]
fn get_nf4_value(index: u32) -> f32 {
    let mut out: f32 = 0.0_f32;
    if index == 0u32        { out = -1.00000000_f32; }
    else if index == 1u32   { out = -0.69619280_f32; }
    else if index == 2u32   { out = -0.52507305_f32; }
    else if index == 3u32   { out = -0.39491749_f32; }
    else if index == 4u32   { out = -0.28444138_f32; }
    else if index == 5u32   { out = -0.18477343_f32; }
    else if index == 6u32   { out = -0.09105004_f32; }
    else if index == 7u32   { out =  0.00000000_f32; }
    else if index == 8u32   { out =  0.07958030_f32; }
    else if index == 9u32   { out =  0.16093020_f32; }
    else if index == 10u32  { out =  0.24611230_f32; }
    else if index == 11u32  { out =  0.33791524_f32; }
    else if index == 12u32  { out =  0.44070983_f32; }
    else if index == 13u32  { out =  0.56261700_f32; }
    else if index == 14u32  { out =  0.72295684_f32; }
    else                    { out =  1.00000000_f32; }
    out
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
            // Placeholder: Not yet wired to cubecl/cuda runtime for the actual kernel launch
            Err(UnslothError::Custom("CubeCL kernel for NF4 not yet wired for launch".into()))
        }
        _ => {
            // CPU Fallback implementation
            println!("Warning: dequantize_nf4 falling back to CPU implementation");
            cpu_dequantize_nf4(weight_packed, scales, num_rows, num_cols, block_size)
        }
    }
}

pub fn cpu_dequantize_nf4(
    weight_packed: &Tensor,
    scales: &Tensor,
    num_rows: usize,
    num_cols: usize,
    block_size: usize,
) -> UnslothResult<Tensor> {
    let device = weight_packed.device();
    let packed_data = weight_packed.flatten_all()
        .map_err(UnslothError::from)?
        .to_vec1::<u8>()
        .map_err(UnslothError::from)?;
        
    let scales_data = scales.flatten_all()
        .map_err(UnslothError::from)?
        .to_vec1::<f32>()
        .map_err(UnslothError::from)?;

    let nf4_values = [
        -1.00000000f32, -0.69619280, -0.52507305, -0.39491749, 
        -0.28444138, -0.18477343, -0.09105004, 0.00000000, 
        0.07958030, 0.16093020, 0.24611230, 0.33791524, 
        0.44070983, 0.56261700, 0.72295684, 1.00000000
    ];

    let mut out_data = vec![0.0f32; num_rows * num_cols];

    for i in 0..packed_data.len() {
        let b = packed_data[i];
        let i0 = (b & 0x0F) as usize;
        let i1 = (b >> 4) as usize;

        let v0 = nf4_values[i0];
        let v1 = nf4_values[i1];

        let out_idx0 = i * 2;
        let out_idx1 = i * 2 + 1;

        if out_idx0 < out_data.len() {
            let s_idx = out_idx0 / block_size;
            out_data[out_idx0] = v0 * scales_data[s_idx];
        }
        if out_idx1 < out_data.len() {
            let s_idx = out_idx1 / block_size;
            out_data[out_idx1] = v1 * scales_data[s_idx];
        }
    }

    Tensor::from_vec(out_data, (num_rows, num_cols), device)
        .map_err(UnslothError::from)
}

#[cfg(not(feature = "cuda"))]
pub fn dequantize_nf4(
    weight_packed: &Tensor,
    scales: &Tensor,
    num_rows: usize,
    num_cols: usize,
    block_size: usize,
) -> UnslothResult<Tensor> {
    cpu_dequantize_nf4(weight_packed, scales, num_rows, num_cols, block_size)
}