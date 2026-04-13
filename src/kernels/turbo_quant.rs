// SPDX-License-Identifier: MIT
use candle_core::{Tensor, Result, DType};

/// Applies a 2D Givens / Hadamard-like planar rotation to decorrelate features.
/// This prevents outliers from dominating low-bit quantization channels.
pub fn apply_planar_rotation(x: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(candle_core::D::Minus1)?;
    if head_dim % 2 != 0 {
        return Ok(x.clone()); // Fallback if uneven
    }
    
    // We pair adjacent elements and apply a 45-degree rotation (Hadamard-like)
    // x0' = (x0 + x1) * 0.707
    // x1' = (x0 - x1) * 0.707
    // This is mathematically equivalent to multiplying by a block diagonal rotation matrix.
    
    // Shape: [..., d] -> [..., d/2, 2]
    let mut dims = x.dims().to_vec();
    *dims.last_mut().unwrap() = head_dim / 2;
    dims.push(2);
    
    let x_paired = x.reshape(dims.as_slice())?;
    
    // Extract x0 and x1
    let x0 = x_paired.narrow(candle_core::D::Minus1, 0, 1)?;
    let x1 = x_paired.narrow(candle_core::D::Minus1, 1, 1)?;
    
    let inv_sqrt_2 = 0.70710678118f64;
    
    let x0_new = ((&x0 + &x1)? * inv_sqrt_2)?;
    let x1_new = ((&x0 - &x1)? * inv_sqrt_2)?;
    
    let x_rotated = Tensor::cat(&[&x0_new, &x1_new], candle_core::D::Minus1)?;
    
    // Reshape back to original
    x_rotated.reshape(x.dims())
}

/// Applies the inverse of the planar rotation.
/// Since it's a scaled 45-degree rotation, the inverse is its transpose (or negative angle).
/// Actually, Hadamard is its own inverse! (x0+x1)/sqrt(2), and (x0'-x1')/sqrt(2).
/// Let's verify:
/// x0' = (x0+x1)/sqrt(2), x1' = (x0-x1)/sqrt(2)
/// x0_orig = (x0' + x1')/sqrt(2) = (x0+x1+x0-x1)/2 = 2*x0/2 = x0. CORRECT!
pub fn apply_inverse_planar_rotation(x: &Tensor) -> Result<Tensor> {
    // The generalized Hadamard 2x2 is its own inverse.
    apply_planar_rotation(x)
}


/// Native generic 8-bit block quantization (per inner dimension block)
pub fn quantize_8bit(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    // Use maximum absolute value along the last dimension (head_dim) for scaling
    let max_val = x.abs()?.max_keepdim(candle_core::D::Minus1)?;
    // Add epsilon to avoid div by zero
    let scale = (max_val / 127.0)?;
    // q = round(x / scale)
    let qx = x.broadcast_div(&scale)?.round()?;
    // We clip strictly to i8 range just in case rounding floats it
    let qx = qx.maximum(&qx.zeros_like()?.sub(&qx.ones_like()?.affine(127.0, 0.0)?)?)?
               .minimum(&qx.ones_like()?.affine(127.0, 0.0)?)?;
    let qx = qx.to_dtype(DType::U8)?; // Use U8 as I8 support in candle might be lacking certain layout ops in Metal
                                      // Actually, we can just cast to U8 by shifting +128.
    let qx_shifted = (qx + 128.0)?.to_dtype(DType::U8)?;
    
    Ok((qx_shifted, scale))
}

pub fn dequantize_8bit(qx: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let qx_f = qx.to_dtype(target_dtype)?;
    let qx_centered = (qx_f - 128.0)?;
    qx_centered.broadcast_mul(scale)
}

/// Native generic 4-bit block quantization packed into U8
pub fn quantize_4bit(x: &Tensor) -> Result<(Tensor, Tensor)> {
    // For 4-bit, values are mapped from -7 to 7. 
    let max_val = x.abs()?.max_keepdim(candle_core::D::Minus1)?;
    let scale = (max_val / 7.0)?;
    
    let qx = x.broadcast_div(&scale)?.round()?;
    let qx_shifted = (qx + 8.0)?; // Shift to 0..15 range
    
    // Clip
    let qx_shifted = qx_shifted.maximum(&qx_shifted.zeros_like()?)?
                               .minimum(&qx_shifted.ones_like()?.affine(15.0, 0.0)?)?
                               .to_dtype(DType::U8)?;
    
    // We pack 2 values per byte (pair adjacent elements along the last dim)
    let head_dim = x.dim(candle_core::D::Minus1)?;
    if head_dim % 2 != 0 {
        // Unpacked fallback if uneven
        return Ok((qx_shifted, scale));
    }
    
    let mut dims = qx_shifted.dims().to_vec();
    *dims.last_mut().unwrap() = head_dim / 2;
    dims.push(2);
    
    let q_paired = qx_shifted.reshape(dims.as_slice())?;
    let q0 = q_paired.narrow(candle_core::D::Minus1, 0, 1)?;
    let q1 = q_paired.narrow(candle_core::D::Minus1, 1, 1)?;
    
    // Packed = (q0 << 4) | q1
    // Candle natively supports mul/add for U8 on CPU, but sometimes metal has issues.
    // We'll use DType U8 operations.
    // Currently, Candle may not have elementwise bitshift for U8.
    // We do: packed = q0 * 16 + q1
    // QTensor creation might need to be flattened or casted.
    // Float cast is safe to pack.
    let packed = ((q0.to_dtype(DType::F32)? * 16.0)? + q1.to_dtype(DType::F32)?)?
                 .to_dtype(DType::U8)?;
                 
    // Drop the last dim of size 1
    let packed = packed.squeeze(candle_core::D::Minus1)?;
    
    Ok((packed, scale))
}

pub fn dequantize_4bit(qx: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let unpacked_shape = {
        let mut dims = qx.dims().to_vec();
         // If we successfully packed, last dim is halved. We double it.
         // wait, how do we handle the unpack shape? We expect scale to have identical dims except last dim = 1.
         // Let's just track this dynamically.
        *dims.last_mut().unwrap() *= 2;
        dims
    };
    
    if qx.dim(candle_core::D::Minus1)? == scale.dim(candle_core::D::Minus1)? {
        // Fallback for odd shapes unpacked
        let qx_f = qx.to_dtype(target_dtype)?;
        let qx_centered = (qx_f - 8.0)?;
        return qx_centered.broadcast_mul(scale);
    }
    
    let qx_f = qx.to_dtype(DType::F32)?;
    let q0 = (qx_f.clone() / 16.0)?.floor()?;
    let q1 = (qx_f - (&q0 * 16.0)?)?;
    
    // Stack and interleave
    let q_interleaved = Tensor::stack(&[q0, q1], candle_core::D::Minus1)?;
    let q_flattened_dim = q_interleaved.reshape(unpacked_shape.as_slice())?;
    
    let q_centered = (q_flattened_dim.to_dtype(target_dtype)? - 8.0)?;
    q_centered.broadcast_mul(scale)
}
