use candle_core::{Result, Tensor};

/// A trait for operations that can be dispatched to different backends (Triton/CUDA, Metal, CPU).
pub trait WickOp {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn backward(&self, grad: &Tensor) -> Result<Tensor>;
}
