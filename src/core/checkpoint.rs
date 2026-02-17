use candle_core::{Tensor, Result, CustomOp1};
use candle_core::backprop::GradStore;
use std::sync::Arc;

/// Checkpointing operation that re-computes the forward pass during the backward pass.
pub struct Checkpointing {
    pub forward_fn: Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>,
}

impl CustomOp1 for Checkpointing {
    fn name(&self) -> &'static str {
        "checkpointing"
    }

    fn cpu_fwd(&self, s1: &candle_core::CpuStorage, l1: &candle_core::Layout) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // Recover tensor from storage
        // Since `Tensor::from_storage` is not easily available/public, we work around it.
        // We handle F32, F64, BF16, F16.
        let device = candle_core::Device::Cpu;
        let shape = l1.shape();
        
        let x = match s1 {
            candle_core::CpuStorage::F32(v) => Tensor::from_slice(v, shape, &device)?,
            candle_core::CpuStorage::F64(v) => Tensor::from_slice(v, shape, &device)?,
            // TODO: Handle other types
            _ => return Err(candle_core::Error::Msg("Unsupported dtype for checkpointing cpu_fwd".to_string())),
        };
        
        // Run forward function strictly for value (no graph)
        let out = (self.forward_fn)(&x)?;
        
        let (storage, layout) = out.storage_and_layout();
        let storage = match &*storage {
            candle_core::Storage::Cpu(s) => s.clone(),
            _ => return Err(candle_core::Error::Msg("Expected CPU storage in checkpoint cpu_fwd".to_string())),
        };
        
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &candle_core::CudaStorage, l1: &candle_core::Layout) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        // For CUDA, we need to construct a Tensor from CudaStorage without copying to CPU.
        // `candle_core::Tensor` constructors from storage are limited publicly.
        // We can try `Tensor::from_raw_buffer` but that takes &[u8], which we can get from CudaSlice?
        // Note: This effectively copies data on GPU (D2D).
        let device = candle_core::Device::Cuda(s1.device().clone());
        let shape = l1.shape();
        let dtype = match s1 {
            candle_core::CudaStorage::Wrap(w) => w.dtype(),
            _ => candle_core::DType::F32, // Fallback guess?
        };
        
        // This is a blocker if we can't create Tensor from CudaStorage cheaply.
        // For now, we return error to indicate limitation, or use a workaround if found.
        Err(candle_core::Error::Msg("Checkpointing not fully implemented for CUDA yet (missing from_storage)".to_string()))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Backward pass:
        // 1. Re-run forward pass on `arg` (input), but THIS time we want to track it to get a graph.
        let x_detached = arg.detach(); // detach() returns Tensor
        let x_var = candle_core::Var::from_tensor(&x_detached)?;
        let x_t = x_var.as_tensor();
        
        // 2. Run the forward function again
        let out = (self.forward_fn)(x_t)?;
        
        // 3. Propagate gradients
        // We calculate dot product of output and incoming gradient, then backpropagate.
        // Ensure shapes match?
        // If out and grad_res have same shape, we can sum(out * grad_res).
        let surrogate = (out * grad_res)?.sum_all()?;
        let grads = surrogate.backward()?;
        
        // Get gradient for x
        let grad_input = grads.get(&x_detached); // or &x_t? Grads keys are usually Tensors (by ID) directly or TensorId?
        // candle_core::backprop::Grads::get takes &Tensor.
        // The tensor we backpropped *from* is `surrogate`.
        // The tensor we want grad *for* is `x_t` (which is same ID as x_var?).
        // `Var::from_tensor` creates a new variable.
        
        match grad_input {
            Some(g) => Ok(Some(g.clone())),
            None => Ok(None),
        }
    }
}

/// Applies gradient checkpointing to a function `f` with input `x`.
pub fn checkpoint(
    f: Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>, 
    x: &Tensor
) -> Result<Tensor> {
    x.apply_op1(Checkpointing { forward_fn: f })
}
