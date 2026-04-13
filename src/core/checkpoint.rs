use candle_core::{Tensor, Result, CustomOp1};
use candle_core::backend::BackendStorage;  // fix: brings .dtype() and .device into scope
use std::sync::Arc;

/// Checkpointing operation that re-computes the forward pass during the backward pass.
pub struct Checkpointing {
    pub forward_fn: Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>,
}

impl CustomOp1 for Checkpointing {
    fn name(&self) -> &'static str {
        "checkpointing"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let device = candle_core::Device::Cpu;
        let shape = l1.shape();

        let x = match s1 {
            candle_core::CpuStorage::F32(v) => Tensor::from_slice(v, shape, &device)?,
            candle_core::CpuStorage::F64(v) => Tensor::from_slice(v, shape, &device)?,
            _ => return Err(candle_core::Error::Msg(
                "Unsupported dtype for checkpointing cpu_fwd".to_string(),
            )),
        };

        let out = (self.forward_fn)(&x)?;

        let (storage, layout) = out.storage_and_layout();
        let storage = match &*storage {
            candle_core::Storage::Cpu(s) => s.clone(),
            _ => return Err(candle_core::Error::Msg(
                "Expected CPU storage in checkpoint cpu_fwd".to_string(),
            )),
        };

        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        // fix 1: s1.device is a field; BackendStorage trait (imported above) exposes .dtype()
        let _device = candle_core::Device::Cuda(s1.device.clone());
        let _shape = l1.shape();
        let _dtype = s1.dtype();  // fix 2: use .dtype() via BackendStorage, not CudaStorage::Wrap

        Err(candle_core::Error::Msg(
            "Checkpointing not fully implemented for CUDA yet (missing from_storage)".to_string(),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<Option<Tensor>> {
        let x_detached = arg.detach();
        let x_var = candle_core::Var::from_tensor(&x_detached)?;
        let x_t = x_var.as_tensor();

        let out = (self.forward_fn)(x_t)?;

        let surrogate = (out * grad_res)?.sum_all()?;
        let grads = surrogate.backward()?;

        let grad_input = grads.get(&x_detached);
        match grad_input {
            Some(g) => Ok(Some(g.clone())),
            None => Ok(None),
        }
    }
}

/// Applies gradient checkpointing to a function `f` with input `x`.
pub fn checkpoint(
    f: Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>,
    x: &Tensor,
) -> Result<Tensor> {
    x.apply_op1(Checkpointing { forward_fn: f })
}