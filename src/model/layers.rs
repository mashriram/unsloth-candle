use candle_core::{Result, Tensor, Module};
use candle_nn::Linear;

#[derive(Clone)]
pub enum AdapterLayer {
    Linear(Linear),
    LoRA(LoRALinear),
    DoRA(DoRALinear),
}

impl Module for AdapterLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(xs),
            Self::LoRA(l) => l.forward(xs),
            Self::DoRA(l) => l.forward(xs),
        }
    }
}

#[derive(Clone)]
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Tensor, // r x in_dim
    pub lora_b: Tensor, // out_dim x r
    pub scaling: f64,
}

impl LoRALinear {
    pub fn new(base: Linear, lora_a: Tensor, lora_b: Tensor, scaling: f64) -> Self {
        Self { base, lora_a, lora_b, scaling }
    }
}

impl Module for LoRALinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // y = base(x) + (x @ A.T @ B.T) * scaling
        // y = base(x) + (x @ A.T @ B.T) * scaling
        let base_out = self.base.forward(xs)?;
        
        let lora_a_t = self.lora_a.t()?.contiguous()?;
        let lora_a_out = xs.broadcast_matmul(&lora_a_t)?;
        
        let lora_b_t = self.lora_b.t()?.contiguous()?;
        let lora_out = lora_a_out.broadcast_matmul(&lora_b_t)?;
                         
        let scaled = (lora_out * self.scaling)?;
        
        base_out + scaled
    }
}

#[derive(Clone)]
pub struct DoRALinear {
    pub base: Linear,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub m: Tensor, // Magnitude vector [out_dim]
    pub scaling: f64,
}

impl DoRALinear {
    pub fn new(base: Linear, lora_a: Tensor, lora_b: Tensor, m: Tensor, scaling: f64) -> Self {
        Self { base, lora_a, lora_b, m, scaling }
    }
}

impl Module for DoRALinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // DoRA:
        // V = W0 + s * B @ A
        // W' = m * (V / ||V||_c)
        // y = x @ W'^T
        
        // 1. Calculate V
        // base.weight(): [out, in]
        let w0 = self.base.weight();
        
        // lora_update = s * B @ A
        // lora_b: [out, r]
        // lora_a: [r, in]
        // update = lora_b @ lora_a
        let lora_update = self.lora_b.matmul(&self.lora_a)?;
        let lora_update = (lora_update * self.scaling)?;
        
        let v = (w0 + lora_update)?;
        
        // 2. Normalize V (column-wise norm of V, which corresponds to row-wise norm of Weight matrix in PyTorch convention [out, in])
        // We want norm of each output neuron's weight vector.
        // w0 shape is [out, in]. Rows are output units.
        // So we take norm along dim 1.
        let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?; // [out, 1]
        
        // 3. Scale by m
        // m is [out]. Reshape to [out, 1]
        let m = self.m.reshape(((), 1))?;
        
        // W' = m * (V / v_norm)
        let w_prime = v.broadcast_div(&v_norm)?.broadcast_mul(&m)?;
        
        // 4. Linear projection
        // y = x @ W'^T
        let w_prime_t = w_prime.t()?.contiguous()?;
        xs.broadcast_matmul(&w_prime_t)
    }
}
