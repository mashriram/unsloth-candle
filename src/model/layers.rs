use candle_core::{Result, Tensor, Module};
use candle_nn::Linear;
use crate::model::linear4bit::Linear4bit;

#[derive(Clone)]
pub enum AdapterLayer {
    Linear(Linear),
    Linear4bit(Linear4bit),
    LoRA(LoRALinear),
    DoRA(DoRALinear),
}

impl Module for AdapterLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(xs),
            Self::Linear4bit(l) => l.forward(xs),
            Self::LoRA(l) => l.forward(xs),
            Self::DoRA(l) => l.forward(xs),
        }
    }
}

impl AdapterLayer {
    pub fn get_weight_f32(&self) -> Result<Tensor> {
        match self {
            Self::Linear(l) => Ok(l.weight().clone()),
            Self::Linear4bit(l) => {
                // Dequantize logic duplicated from forward, or expose dequantize method on Linear4bit?
                // Better to expose dequantize method.
                l.dequantize()
            },
            _ => Err(candle_core::Error::Msg("Unsupported base layer for DoRA weight access".to_string())),
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        match self {
            Self::Linear(l) => l.bias(),
            Self::Linear4bit(l) => l.bias.as_ref(),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct LoRALinear {
    pub base: Box<AdapterLayer>,
    pub lora_a: Tensor, // r x in_dim
    pub lora_b: Tensor, // out_dim x r
    pub scaling: f64,
}

impl LoRALinear {
    pub fn new(base: AdapterLayer, lora_a: Tensor, lora_b: Tensor, scaling: f64) -> Self {
        Self { base: Box::new(base), lora_a, lora_b, scaling }
    }
}

impl Module for LoRALinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
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
    pub base: Box<AdapterLayer>,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub m: Tensor, // Magnitude vector [out_dim]
    pub scaling: f64,
}

impl DoRALinear {
    pub fn new(base: AdapterLayer, lora_a: Tensor, lora_b: Tensor, m: Tensor, scaling: f64) -> Self {
        Self { base: Box::new(base), lora_a, lora_b, m, scaling }
    }
}

impl Module for DoRALinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // DoRA:
        // V = W0 + s * B @ A
        // W' = m * (V / ||V||_c)
        // y = x @ W'^T
        
        // 1. Calculate V
        // We need base weights.
        // If base is Linear, we have .weight().
        // If base is Linear4bit, we have .weight_packed and need to dequantize.
        // If base is LoRA/DoRA... wait, recursive? Usually not.
        
        // Problem: DoRA needs the full weight matrix W0 to compute V = W0 + BA.
        // If W0 is 4-bit, we must dequantize it here to add BA.
        // We need a helper on AdapterLayer to get "unquantized_weight()".
        
        // For now, let's implement `weight()` on AdapterLayer?
        let w0 = self.base.get_weight_f32()?; 
        
        // lora_update = s * B @ A
        let lora_update = self.lora_b.matmul(&self.lora_a)?;
        let lora_update = (lora_update * self.scaling)?;
        
        let v = (w0 + lora_update)?;
        
        // 2. Normalize V (column-wise norm of V, which corresponds to row-wise norm of Weight matrix in PyTorch convention [out, in])
        let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?; // [out, 1]
        
        // 3. Scale by m
        let m = self.m.reshape(((), 1))?;
        let weight_prime = v.broadcast_mul(&m)?.broadcast_div(&v_norm)?;
        
        // 4. Forward with new weight
        let x = xs.matmul(&weight_prime.t()?)?;
        
        // 5. Add bias if exists
        // Where is bias? in base layer.
        // We need to access base bias.
        if let Some(bias) = self.base.bias() {
             x.broadcast_add(bias)
        } else {
             Ok(x)
        }
    }
}

#[derive(Clone)]
pub struct UnslothRmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl UnslothRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        unsloth_rs::kernels::rmsnorm_cubecl(x, &self.weight, self.eps)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}
