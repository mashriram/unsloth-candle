use candle_core::{Result, Tensor, Var};
use candle_nn::{Optimizer, ParamsAdamW};

pub struct AdamW {
    inner: candle_nn::AdamW,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        let inner = candle_nn::AdamW::new(vars, params)?;
        Ok(Self { inner })
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.inner.step(grads)
    }
}
