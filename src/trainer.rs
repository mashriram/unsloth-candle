use candle_core::{Result, Tensor};
use candle_nn::Optimizer;
use crate::{model::RustModel, core::optimizer::AdamW};
use std::sync::{Arc, Mutex};

pub struct Trainer {
    pub model: Arc<Mutex<RustModel>>,
    optimizer: Option<AdamW>,
}

impl Trainer {
    pub fn new(model: Arc<Mutex<RustModel>>) -> Self {
        Self {
            model,
            optimizer: None,
        }
    }

    pub fn configure_optimizer(&mut self, learning_rate: f64) -> Result<()> {
        // Collect variables from model
        let model = self.model.lock().unwrap();
        let vars = model.varmap().all_vars();
        if vars.is_empty() {
             // If empty, maybe we are fine-tuning the whole model?
             // But the model is loaded as Tensors.
             // We can't easily convert Tensor to Var in-place without reloading logic.
             // For the vertical slice of "Trainer", we should add at least one dummy Var to provenance.
             // Or verify that we can act on `varmap` if populated.
             println!("Warning: No trainable variables found in VarMap. Optimizer will be empty.");
        }
        
        let optimizer = AdamW::new(vars, learning_rate)?;
        self.optimizer = Some(optimizer);
        Ok(())
    }
    
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        if let Some(opt) = &mut self.optimizer {
            opt.step(grads)
        } else {
            Ok(())
        }
    }
    
    // Real training step
    pub fn train_step(&mut self, input: &Tensor, labels: &Tensor) -> Result<f32> {
        let loss = {
            let mut model = self.model.lock().unwrap();
            model.clear_cache();
            let logits = model.forward(input, None, 0)?; // [batch, seq, vocab]
            
            // Shift so that tokens < n predict n
            let (_b, seq_len, _vocab) = logits.dims3()?;
            
            // logits should be [..., :-1, :]
            // labels should be [..., 1:]
            
            let logits = logits.narrow(1, 0, seq_len - 1)?;
            let labels = labels.narrow(1, 1, seq_len - 1)?;
            
            // Flatten for cross_entropy: [batch * (seq-1), vocab]
            let (b, s, vocab) = logits.dims3()?;
            let logits = logits.reshape((b * s, vocab))?;
            let labels = labels.reshape((b * s,))?;
            
            candle_nn::loss::cross_entropy(&logits, &labels)?
        };
        
        let mut grads = loss.backward()?;
        self.step(&grads)?;
        
        let loss_scalar = loss.to_vec0::<f32>()?;
        Ok(loss_scalar)
    }
}
