use candle_core::{Result, Tensor};
use candle_nn::Optimizer;
use crate::core::optimizer::AdamW;
use std::sync::{Arc, Mutex};

/// Internal model state shared between FastLanguageModel and Trainer.
/// Re-exported here to avoid circular deps.
pub struct ModelState {
    pub model: crate::model::RustModel,
    pub config_json: serde_json::Value,
    pub model_name: String,
    pub model_dir: std::path::PathBuf,  // Path to cached HF model dir (for save/merge)
}

pub struct Trainer {
    pub state: Arc<Mutex<ModelState>>,
    optimizer: Option<AdamW>,
}

impl Trainer {
    pub fn new_from_state(state: Arc<Mutex<ModelState>>) -> Self {
        Self { state, optimizer: None }
    }

    pub fn configure_optimizer(&mut self, learning_rate: f64) -> Result<()> {
        let state = self.state.lock().unwrap();
        let vars = state.model.varmap().all_vars();
        if vars.is_empty() {
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

    pub fn train_step(&mut self, input: &Tensor, labels: &Tensor) -> Result<f32> {
        let loss = {
            let mut state = self.state.lock().unwrap();
            state.model.clear_cache();
            let logits = state.model.forward(input, None, 0)?;
            let (_b, seq_len, _vocab) = logits.dims3()?;
            let logits = logits.narrow(1, 0, seq_len - 1)?;
            let labels = labels.narrow(1, 1, seq_len - 1)?;
            let (b, s, vocab) = logits.dims3()?;
            let logits = logits.reshape((b * s, vocab))?;
            let labels = labels.reshape((b * s,))?;
            candle_nn::loss::cross_entropy(&logits, &labels)?
        };

        let grads = loss.backward()?;
        self.step(&grads)?;

        loss.to_vec0::<f32>()
    }
}
