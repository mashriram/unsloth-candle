use pyo3::prelude::*;
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};

mod core;
mod kernels;
mod model;
// model/mod.rs is loaded as `model` module in crate root?
// Yes, `mod model;` looks for `model.rs` OR `model/mod.rs`.
// So path remains `crate::model::RustModel`.
mod trainer;

#[pyclass]
struct FastLanguageModel {
    inner: Option<Arc<Mutex<model::RustModel>>>,
}

#[pymethods]
impl FastLanguageModel {
    #[new]
    fn new() -> Self {
        FastLanguageModel { inner: None }
    }

    #[staticmethod]
    #[pyo3(signature = (model_name, max_seq_length=None, load_in_4bit=None))]
    fn from_pretrained(model_name: &str, max_seq_length: Option<usize>, load_in_4bit: Option<bool>) -> PyResult<Self> {
        println!("Loading {}...", model_name);
        
        let device = Device::cuda_if_available(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        println!("Using device: {:?}", device);
        
        let model = core::loader::load_model(model_name, load_in_4bit.unwrap_or(false), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(FastLanguageModel { inner: Some(Arc::new(Mutex::new(model))) })
    }

    fn forward(&mut self, x: Vec<u32>) -> PyResult<String> {
        if let Some(model) = &mut self.inner {
            let mut model = model.lock().unwrap();
             let device = model.device().clone();
             let input = Tensor::new(x, &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?.unsqueeze(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             let _res = model.forward(&input, None, 0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             Ok("Forward pass success".to_string())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))
        }
    }

    #[pyo3(signature = (target_modules, rank=16, alpha=16.0, dropout=0.0, use_dora=false))]
    fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> PyResult<()> {
        if let Some(model) = &mut self.inner {
             let mut model = model.lock().unwrap();
             model.apply_lora(target_modules, rank, alpha, dropout, use_dora).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             Ok(())
        } else {
             Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))
        }
    }
}

#[pyclass]
struct Trainer {
    inner: trainer::Trainer,
}

#[pymethods]
impl Trainer {
    #[new]
    fn new(model: &FastLanguageModel) -> PyResult<Self> {
        if let Some(inner) = &model.inner {
             let model_clone = inner.clone(); 
             Ok(Trainer {
                 inner: trainer::Trainer::new(model_clone)
             })
        } else {
             Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded in FastLanguageModel"))
        }
    }

    fn configure_optimizer(&mut self, learning_rate: f64) -> PyResult<()> {
        self.inner.configure_optimizer(learning_rate).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn train_step(&mut self, input: Vec<u32>, labels: Option<Vec<u32>>) -> PyResult<f32> {
        let device = {
             let model = self.inner.model.lock().unwrap();
             model.device().clone()
        };
        
        let input_tensor = Tensor::new(input.clone(), &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?.unsqueeze(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let labels_tensor = if let Some(l) = labels {
             Tensor::new(l, &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?.unsqueeze(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
             input_tensor.clone()
        };
        
        self.inner.train_step(&input_tensor, &labels_tensor).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn unsloth_candle(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastLanguageModel>()?;
    m.add_class::<Trainer>()?;
    Ok(())
}
