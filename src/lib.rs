use pyo3::prelude::*;
use pyo3::types::PyDict;
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

pub mod core;
pub mod kernels;
pub mod model;
mod trainer;
use trainer::ModelState;

#[pyclass]
struct FastLanguageModel {
    inner: Option<Arc<Mutex<ModelState>>>,
}

#[pymethods]
impl FastLanguageModel {
    #[new]
    fn new() -> Self {
        FastLanguageModel { inner: None }
    }

    #[staticmethod]
    #[pyo3(signature = (model_name, max_seq_length=None, load_in_4bit=None, use_gradient_checkpointing=None, dtype=None, token=None))]
    fn from_pretrained(
        py: Python<'_>,
        model_name: &str,
        max_seq_length: Option<usize>,
        load_in_4bit: Option<bool>,
        use_gradient_checkpointing: Option<bool>,
        dtype: Option<String>,
        token: Option<String>,
    ) -> PyResult<(Self, PyObject)> {
        let load_4bit = load_in_4bit.unwrap_or(false);
        let grad_ckpt = use_gradient_checkpointing.unwrap_or(false);
        println!("Loading {}...", model_name);

        let device = Device::cuda_if_available(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)
            .unwrap_or(Device::Cpu);

        println!("Using device: {:?}", device);

        // Load model_dir and tokenizer via Python
        let (model_dir, tokenizer) = Python::with_gil(|py| -> PyResult<(PathBuf, PyObject)> {
            let transformers = PyModule::import_bound(py, "transformers")?;
            let auto_tokenizer = transformers.getattr("AutoTokenizer")?;

            let model_path = std::path::Path::new(model_name);
            if model_path.is_dir() {
                let model_dir = model_path.to_path_buf();
                let tokenizer = auto_tokenizer.call_method1("from_pretrained", (model_name,))?;
                return Ok((model_dir, tokenizer.into_py(py)));
            }

            let hf_hub = PyModule::import_bound(py, "huggingface_hub")?;
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("repo_id", model_name)?;
            kwargs.set_item("allow_patterns", vec![
                "config.json", "*.safetensors", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            ])?;
            
            let path: String = hf_hub.call_method("snapshot_download", (), Some(&kwargs))?.extract()?;
            let model_dir = PathBuf::from(path);

            // Load tokenizer using transformers (standard Unsloth behavior)
            let tokenizer = auto_tokenizer.call_method1("from_pretrained", (model_name,))?;
            
            Ok((model_dir, tokenizer.into_py(py)))
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("HF/Transformers setup: {}", e)))?;

        let config_path = model_dir.join("config.json");
        let config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let model = core::loader::load_model(model_name, load_4bit, grad_ckpt, &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let state = ModelState {
            model,
            config_json,
            model_name: model_name.to_string(),
            model_dir: model_dir,
        };

        let flm = FastLanguageModel {
            inner: Some(Arc::new(Mutex::new(state))),
        };

        Ok((flm, tokenizer))
    }

    /// Run the forward pass. Returns the next predicted token ID (greedy).
    #[pyo3(signature = (x, pos=0))]
    fn forward(&mut self, x: Vec<u32>, pos: usize) -> PyResult<u32> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let mut state = state.lock().unwrap();
        let device = state.model.device().clone();
        
        let itensor = Tensor::new(x, &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        let logits = state.model.forward(&itensor, None, pos)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        // Get the last token's logits
        let s = logits.dims();
        let last_logits = logits.narrow(1, s[1] - 1, 1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .squeeze(1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .squeeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        let next_token = last_logits.argmax(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_scalar::<u32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(next_token)
    }

    /// Enable inference mode (clears cache, sets flags).
    fn for_inference(&mut self) -> PyResult<()> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let mut state = state.lock().unwrap();
        state.model.clear_cache();
        println!("Inference mode enabled (KV cache cleared)");
        Ok(())
    }

    /// Apply LoRA (or DoRA) adapters to the model.
    #[pyo3(signature = (target_modules, rank=16, alpha=16.0, dropout=0.0, use_dora=false))]
    fn apply_lora(
        &mut self,
        target_modules: Vec<String>,
        rank: usize,
        alpha: f64,
        dropout: f64,
        use_dora: bool,
    ) -> PyResult<()> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let mut state = state.lock().unwrap();
        state.model.apply_lora(target_modules, rank, alpha, dropout, use_dora)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Save as merged HuggingFace safetensors (LoRA merged into base weights).
    #[pyo3(signature = (output_dir))]
    fn save_pretrained_merged(&self, output_dir: &str) -> PyResult<String> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let state = state.lock().unwrap();
        let path = core::saver::save_pretrained_merged(
            &state.model_dir,
            state.model.varmap(),
            &state.config_json,
            std::path::Path::new(output_dir),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(path.to_string_lossy().to_string())
    }

    /// Save in 4-bit NF4 quantized safetensors format.
    #[pyo3(signature = (output_dir, block_size=64))]
    fn save_in_4bit(&self, output_dir: &str, block_size: usize) -> PyResult<String> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let state = state.lock().unwrap();
        let path = core::saver::save_in_4bit(
            &state.model_dir,
            state.model.varmap(),
            &state.config_json,
            std::path::Path::new(output_dir),
            block_size,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(path.to_string_lossy().to_string())
    }

    /// Save as GGUF format (for llama.cpp / Ollama / etc.).
    #[pyo3(signature = (output_dir, quantization_type="q8_0"))]
    fn save_to_gguf(&self, output_dir: &str, quantization_type: &str) -> PyResult<String> {
        let state = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        let state = state.lock().unwrap();
        let path = core::saver::save_to_gguf(
            &state.model_dir,
            state.model.varmap(),
            &state.config_json,
            std::path::Path::new(output_dir),
            quantization_type,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(path.to_string_lossy().to_string())
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
        let state = model.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;
        // We need an Arc<Mutex<RustModel>> for the trainer — extract it
        // Since ModelState stores the model, we need to restructure slightly.
        // For now, we'll wrap the state-level Arc.
        // The trainer needs access to the model for forward + varmap for optimizer.
        Ok(Trainer {
            inner: trainer::Trainer::new_from_state(state.clone()),
        })
    }

    fn configure_optimizer(&mut self, learning_rate: f64) -> PyResult<()> {
        self.inner.configure_optimizer(learning_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn train_step(&mut self, input: Vec<u32>, labels: Option<Vec<u32>>) -> PyResult<f32> {
        let device = {
            let state = self.inner.state.lock().unwrap();
            state.model.device().clone()
        };

        let input_tensor = Tensor::new(input.clone(), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let labels_tensor = if let Some(l) = labels {
            Tensor::new(l, &device)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            input_tensor.clone()
        };

        self.inner.train_step(&input_tensor, &labels_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

/// unsloth_candle Python module
#[pymodule]
fn unsloth_candle(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastLanguageModel>()?;
    m.add_class::<Trainer>()?;
    Ok(())
}
