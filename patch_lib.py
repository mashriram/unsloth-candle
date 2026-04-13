import re

with open("src/lib.rs", "r") as f:
    text = f.read()

# Replace for_inference to take kv_q and rotor
text = re.sub(
    r'fn for_inference\(&mut self\) -> PyResult<\(\)> \{(.*?)\s+state\.model\.clear_cache\(\);(.*?)Ok\(\(\)\)\s+\}',
    r'''#[pyo3(signature = (kv_quantization=None, use_rotor=None))]
    fn for_inference(&mut self, kv_quantization: Option<String>, use_rotor: Option<bool>) -> PyResult<()> {\1
        let q_str = kv_quantization.unwrap_or("none".to_string());
        let q = match q_str.to_lowercase().as_str() {
            "q4_0" | "q4" | "4bit" => core::cache::KVQuantization::Q4_0,
            "q8_0" | "q8" | "8bit" => core::cache::KVQuantization::Q8_0,
            _ => core::cache::KVQuantization::None,
        };
        let rotor = use_rotor.unwrap_or(false);
        state.model.configure_cache(q, rotor);
\2Ok(())
    }''',
    text,
    flags=re.DOTALL
)
with open("src/lib.rs", "w") as f:
    f.write(text)
