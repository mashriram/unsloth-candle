import glob
import re

for file in glob.glob("src/model/*.rs"):
    if file == "src/model/mod.rs":
        with open(file, "r") as f:
            t = f.read()
        
        # fix the enum match
        t = re.sub(
            r'm\.cache\.quantization = q\.clone\(\); m\.cache\.use_rotor = rotor; m\.clear_cache\(\);',
            r'm.configure_cache(q.clone(), rotor);',
            t
        )
        with open(file, "w") as f:
            f.write(t)
        continue

    with open(file, "r") as f:
        t = f.read()

    # If it has pub fn clear_cache(&mut self) { ... }, inject configure_cache right after that.
    # Note: `clear_cache` might be empty `{}` or just `{ ... }`
    pattern = re.compile(
        r'(pub fn clear_cache\(&mut self\)\s*\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\})', 
        re.DOTALL
    )
    
    if "configure_cache" not in t:
        if "pub cache: Cache" in t or "pub cache: crate::core::cache::Cache" in t or "pub cache: crate::model::llama::Cache" in t:
            # Add proper configure
            t = pattern.sub(
                r'\1\n\n    pub fn configure_cache(&mut self, q: crate::core::cache::KVQuantization, rotor: bool) {\n        self.cache.quantization = q;\n        self.cache.use_rotor = rotor;\n        self.clear_cache();\n    }', 
                t
            )
        else:
            # Fallback for empty cache like gpt_neox
            t = pattern.sub(
                r'\1\n\n    pub fn configure_cache(&mut self, _q: crate::core::cache::KVQuantization, _rotor: bool) {}', 
                t
            )

        with open(file, "w") as f:
            f.write(t)

