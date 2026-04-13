import glob
import re

# 1. Update mod.rs to inject configure_cache in RustModel
with open("src/model/mod.rs", "r") as f:
    text = f.read()

# We look for the clear_cache match block and duplicate it for configure_cache
match_block = re.search(r'pub fn clear_cache\(&mut self\)\s*\{\s*match self \{.*?\n\s*\}\s*\}', text, re.DOTALL)
if match_block and "pub fn configure_cache" not in text:
    clear_fn = match_block.group(0)
    # create configure_cache
    config_fn = clear_fn.replace("clear_cache(&mut self)", "configure_cache(&mut self, q: crate::core::cache::KVQuantization, rotor: bool)")
    config_fn = re.sub(r'=> m\.clear_cache\(\),', r'=> m.configure_cache(q.clone(), rotor),', config_fn)
    
    text = text.replace(clear_fn, clear_fn + "\n\n    " + config_fn)

with open("src/model/mod.rs", "w") as f:
    f.write(text)

# 2. Update every model to have configure_cache
for file in glob.glob("src/model/*.rs"):
    if file.endswith("mod.rs") or file.endswith("linear4bit.rs") or file.endswith("vision.rs"):
        continue
        
    with open(file, "r") as f:
        t = f.read()

    if "pub fn clear_cache" in t and "pub fn configure_cache" not in t:
        # Check if the cache field is public
        if "pub cache: Cache" in t or "pub cache: crate::core::cache::Cache" in t or "pub cache: crate::model::llama::Cache" in t:
            func = """
    pub fn configure_cache(&mut self, q: crate::core::cache::KVQuantization, rotor: bool) {
        self.cache.quantization = q;
        self.cache.use_rotor = rotor;
        self.clear_cache();
    }"""
        else:
            func = """
    pub fn configure_cache(&mut self, _q: crate::core::cache::KVQuantization, _rotor: bool) {}"""

        # inject after clear_cache
        t = re.sub(r'(pub fn clear_cache\(&mut self\)\s*(?:\{[^}]*\}|;))', r'\1' + func, t)
        
        with open(file, "w") as f:
            f.write(t)
