import os
import glob
import re

d = 'src/model'
files = glob.glob(os.path.join(d, '*.rs'))

cache_def = re.compile(r'#\[derive\(Clone\)\]\s*pub struct Cache \{[^\}]+\}\s*impl Cache \{[^\}]+\}', re.MULTILINE)

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()

    # Special case: llama.rs defines Cache, let's remove it and add pub use
    if os.path.basename(fpath) == 'llama.rs':
        # Remove Cache struct and impl
        # A simple regex for struct Cache and impl Cache
        # Since it's multi-line, it's easier to find lines
        lines = content.split('\n')
        out_lines = []
        skip = False
        for line in lines:
            if line.startswith('#[derive(Clone)]'):
                # peek ahead
                pass
            if 'pub struct Cache {' in line or 'impl Cache {' in line:
                skip = True
            
            if not skip:
                out_lines.append(line)
            if skip and line == '}':
                skip = False
                continue
        content = '\n'.join(out_lines)
        # Add pub use
        content = content.replace('use crate::model::layers::{AdapterLayer, UnslothRmsNorm};', 
                                  'use crate::model::layers::{AdapterLayer, UnslothRmsNorm};\npub use crate::core::cache::{Cache, CacheState, KVQuantization};')

    # Now replace the KV cache update logic
    # It might look like:
    # let (k, v) = if cache.use_kv_cache {
    #     let (k, v) = match &cache.kvs[layer_idx] {
    # ...
    # cache.kvs[layer_idx] = ...
    # (k, v)
    # } else { (k, v) };
    
    # We want to just replace:
    """
        let (k, v) = if cache.use_kv_cache {
            let (k, v) = match &cache.kvs[layer_idx] {
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
                None => (k, v),
            };
            cache.kvs[layer_idx] = Some::<(Tensor, Tensor)>((k.clone(), v.clone()));
            (k, v)
        } else {
            (k, v)
        };
    """
    # Or variants without "::<...>"

    pattern1 = re.compile(
        r'let\s+\(k,\s*v\)\s*=\s*if\s+cache\.use_kv_cache\s*\{\s*let\s+\(k,\s*v\)\s*=\s*match\s+&?cache\.kvs\[layer_idx\]\s*\{[^;]+;\s*cache\.kvs\[layer_idx\]\s*=\s*Some(?:[^;]*?)(\(\(k\.clone\(\),\s*v\.clone\(\)\)\)|[^;]+);\s*\(k,\s*v\)\s*\}\s*else\s*\{\s*\(k,\s*v\)\s*\};',
        re.MULTILINE | re.DOTALL
    )
    
    # Actually some files like gemma.rs might not have `if cache.use_kv_cache` wrapping it, they just do:
    # let (k, v) = match &cache.kvs[layer_idx] ... cache.kvs[layer_idx] = Some(...)
    
    # General regex to catch:
    # let (k, v) = match &cache.kvs[layer_idx] { ... };
    # cache.kvs[layer_idx] = ...;
    # It might be indented.

    pattern2 = re.compile(
        r'(let\s+\(k,\s*v\)\s*=\s*match\s+&?\w*\.cache\.kvs\[layer_idx\]\s*\{.*?None\s*=>\s*\(k,\s*v\),\s*\};\s*.*?cache\.kvs\[layer_idx\]\s*=\s*[^\n;]+;)',
        re.DOTALL
    )

    pattern_llama = re.compile(
        r'(let\s+\(k,\s*v\)\s*=\s*if\s+cache\.use_kv_cache\s*\{.*?cache\.kvs\[layer_idx\]\s*=\s*[^\n;]+;\s*\(k,\s*v\)\s*\}\s*else\s*\{\s*\(k,\s*v\)\s*\};)',
        re.DOTALL
    )

    if pattern_llama.search(content):
        content = pattern_llama.sub(r'let (k, v) = cache.append_and_fetch(layer_idx, &k, &v)?;\n', content)
    elif pattern2.search(content):
        # We also need to add if cache.use_kv_cache inside append_and_fetch, so just replacing the match segment is fine.
        content = pattern2.sub(r'let (k, v) = cache.append_and_fetch(layer_idx, &k, &v)?;\n', content)


    with open(fpath, 'w') as f:
        f.write(content)

