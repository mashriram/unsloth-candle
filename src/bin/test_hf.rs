use hf_hub::api::sync::ApiBuilder;
use std::env;

fn main() {
    println!("Testing hf-hub...");
    println!("HF_HOME: {:?}", env::var("HF_HOME"));
    println!("HF_HUB_CACHE: {:?}", env::var("HF_HUB_CACHE"));
    println!("HF_HUB_ENDPOINT: {:?}", env::var("HF_HUB_ENDPOINT"));

    env::set_var("HF_HUB_ENDPOINT", "https://huggingface.co");
    let api = ApiBuilder::new().build().expect("Failed to create API");
    let repo = api.model("HuggingFaceTB/SmolLM-135M".to_string());
    println!("Repo created. Fetching config...");
    match repo.get("config.json") {
        Ok(path) => println!("Config path: {:?}", path),
        Err(e) => println!("Error fetching config: {:?}", e),
    }
}
