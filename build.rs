fn main() {
    println!("cargo:rerun-if-changed=src/kernels");
    // In the future, we will iterate over kernel files and compile/embed them here.
}
