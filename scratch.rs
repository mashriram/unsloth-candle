use candle_core::{Tensor, Device};
use candle_core::quantized::{QTensor, GgmlDType};

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let t = Tensor::randn(0f32, 1f32, (2, 32), &dev)?;
    let qt = QTensor::quantize(&t, GgmlDType::Q4_0)?;
    let t_deq = qt.dequantize(&dev)?;
    println!("shape: {:?}", t_deq.dims());
    Ok(())
}
