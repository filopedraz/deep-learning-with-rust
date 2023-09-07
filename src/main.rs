use burn::optim::AdamConfig;
use burn::backend::{WgpuBackend, wgpu::AutoGraphicsApi};
use burn::autodiff::ADBackendDecorator;

use burn_wgpu::WgpuDevice;

use deep_learning_with_rust::model::ModelConfig;

pub fn train_model() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = WgpuDevice::default();
    
    deep_learning_with_rust::train::train::<MyAutodiffBackend>(
        "./models/mnist",
        deep_learning_with_rust::train::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}


fn main() {
    
}