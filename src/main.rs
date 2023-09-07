use std::env;

use burn::data::dataset::Dataset;
use burn::data::dataset::source::huggingface::MNISTDataset;

use burn::optim::AdamConfig;
use burn::backend::{WgpuBackend, wgpu::AutoGraphicsApi};
use burn::autodiff::ADBackendDecorator;

use burn_wgpu::WgpuDevice;

use deep_learning_with_rust::infer::infer;
use deep_learning_with_rust::model::ModelConfig;
use deep_learning_with_rust::train::{train, TrainingConfig};

fn fit() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = WgpuDevice::default();
    
    train::<MyAutodiffBackend>(
        "./models/mnist",
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}

fn predict() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;

    let device = WgpuDevice::default();
    let dataset = MNISTDataset::test();
    let item = dataset.get(0).unwrap();

    infer::<MyBackend>("./models/mnist", device, item);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Please provide an argument. Options: fit, predict");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "fit" => fit(),
        "predict" => predict(),
        _ => {
            eprintln!("Invalid argument. Options: fit, predict");
            std::process::exit(1);
        }
    }
}