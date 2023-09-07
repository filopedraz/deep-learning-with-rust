use burn::{record::{CompactRecorder, Recorder}, data::{dataset::source::huggingface::MNISTItem, dataloader::batcher::Batcher}, tensor::backend::Backend, config::Config, module::Module};

use crate::{train::TrainingConfig, data::MNISTBatcher};


pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to load trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}