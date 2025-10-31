use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

struct Model {
    epoch: u32,
}

fn main() {
    let model = Arc::new(Mutex::new(Model { epoch: 0 }));

    let model_clone = Arc::clone(&model);
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(5));
        let m = model_clone.lock().unwrap();
        println!("Evaluating at epoch {}", m.epoch);
    });

    for epoch in 1..=10 {
        {
            let mut m = model.lock().unwrap();
            m.epoch = epoch;
        }
        println!("Training epoch {}", epoch);
        thread::sleep(Duration::from_secs(2));
    }
}
