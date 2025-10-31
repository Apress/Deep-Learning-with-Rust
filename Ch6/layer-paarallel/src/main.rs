use std::sync::{Arc, Barrier};
use std::thread;

fn compute_layer(name: &str, barrier: Arc<Barrier>) {
    println!("Computing layer: {}", name);
    // simulate computation time
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("Layer {} done.", name);
    barrier.wait(); // wait for other layers
}

fn main() {
    let barrier = Arc::new(Barrier::new(3)); // synchronize 3 threads

    let b1 = barrier.clone();
    let t1 = thread::spawn(move || {
        compute_layer("Layer 1", b1);
    });

    let b2 = barrier.clone();
    let t2 = thread::spawn(move || {
        compute_layer("Layer 2", b2);
    });

    let b3 = barrier.clone();
    let t3 = thread::spawn(move || {
        compute_layer("Layer 3", b3);
    });

    t1.join().unwrap();
    t2.join().unwrap();
    t3.join().unwrap();

    println!("All layers completed.");
}
