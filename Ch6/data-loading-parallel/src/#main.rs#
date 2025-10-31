use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    // Data loader thread
    thread::spawn(move || {
        for i in 0..5 {
            let data_batch = format!("batch {}", i);
            tx.send(data_batch).unwrap();
        }
    });

    // Training loop
    for received in rx {
        println!("Training on {}", received);
    }
}
