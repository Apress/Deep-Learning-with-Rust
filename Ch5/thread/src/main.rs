use std::thread;
use std::time::Duration;

fn main() {
    // Start a new thread
    let handle = thread::spawn(|| {
        for i in 1..5 {
            println!("Hello from the spawned thread {}", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Wait for the spawned thread to finish
    handle.join().unwrap();

    for i in 1..5 {
        println!("Hello from the main thread {}", i);
        thread::sleep(Duration::from_millis(1));
    }
    //handle.join().unwrap();
}
