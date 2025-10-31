use rayon::prelude::*;
use std::time::Instant;

fn square(x: u64) -> u64 {
    x * x
}

fn main() {
    let n: u64 = 1_000_000;
    let data: Vec<u64> = (0..n).collect();

    // Serial sum of squares
    let start = Instant::now();
    let _sum_serial: u128 = data.iter().map(|&x| square(x) as u128).sum();
    let duration_serial = start.elapsed().as_secs_f64();
    println!("Serial time: {:.6} sec", duration_serial);

    // Parallel sum of squares
    let start = Instant::now();
    let _sum_parallel: u64 = data.par_iter().map(|&x| square(x)).sum();
    let duration_parallel = start.elapsed().as_secs_f64();
    println!("Parallel time: {:.6} sec", duration_parallel);

    // Speedup calculation
    if duration_parallel > 0.0 {
        let speedup = duration_serial / duration_parallel;
        println!("Speedup: {:.2}x", speedup);
    } else {
        println!("Parallel duration too small to measure speedup.");
    }
}
