use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

fn main() {
    let mut rng = thread_rng();

    // Uniform distribution between -1.0 and 1.0
    let uniform = Uniform::new(-1.0, 1.0);
    let u_sample: f64 = uniform.sample(&mut rng);
    println!("Uniform sample: {}", u_sample);

    // Normal distribution with mean 0.0 and standard deviation 1.0
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_sample: f64 = normal.sample(&mut rng);
    println!("Normal sample: {}", n_sample);
}
