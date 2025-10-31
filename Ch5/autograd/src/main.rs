use autodiff::*;

fn main() {
    // Define the multivariable function f(v0, v1)=v0*sin(v1)+v1^2
    let f = |v: &[FT<f64>]| v[0] * v[1].sin() + v[1] * v[1];

    // Compute gradient at x = 1.0, y = 2.0
    let df = grad(f, &vec![1.0, 2.0]);

    println!("df/dx = {}", df[0]);
    println!("df/dy = {}", df[1]);
}
