use autodiff::*;

fn main() {
    let y: FT<f64> = FT { x: 3.5, dx: 2.0 };
    println!("y.x={}, y.dx={}!", y.x, y.dx);

    let z: &[FT<f64>] = &[FT { x: 1.0, dx: 0.2 }, FT { x: 3.9, dx: 5.8 }];
    println!("Slice of the dual numbers{:?}", z);

    let duals: Vec<FT<f64>> = vec![
        FT { x: 1.0, dx: 0.1 },
        FT { x: 2.0, dx: 0.2 },
        FT { x: 3.0, dx: 0.3 },
    ];

    println!("Vectors of dual numbers{:?}", duals);

    let a = FT { x: 3.0, dx: 1.0 };
    let b = FT::cst(5.0);
    let c = a * b;
    println!("c={:?}", c);
}
