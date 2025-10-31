use autodiff::*;

fn main() {
    // Flat vector of 12 dummy values
    let params: Vec<i32> = (1..=12).collect();

    let input_size = 3;
    let hidden_size = 4;
    let mut idx = 0;

    // Convert flat list into 4x3 matrix
    let matrix: Vec<Vec<i32>> = (0..hidden_size)
        .map(|_| {
            (0..input_size)
                .map(|_| {
                    let val = params[idx];
                    idx += 1;
                    val
                })
                .collect() // inner collect → row of 3 values
        })
        .collect(); // outer collect → 4 rows = 4x3 matrix

    println!("{:?}", matrix);
}
