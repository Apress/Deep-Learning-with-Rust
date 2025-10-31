fn step(x: f64) -> u8 {
    if x >= 0.0 {
        1
    } else {
        0
    }
}

fn perceptron(x: u8, y: u8) -> u8 {
    let w1 = 1.0;
    let w2 = 1.0;
    let bias = -1.5;

    let sum = (x as f64) * w1 + (y as f64) * w2 + bias;
    step(sum)
}

fn main() {
    let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    for (x, y) in inputs {
        let output = perceptron(x, y);
        println!("Input: ({}, {}) => AND: {}", x, y, output);
    }
}
