fn step(x: f64) -> u8 {
    if x >= 0.0 {
        1
    } else {
        0
    }
}

// Single-layer logic gates using perceptrons
fn and(x: u8, y: u8) -> u8 {
    step((x as f64) * 1.0 + (y as f64) * 1.0 - 1.5)
}

fn or(x: u8, y: u8) -> u8 {
    step((x as f64) * 1.0 + (y as f64) * 1.0 - 0.5)
}

fn not(x: u8) -> u8 {
    step((x as f64) * -1.0 + 0.5)
}

// XOR as a composition of perceptrons
fn xor(x: u8, y: u8) -> u8 {
    let or_out = or(x, y);
    let and_out = and(x, y);
    let not_and = not(and_out);
    and(or_out, not_and)
}

fn main() {
    let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    for (x, y) in inputs {
        let output = xor(x, y);
        println!("XOR({}, {}) = {}", x, y, output);
    }
}
