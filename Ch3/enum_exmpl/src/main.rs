#[allow(dead_code)]
enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
}

fn describe_activation(activation: Activation) {
    match activation {
        Activation::ReLU => println!("Rectified Linear Unit"),
        Activation::Sigmoid => println!("Sigmoid Activation"),
        Activation::Tanh => println!("Hyperbolic Tangent"),
    }
}

fn main() {
    let activation = Activation::Sigmoid;
    describe_activation(activation);
}
