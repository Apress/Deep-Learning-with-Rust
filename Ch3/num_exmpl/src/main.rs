struct Layer {
    name: String,
    num_neurons: usize, // Corrected type
    activation: String,
}

fn main() {
    let layer = Layer {
        name: String::from("Hidden Layer 1"),
        num_neurons: 128,
        activation: String::from("ReLU"),
    };

    println!(
        "Layer: {}, Neurons: {}, Activation: {}",
        layer.name, layer.num_neurons, layer.activation
    );
}
