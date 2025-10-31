// Define a struct for a neural network layer
struct Layer {
    name: String,
    num_neurons: usize,
    activation: String,
}

// Implement methods for the Layer struct
impl Layer {
    fn describe(&self) -> String {
        format!(
            "Layer: {}, Neurons: {}, Activation: {}",
            self.name, self.num_neurons, self.activation
        )
    }
}

fn main() {
    let layer = Layer {
        name: String::from("Output Layer"),
        num_neurons: 10,
        activation: String::from("Softmax"),
    };
    println!("{}", layer.describe());
}
