struct Layer {
    name: String,
    neurons: usize,
}

fn build_pipeline() -> Vec<Layer> {
    vec![
        Layer {
            name: String::from("Input Layer"),
            neurons: 64,
        },
        Layer {
            name: String::from("Hidden Layer"),
            neurons: 128,
        },
        Layer {
            name: String::from("Output Layer"),
            neurons: 10,
        },
    ]
}

fn main() {
    let pipeline = build_pipeline();
    for layer in pipeline {
        println!("Layer: {}, Neurons: {}", layer.name, layer.neurons);
    }
}
