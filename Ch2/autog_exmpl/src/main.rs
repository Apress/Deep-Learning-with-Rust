use autograd as ag;
use autograd::tensor::Variable; // Import the Variable trait to use the `variable` method

fn main() {
    // Create a computational graph
    ag::with(|graph| {
        // Define a scalar variable x
        let x = graph.variable(ag::ndarray::arr0(3.0)); // x = 3.0
        let y = x * x; // y = x^2

        // Compute the gradient of y with respect to x
        let grads = graph.grad(&[y], &[x]);

        // Handle the result from the gradient computation
        match grads[0].eval(&[]) {
            Ok(grad) => println!("Gradient of y with respect to x: {:?}", grad),
            Err(e) => println!("Error computing gradient: {:?}", e),
        }
    });
}
