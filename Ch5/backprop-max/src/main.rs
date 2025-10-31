use autodiff::*;

fn relu(x: FT<f64>) -> FT<f64> {
    if x.x > 0.0 {
        x
    } else {
        FT::cst(0.0)
    }
}

fn main() {
    // --- Architecture: 3 input → 4 hidden → 1 output ---
    let input_size = 3;
    let hidden_size = 4;

    // --- Initialize weights and biases with small values ---
    let mut w1 = vec![vec![0.1; input_size]; hidden_size]; // W1: [4 x 3]
    let mut b1 = vec![0.0; hidden_size]; // b1: [4]
    let mut w2 = vec![0.1; hidden_size]; // W2: [4]
    let mut b2 = 0.0; // b2: scalar

    // Training data: (x1, x2, x3) => max(x1, x2, x3)
    let data = vec![
        (vec![3.0, 5.0, 2.0], 5.0),
        (vec![5.0, 7.0, 1.0], 7.0),
        (vec![1.0, 9.0, 4.0], 9.0),
        (vec![2.0, 3.0, 6.0], 6.0),
        (vec![4.0, 1.0, 3.0], 4.0),
        (vec![0.0, 8.0, 2.0], 8.0),
        (vec![7.0, 2.0, 1.0], 7.0),
        (vec![6.0, 3.0, 4.0], 6.0),
        (vec![5.0, 0.0, 2.0], 5.0),
        (vec![2.0, 6.0, 9.0], 9.0),
    ];

    let lr = 0.001;
    let epochs = 1000;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, target) in &data {
            // --- Define the loss function using autodiff ---
            let loss_fn = |params: &[FT<f64>]| {
                // Unpack parameters from a flat vector:
                let mut idx = 0;

                // W1: [hidden][input]
                let w1_ft: Vec<Vec<FT<f64>>> = (0..hidden_size)
                    .map(|_| {
                        (0..input_size)
                            .map(|_| {
                                let v = params[idx];
                                idx += 1;
                                v
                            })
                            .collect()
                    })
                    .collect();

                // b1: [hidden]
                let b1_ft: Vec<FT<f64>> = (0..hidden_size)
                    .map(|_| {
                        let v = params[idx];
                        idx += 1;
                        v
                    })
                    .collect();

                // W2: [hidden]
                let w2_ft: Vec<FT<f64>> = (0..hidden_size)
                    .map(|_| {
                        let v = params[idx];
                        idx += 1;
                        v
                    })
                    .collect();

                // b2: scalar
                let b2_ft = params[idx];

                // --- Forward pass ---
                // Hidden layer output: h_i = relu(w1_i · x + b1_i)
                let h: Vec<FT<f64>> = (0..hidden_size)
                    .map(|i| {
                        let z =
                            (0..input_size).map(|j| w1_ft[i][j] * x[j]).sum::<FT<f64>>() + b1_ft[i];
                        relu(z)
                    })
                    .collect();

                // Output: y_hat = W2 · h + b2
                let y_hat = (0..hidden_size).map(|i| w2_ft[i] * h[i]).sum::<FT<f64>>() + b2_ft;

                // --- Loss: MSE ---
                (y_hat - *target).powi(2)
            };

            // --- Flatten parameters into single vector ---
            let mut flat_params = vec![];
            for row in &w1 {
                flat_params.extend_from_slice(row);
            }
            flat_params.extend_from_slice(&b1);
            flat_params.extend_from_slice(&w2);
            flat_params.push(b2);

            // --- Compute gradient using autodiff ---
            let grads = grad(loss_fn, &flat_params);

            // --- Compute current loss ---
            let input_ft: Vec<FT<f64>> = flat_params.iter().map(|&x| FT::cst(x)).collect();
            let loss = loss_fn(&input_ft);
            total_loss += loss.x;

            // --- Update parameters using gradient descent ---
            let mut idx = 0;
            for i in 0..hidden_size {
                for j in 0..input_size {
                    w1[i][j] -= lr * grads[idx];
                    idx += 1;
                }
            }
            for i in 0..hidden_size {
                b1[i] -= lr * grads[idx];
                idx += 1;
            }
            for i in 0..hidden_size {
                w2[i] -= lr * grads[idx];
                idx += 1;
            }
            b2 -= lr * grads[idx]; // final bias
        }

        if epoch % 20 == 0 {
            println!("Epoch {epoch}: Loss = {:.5}", total_loss);
        }
    }

    // --- After training: test a prediction ---
    let test_input = vec![0.7, 0.4, 1.0];

    // Manual forward pass for prediction
    let hidden_out: Vec<f64> = (0..hidden_size)
        .map(|i| {
            let z: f64 = (0..input_size)
                .map(|j| w1[i][j] * test_input[j])
                .sum::<f64>()
                + b1[i];
            z.max(0.0) // ReLU
        })
        .collect();

    let y_pred: f64 = hidden_out.iter().zip(&w2).map(|(h, w)| h * w).sum::<f64>() + b2;

    println!("\nTest input: {:?}", test_input);
    println!("Prediction (network): {:.4}", y_pred);
    println!(
        "Ground truth (max): {:.4}",
        test_input.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
}
