enum DataPoint {
    Numeric(f64),
    Categorical { label: String, probability: f64 },
}

fn main() {
    let point1 = DataPoint::Numeric(42.0);
    let point2 = DataPoint::Categorical {
        label: String::from("Category A"),
        probability: 0.8,
    };

    match point1 {
        DataPoint::Numeric(value) => println!("Numeric Value: {}", value),
        DataPoint::Categorical { label, probability } => {
            println!("Label: {}, Probability: {}", label, probability);
        }
    }

    match point2 {
        DataPoint::Numeric(value) => println!("Numeric Value: {}", value),
        DataPoint::Categorical { label, probability } => {
            println!("Label: {}, Probability: {}", label, probability);
        }
    }
}
