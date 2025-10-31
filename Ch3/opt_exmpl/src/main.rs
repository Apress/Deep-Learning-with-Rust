fn get_value(data: Option<i32>) {
    match data {
        Some(value) => println!("Value: {}", value),
        None => println!("No value found"),
    }
}

fn main() {
    let present = Some(42);
    let missing: Option<i32> = None;

    get_value(present);
    get_value(missing);
}
