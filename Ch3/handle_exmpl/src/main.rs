fn parse_and_add(a: &str, b: &str) -> Result<i32, String> {
    let num_a: i32 = a.parse().map_err(|_| "Invalid number".to_string())?;
    let num_b: i32 = b.parse().map_err(|_| "Invalid number".to_string())?;
    Ok(num_a + num_b)
}

fn main() {
    match parse_and_add("42", "x") {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}
