fn main() {
    println!("Starting memory allocation...");

    // Allocate a large vector of 100 million bytes (approximately 100 MB)
    let v = vec![0u8; 100_000_000];
    println!("Vector created with length: {}", v.len());

    // Use the vector to prevent optimization removing it
    let sum: u64 = v.iter().map(|&x| x as u64).sum();
    println!("Sum of elements: {}", sum);
}
