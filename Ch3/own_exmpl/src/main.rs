fn process_dataset(data: Vec<i32>) {
    println!("Processing dataset with {} elements", data.len());
    // Ownership of `data` ends here
}

fn main() {
    let dataset = vec![1, 2, 3, 4, 5];
    process_dataset(dataset);
    // dataset cannot be accessed here as ownership is moved
    // println!("{:?}", dataset); // Uncommenting this will cause a compile error
}
