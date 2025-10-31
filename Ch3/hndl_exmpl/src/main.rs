fn calculate_sum(data: &Vec<i32>) -> i32 {
    data.iter().sum()
}

fn update_first_element(data: &mut Vec<i32>, value: i32) {
    if let Some(first) = data.get_mut(0) {
        *first = value;
    }
}

fn main() {
    let mut dataset = vec![1, 2, 3, 4, 5];
    println!("Sum: {}", calculate_sum(&dataset));
    update_first_element(&mut dataset, 10);
    println!("Updated dataset: {:?}", dataset);
}
