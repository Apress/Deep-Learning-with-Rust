fn main() {
    
  let numbers = vec![1, 2, 3, 4];
  let squared: Vec<i32> = numbers.iter().map(|x| x * x ).collect();
  println!("Squared values: {:?}", squared);

}
