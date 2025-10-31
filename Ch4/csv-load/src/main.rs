use csv::Reader;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("data.csv")?;
    let mut rdr = Reader::from_reader(file);

    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }

    Ok(())
}
