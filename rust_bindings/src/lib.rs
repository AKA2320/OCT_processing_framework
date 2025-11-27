// #![allow(unused, unused_imports, dead_code)]
use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use ndarray::{Array2, Array3, Axis};
use ndarray::parallel::prelude::*;
mod utility;
mod y_minimization;
mod flat_minimization;
use y_minimization::*;
use flat_minimization::*;


#[pyfunction]
fn run_y_correction_compute_rust(py: Python, static_data: PyReadonlyArray2<f32>, mov_data: PyReadonlyArray3<f32>)  
    -> PyResult<Vec<(f32,f32)>> {
    let static_image: Array2<f32> = static_data.as_array().to_owned(); // (m * n)
    let moving_data: Array3<f32>  = mov_data.as_array().to_owned(); // (l * m * n)

    let transforms: Vec<(f32,f32)> = py.detach(|| {
        moving_data.axis_iter(Axis(0)).into_par_iter().map(|slice1| {
            compute_y_motion(slice1.to_owned(), static_image.clone())
        }).collect()
    });
    Ok(transforms)
}

#[pyfunction]
fn run_flat_correction_compute_rust(py: Python, static_data: PyReadonlyArray2<f32>, mov_data: PyReadonlyArray3<f32>)  
    -> PyResult<Vec<(f32,f32)>> {
    let static_image: Array2<f32> = static_data.as_array().to_owned(); // (m * n)
    let moving_data: Array3<f32>  = mov_data.as_array().to_owned(); // (l * m * n)

    let transforms: Vec<(f32,f32)> = py.detach(|| {
        moving_data.axis_iter(Axis(2)).into_par_iter().map(|slice1| {
            compute_flat_motion(slice1.to_owned(), static_image.clone())
        }).collect()
    });
    Ok(transforms)
}

#[pymodule]
fn rust_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_y_correction_compute_rust, m)?)?;
    m.add_function(wrap_pyfunction!(run_flat_correction_compute_rust, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array3, s};
    // use image::{ImageBuffer, Luma};
    use std::time::{Instant};

    #[test]
    fn run_y_correct(){
        let start_time = Instant::now();
        let mut array1: Array3<f32> = Array::<f32,_>::zeros((20, 100, 500));
        array1.slice_mut(s![10, 20..75, 300..350]).fill(1.0);

        let shifts: Vec<i32> = (-10..10).collect();
        // let results:Vec<i32> = vec![];
        // println!("{:?}", shifts);
        // for idx in 0..array1.shape()[0]{
        //     let x = shifts[idx];
        //     let mut temp_arr2 = array1.slice(s![idx, .., ..]).to_owned();
        //     temp_arr2.slice_mut(s![(20+x)..(75+x), 300..350]).fill(1.0);
        //     results.push(compute_y_motion(array1.slice(s![10, .., ..]).to_owned(), temp_arr2).1.round() as i32);
        // }
        // println!("{:?}", results);
        let results: Vec<i32> = (0..array1.shape()[0]).into_par_iter()
        .map(|idx|{
                let x = shifts[idx];
                let mut temp_arr2 = array1.slice(s![idx, .., ..]).to_owned();
                temp_arr2.slice_mut(s![(20+x)..(75+x), 300..350]).fill(1.0);
                compute_y_motion(array1.slice(s![10, .., ..]).to_owned(), temp_arr2).1.round() as i32
            }
        ).collect();

        for i in 0..results.len(){
            assert_eq!(-results[i], shifts[i]);
        }
        println!("Compute Y motion function took: {:?} seconds", Instant::now().duration_since(start_time).as_secs());
    }

    #[test]
    fn run_flatten(){
        let start_time = Instant::now();
        let mut array1: Array3<f32> = Array::<f32,_>::zeros((20, 100, 50));
        // array1.slice_mut(s![25, 20..75, 300..350]).fill(1.0);
        let shifts: Vec<i32> = (-25..25).collect();
        for idx in 0..shifts.len(){
            array1.slice_mut(s![.., (50+shifts[idx])..(75+shifts[idx]), idx]).fill(1.0);
        }

        // let (h, w1) = array1.slice(s![25, .., ..]).dim();
        // let buffer1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
        // w1 as u32,
        // h as u32,
        // array1.slice(s![25, .., ..]).mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8).as_slice().unwrap().to_vec(),
        // )
        // .unwrap();
        // buffer1.save("test_array_flatten1.png").unwrap();

        // let shifts: Vec<i32> = (-25..25).collect();
        let mut results:Vec<i32> = vec![];
        // println!("{:?}", shifts);
        for idx in 0..array1.shape()[2]{
            // let x = shifts[idx];
            let temp_arr2 = array1.slice(s![.., .., idx]).to_owned();
            // temp_arr2.slice_mut(s![(30+x)..(65+x), 300..350]).fill(1.0);
            results.push(compute_flat_motion(array1.slice(s![.., .., 25]).to_owned(), temp_arr2).0.round() as i32);
        }

        // let (h, w1) = array1.slice(s![.., .., 25]).dim();
        // let buffer1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
        // w1 as u32,
        // h as u32,
        // array1.slice(s![.., .., 25]).mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8).as_slice().unwrap().to_vec(),
        // )
        // .unwrap();
        // buffer1.save("test_sidearray_flatten1.png").unwrap();

        // println!("{:?}", results);
        for i in 0..results.len(){
            assert_eq!(-results[i], shifts[i]);
        }
        println!("Compute flatten function took: {:?} seconds", Instant::now().duration_since(start_time).as_secs());
    }
}