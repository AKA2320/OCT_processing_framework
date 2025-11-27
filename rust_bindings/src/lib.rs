#![allow(unused, unused_imports, dead_code)]
use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, PyReadonlyArray2, PyReadonlyArray3};
use ndarray::{Ix2, Array2, Array3, Ix3};
use ndarray::parallel::prelude::*;
mod utility;
mod y_minimization;
use utility::*;
use y_minimization::*;


#[pyfunction]
fn run_y_correction_compute_rust(py: Python, static_data: PyReadonlyArray2<f32>, mov_data: PyReadonlyArray3<f32>)  
    -> PyResult<Vec<(f32,f32)>> {
    let static_image: Array2<f32> = static_data.as_array().to_owned(); // (m * n)
    let moving_data: Array3<f32>  = mov_data.as_array().to_owned(); // (l * m * n)

    let mut transforms: Vec<(f32,f32)> = py.detach(|| {
        moving_data.outer_iter().into_par_iter().map(|(slice1)| {
            compute_y_motion(slice1.to_owned(), static_image.clone())
        }).collect::<Vec<(f32,f32)>>()
    });
    Ok(transforms)
}

#[pymodule]
fn rust_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_y_correction_compute_rust, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis, Array3, s, ArrayView2, Array2};
    use image::{ImageBuffer, Luma};
    use std::time::{Instant, Duration};

    #[test]
    fn run_flatten(){
        let start_time = Instant::now();
        let mut array1: Array3<f32> = Array::<f32,_>::zeros((50, 100, 500));
        array1.slice_mut(s![25, 20..75, 300..350]).fill(1.0);

        let shifts: Vec<i32> = (-25..25).collect();
        let mut results:Vec<i32> = vec![];
        // println!("{:?}", shifts);
        for idx in 0..array1.shape()[0]{
            let x = shifts[idx];
            let mut temp_arr2 = array1.slice(s![idx, .., ..]).to_owned();
            temp_arr2.slice_mut(s![(30+x)..(65+x), 300..350]).fill(1.0);
            results.push(compute_y_motion(array1.slice(s![25, .., ..]).to_owned(), temp_arr2).1.round() as i32);
        }
        for i in 0..results.len(){
            assert_eq!(-results[i], shifts[i]);
        }
        println!("Compute flatten function took: {:?} seconds", Instant::now().duration_since(start_time).as_secs());
    }
}