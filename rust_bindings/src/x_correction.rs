use tch::{CModule, Device, Tensor, IValue, Kind};
use ndarray::{Array2, ArrayView2, concatenate, ArrayView3, Axis, s};
use std::cmp::max;

use crate::utility::min_max;


pub fn infer_x_translation(model: CModule, static_arr: Array2<f32>, moving_arr: Array2<f32>, device: Device) -> (f32,f32){
    let inp_static_arr = crop_or_pad(min_max(static_arr));
    let inp_moving_arr = crop_or_pad(min_max(moving_arr));

    let input_pair = concat_to_tensor(inp_static_arr.insert_axis(Axis(0)), inp_moving_arr.insert_axis(Axis(0)));
    let result: IValue = tch::no_grad(|| {
            model.forward_is(&vec![IValue::Tensor(input_pair)]).expect("Forward failed")
        });

    let input_pair_rev = concat_to_tensor(inp_moving_arr.insert_axis(Axis(0)), inp_static_arr.insert_axis(Axis(0)));
    let result_rev: IValue = tch::no_grad(|| {
            model.forward_is(&vec![IValue::Tensor(input_pair_rev)]).expect("Forward failed")
        });
    (extract_shift_val(result), extract_shift_val(result_rev))
}


pub fn crop_or_pad(mut arr: Array2<f32>) -> Array2<f32>{
    let target_height = 64;
    let target_width = 416;

    let (current_height, current_width): (usize,usize) = (arr.shape()[0],arr.shape()[1]);
    // PADDING LOGIC
    if (target_height > current_height) | (target_width > current_width){
        let pad_top: usize = max(0, (target_height.checked_sub(current_height).unwrap_or(0))/2);
        let pad_left: usize = max(0, (target_width.checked_sub(current_width).unwrap_or(0))/2);
        let new_shape = (max(target_height, current_height),max(target_width, current_width));
        
        let mut oversized_array = Array2::<f32>::zeros(new_shape);
        let row_start = pad_top as usize;
        let row_end = (pad_top + current_height) as usize;
        let col_start = pad_left as usize;
        let col_end = (pad_left + current_width) as usize;
        let target_slice = s![row_start..row_end, col_start..col_end];

        let mut center_view = oversized_array.slice_mut(target_slice);
        center_view.assign(&arr);

        arr = oversized_array.to_owned();
    }
    let (current_height, current_width): (usize,usize) = (arr.shape()[0],arr.shape()[1]);
    // CROPPING LOGIC
    if (target_height < current_height) | (target_width < current_width){
        let crop_top: usize = max(0, (current_height.checked_sub(target_height).unwrap_or(0))/2);
        let crop_left: usize = max(0, (current_width.checked_sub(target_width).unwrap_or(0))/2);
        
        let row_start = crop_top as usize;
        let row_end = (crop_top + target_height) as usize;
        let col_start = crop_left as usize;
        let col_end = (crop_left + target_width) as usize;
        let target_slice = s![row_start..row_end, col_start..col_end];

        arr = arr.slice(target_slice).to_owned();
    }
    arr
}

pub fn load_model() -> CModule{
    // let device = Device::Cpu;
    let mut model = CModule::load_on_device("../models/transmorph_lateral_X_translation.pt", Device::Cpu)
                    .expect("Failed to load model");
    model.set_eval();
    model
}

pub fn concat_to_tensor<'a>(arr1: ArrayView3<'a, f32>, arr2: ArrayView3<'a, f32>) -> Tensor {
    let concatenated = concatenate(Axis(0), &[arr1, arr2]).unwrap();
    let shape: Vec<i64> = concatenated.shape().iter().map(|&dim| dim as i64).collect();
    let data: Vec<f32> = concatenated.into_iter().collect();
    let tensor = Tensor::from_slice(&data)
                        .reshape(&shape)
                        .to_kind(Kind::Double)
                        .unsqueeze(0)
                        .to_device(Device::Cpu);
    tensor
}

pub fn extract_shift_val(result: IValue) -> f32{
    let output: Tensor = if let IValue::Tuple(mut outputs) = result {
        let t1: Tensor = if let IValue::Tensor(temp_t1) = outputs.pop().unwrap() {
            temp_t1
        }else{
            panic!("Pop failed, vec was empty");
        };
        t1
    }else{
        panic!("something wrong with the output of the model");
    };
    let val: f32 = output.double_value(&[0,0]) as f32;
    val
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, array, Array2, Array3};
    use image::{ImageBuffer, Luma};

    #[test]
    fn check_crop_or_pad() {
        let test_rows = &[47,50,64,10,151];
        let test_cols = &[9,50,416,553,5];
        for i in 0..test_rows.len(){
            let mut array1 = Array2::<f32>::ones((test_rows[i],test_cols[i]));
            array1 = crop_or_pad(array1);
            assert_eq!(array1.shape(), &[64 as usize, 416 as usize],
            "Shape mismatch at index {}. Expected [{}, {}], got {:?}", 
            i, 64, 416, array1.shape());
        }
    }

    #[test]
    fn torch_test(){
        let model = load_model();

        let mut array1: Array3<f32> = crop_or_pad(Array2::<f32>::zeros((64, 416))).insert_axis(Axis(0));
        array1.slice_mut(s![.., .., 300..350]).fill(1.0);

        let mut array2: Array3<f32> = crop_or_pad(Array2::<f32>::zeros((64, 416))).insert_axis(Axis(0));
        array2.slice_mut(s![.., .., 303..353]).fill(1.0);

        let input1 = concat_to_tensor(array1.view(), array2.view());
        let input2 = concat_to_tensor(array2.view(), array1.view());

        println!("input1 is : {:?}", input1);
        println!("input2 is : {:?}", input2);
        
        let result: IValue = tch::no_grad(|| {
            model.forward_is(&vec![IValue::Tensor(input1)]).expect("Forward failed")
        });
        let val1 = extract_shift_val(result);
        
        let result: IValue = tch::no_grad(|| {
            model.forward_is(&vec![IValue::Tensor(input2)]).expect("Forward failed")
        });
        let val2 = extract_shift_val(result);
        
        println!("Val1: {} \n Val2: {}",val1, val2);
        assert!(val1+val2 < 0.2);
    }

}