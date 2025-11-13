from skimage.transform import warp, AffineTransform
import gc
from tqdm import tqdm
import numpy as np
from utils.util_funcs import ncc
from scipy.optimize import minimize as minz
from utils.util_funcs import warp_image_affine
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

## Flattening Functions

def _compute_flatten_transform(i, stat, sampled_data):
    """Worker function for parallel per-slice flatten correction."""
    try:
        temp_img = sampled_data[:, :, i]
        past_shift = 0
        shift_threshold = 0.1
        for _ in range(10):
            move = minz(method='powell', fun=err_fun_flat, x0=np.array([0.0]), bounds=[(-4,4)],
                        args=(stat, temp_img, past_shift))['x']
            if abs(move[0]) < shift_threshold:
                break
            past_shift += move[0]
        temp_tform_manual = AffineTransform(translation=(past_shift*2, 0))
        return temp_tform_manual.params
    except Exception as e:
        return np.eye(3)

def err_fun_flat(shif, x, y , past_shift):
    """Optimized error function with reduced memory allocation."""
    # Warp once and reuse for both x and y adjustments
    warped_x = warp_image_affine(x, [-past_shift, 0])
    warped_y = warp_image_affine(y, [past_shift, 0])

    warped_x = warp_image_affine(warped_x, [-shif[0], 0])
    warped_y = warp_image_affine(warped_y, [shif[0], 0])

    corr = ncc(warped_x, warped_y)
    return float(1 - corr)

def all_tran_flat(data, static_flat, disable_tqdm, scan_num):
    data_depth_y = data.shape[2]
    transforms_all = np.tile(np.eye(3),(data_depth_y,1,1))
    sampled_data = data[::20] # dont need too much info surface registration
    static_data = sampled_data[:,:,static_flat]
    del data
    worker = partial(_compute_flatten_transform, stat=static_data, sampled_data=sampled_data)
    with ThreadPoolExecutor(max_workers = None) as executor:
        transforms_all = list(executor.map(worker, range(data_depth_y)))
    transforms_all = np.array(transforms_all)
    del sampled_data, static_data
    gc.collect()
    return transforms_all

def flatten_data(data, slice_coords, top_surf, partition_coord, disable_tqdm, scan_num):
    """Memory optimized flattening that works in-place."""
    # Create view of sliced data to avoid unnecessary copy
    slice_indices = np.r_[tuple(np.r_[start:end] for start, end in slice_coords)]
    temp_sliced_data = data[:, slice_indices, :]  # This is a view, not a copy

    # Find reference slice for flattening
    static_flat = np.argmax(np.sum(temp_sliced_data, axis=(0, 1)))

    # Compute all transforms at once
    tr_all = all_tran_flat(temp_sliced_data, static_flat, disable_tqdm, scan_num)

    # Clean up the temporary sliced data reference
    del temp_sliced_data

    # Apply transforms in-place based on partition settings
    if partition_coord is None:
        # Apply to entire data volume
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, :, i] = warp(data[:, :, i], AffineTransform(matrix=tr_all[i]), order=3)
        gc.collect()
        return data
    elif top_surf:
        # Apply only to top portion
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, :partition_coord, i] = warp(data[:, :partition_coord, i], AffineTransform(matrix=tr_all[i]), order=3)
    else:
        # Apply only to bottom portion
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, partition_coord:, i] = warp(data[:, partition_coord:, i], AffineTransform(matrix=tr_all[i]), order=3)

    gc.collect()
    return data
