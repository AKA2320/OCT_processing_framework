from skimage.transform import warp, AffineTransform
from tqdm import tqdm
import numpy as np
from utils.util_funcs import ncc
from scipy.optimize import minimize as minz
from utils.util_funcs import warp_image_affine
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import gc

## Y-Motion Functions (Memory optimized and vectorized)

def _compute_y_transform(i, stat, sampled_data):
    """Worker function for parallel per-slice y-motion correction."""
    try:
        temp_img = sampled_data[i, :, :]
        past_shift = 0
        for _ in range(10):
            move = minz(method='powell', fun=err_fun_y, x0=np.array([0.0]), bounds=[(-5,5)],
                        args=(stat, temp_img, past_shift))['x']
            past_shift += move[0]
        temp_tform_manual = AffineTransform(translation=(0, past_shift*2))
        return temp_tform_manual.matrix
    except Exception as e:
        return np.eye(3)

def err_fun_y(shif, x, y, past_shift):
    """Optimized error function for Y-motion correction."""
    # Warp once per call and reuse
    warped_x = warp_image_affine(x, [0, -past_shift])
    warped_y = warp_image_affine(y, [0, past_shift])

    warped_x = warp_image_affine(warped_x, [0, -shif[0]])
    warped_y = warp_image_affine(warped_y, [0, shif[0]])

    corr = ncc(warped_x, warped_y)
    return float(1 - corr)

def all_trans_y(data,static_y_motion,disable_tqdm,scan_num):
    data_depth_z = data.shape[0]
    transforms_all = np.tile(np.eye(3),(data_depth_z,1,1))
    sampled_data = data[:,:,::20] # dont need too much info surface registration
    static_data = sampled_data[static_y_motion,:,:]
    del data
    worker = partial(_compute_y_transform, stat=static_data, sampled_data=sampled_data)
    with ThreadPoolExecutor(max_workers=4) as executor:
        computed_transforms = list(executor.map(worker, range(data_depth_z-1)))
    transforms_all[:data_depth_z-1] = computed_transforms
    # Note: only first data_depth_z-1 transforms are computed, last remains eye as initialized
    del sampled_data, static_data
    gc.collect()
    return transforms_all

def y_motion_correcting(data, slice_coords, top_surf, partition_coord, disable_tqdm, scan_num):
    """Memory optimized Y-motion correction that works in-place."""
    # Create view of sliced data to avoid unnecessary copy
    slice_indices = np.r_[tuple(np.r_[start:end] for start, end in slice_coords)]
    temp_sliced_data = data[:, slice_indices, :]  # This is a view, not a copy

    # Find reference B-scan for motion correction
    static_y_motion = np.argmax(np.sum(temp_sliced_data, axis=(1, 2)))

    # Compute all transforms
    tr_all_y = all_trans_y(temp_sliced_data, static_y_motion, disable_tqdm, scan_num)

    # Clean up the temporary sliced data reference
    del temp_sliced_data

    # Apply transforms in-place based on partition settings
    if partition_coord is None:
        # Apply to entire dataset
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i] = warp(data[i], AffineTransform(matrix=tr_all_y[i]), order=3)
        del tr_all_y
        import gc
        gc.collect()

        return data
    elif top_surf:
        # Apply only to top portion
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i, :partition_coord] = warp(data[i, :partition_coord], AffineTransform(matrix=tr_all_y[i]), order=3)
    else:
        # Apply only to bottom portion
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i, partition_coord:] = warp(data[i, partition_coord:], AffineTransform(matrix=tr_all_y[i]), order=3)

    # Clean up transforms array
    del tr_all_y
    import gc
    gc.collect()

    return data
