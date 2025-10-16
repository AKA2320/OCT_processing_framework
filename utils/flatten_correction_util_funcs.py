from skimage.transform import warp, AffineTransform
import cv2
from tqdm import tqdm
import numpy as np
from utils.util_funcs import ncc
# from collections import defaultdict
from scipy.optimize import minimize as minz
from utils.util_funcs import warp_image_affine
# import h5py

## Flattening Functions (Memory optimized and vectorized)

def err_fun_flat(shif, x, y , past_shift):
    """Optimized error function with reduced memory allocation."""
    # Warp once and reuse for both x and y adjustments
    x_warped = warp_image_affine(x, [-past_shift, 0])
    y_warped = warp_image_affine(y, [past_shift, 0])

    # Additional shifts for NCC computation
    warped_x_stat = warp_image_affine(x_warped, [-shif[0], 0])
    warped_y_mov = warp_image_affine(y_warped, [shif[0], 0])

    corr = ncc(warped_x_stat, warped_y_mov)
    return float(1 - corr)
    
def all_tran_flat(data,static_flat,disable_tqdm, scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[2],1,1))
    for i in tqdm(range(data.shape[2]),desc='Flattening surfaces',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            stat = data[:,:,static_flat][::20]
            temp_img = data[:,:,i][::20]
            # MANUAL
            past_shift = 0
            for _ in range(10):
                move = minz(method='powell',fun = err_fun_flat,x0 = np.array([0.0]), bounds=[(-4,4)],
                            args = (stat
                                    ,temp_img
                                    ,past_shift))['x']

                past_shift += move[0]
            temp_tform_manual = AffineTransform(translation=(past_shift*2,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        except Exception as e:
            # raise Exception(e)
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
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
        del tr_all
        import gc
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

    # Clean up transforms array
    del tr_all
    import gc
    gc.collect()

    return data
