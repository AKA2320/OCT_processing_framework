# from pydicom import dcmread
# import time
from skimage.transform import warp, AffineTransform
from tqdm import tqdm
import numpy as np
from utils.util_funcs import ncc
from utils.transmorph_helper_funcs import infer_x_translation
# from collections import defaultdict
from scipy.optimize import minimize as minz
from scipy import ndimage as scp
from utils.util_funcs import warp_image_affine

## X-Motion Functions (Memory optimized and vectorized)

def shift_func(shif, x, y, past_shift):
    """Optimized shift function for line-based corrections."""
    # Reuse shifted images to avoid redundant computations
    x_shifted = scp.shift(x, -past_shift, order=3, mode='nearest')
    y_shifted = scp.shift(y, past_shift, order=3, mode='nearest')

    warped_x_stat = scp.shift(x_shifted, -shif[0], order=3, mode='nearest')
    warped_y_mov = scp.shift(y_shifted, shif[0], order=3, mode='nearest')

    corr = ncc(warped_x_stat, warped_y_mov)
    return 1 - corr

def err_fun_x(shif, x, y, past_shift):
    """Optimized error function for patch-based corrections."""
    # Warp once per call and reuse
    x_warped = warp_image_affine(x, [-past_shift, 0])
    y_warped = warp_image_affine(y, [past_shift, 0])

    warped_x_stat = warp_image_affine(x_warped, [-shif[0], 0])
    warped_y_mov = warp_image_affine(y_warped, [shif[0], 0])

    corr = ncc(warped_x_stat, warped_y_mov)
    return float(1 - corr)

def get_line_shift(line_1d_stat, line_1d_mov):
    past_shift = 0
    for _ in range(7):
        move = minz(method='powell',fun = shift_func,x0 = np.array([0.0]),bounds =[(-4,4)],
                args = (line_1d_stat
                        ,line_1d_mov
                        ,past_shift))['x']
        past_shift += move[0]
    return -past_shift*2 # Negative because scipy shift returns opposite direction shift

def get_cell_patch_shift(patch_stat, patch__mov):
    past_shift = 0
    for _ in range(7):
        move = minz(method='powell',fun = err_fun_x,x0 = np.array([0.0]), bounds=[(-4,4)],
                    args = (patch_stat
                            ,patch__mov
                            ,past_shift))['x']
        past_shift += move[0]
    return past_shift*2

def check_best_warp(stat, mov, value, is_shift_value = False):
    err = ncc(stat,warp(mov, AffineTransform(translation=(-value,0)),order=3))
    return err

def check_multiple_warps(stat_img, mov_img, *args):
    errors = []
    warps = args[0]
    for warp_value in range(len(warps)):
        errors.append(check_best_warp(stat_img, mov_img, warps[warp_value]))
    return np.argmax(errors)

def x_motion_coorection(data, cells_coords, valid_args, enface_extraction_rows, disable_tqdm, scan_num, MODEL_X_TRANSLATION):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(0,data.shape[0]-1,2),desc='X-motion Correction',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        # st_time = time.time()
        if i not in valid_args:
            continue
        cell_warps = []    
        try:
            if (cells_coords is not None):
                if MODEL_X_TRANSLATION is not None:
                    for UP_x, DOWN_x in cells_coords:
                        stat = data[i, UP_x:DOWN_x, :]
                        temp_manual = data[i+1, UP_x:DOWN_x, :]
                        temp_cell_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, stat, temp_manual))[0]
                        inv_temp_cell_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, temp_manual, stat))[0]
                        error_cell = abs(temp_cell_shift + inv_temp_cell_shift)
                        cell_warps.append((error_cell, temp_cell_shift))
                else:
                    if cells_coords.shape[0]==1:
                        UP_x, DOWN_x = cells_coords[0,0], cells_coords[0,1]
                        stat = data[i, UP_x:DOWN_x, :]
                        temp_manual = data[i+1, UP_x:DOWN_x, :]
                    else:
                        stat = data[i,np.r_[tuple(np.r_[start:end] for start, end in cells_coords)],:]
                        temp_manual = data[i+1,np.r_[tuple(np.r_[start:end] for start, end in cells_coords)],:]
                    # MANUAL
                    temp_cell_patch_shift = get_cell_patch_shift(stat,temp_manual)
                    inv_temp_cell_patch_shift = get_cell_patch_shift(temp_manual,stat)
                    error_cell = abs(temp_cell_patch_shift + inv_temp_cell_patch_shift)
                    cell_warps.append((error_cell, temp_cell_patch_shift))
            else:
                cell_warps = [(float('inf'), 0.0)]
        except Exception as e:
            cell_warps = [(float('inf'), 0.0)]
        # enface_shape = data[:,0,:].shape[1]
        enface_wraps = []
        if len(enface_extraction_rows)>0:
            for enf_idx in range(len(enface_extraction_rows)):
                try:
                    if MODEL_X_TRANSLATION is not None:
                        bottom_row = max(0, enface_extraction_rows[enf_idx]-32)
                        stat = data[i,bottom_row:enface_extraction_rows[enf_idx]+32]
                        temp_manual = data[i+1,bottom_row:enface_extraction_rows[enf_idx]+32]
                        temp_enface_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, stat, temp_manual))[0]
                        inv_temp_enface_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, temp_manual, stat))[0]
                        error_enface = abs(temp_enface_shift + inv_temp_enface_shift)
                        enface_wraps.append((error_enface, temp_enface_shift))
                    else:
                        stat = data[i, enface_extraction_rows[enf_idx]]
                        temp_manual = data[i+1, enface_extraction_rows[enf_idx]]
                        temp_enface_shift = get_line_shift(stat, temp_manual)
                        inv_temp_enface_shift = get_line_shift(temp_manual, stat)
                        error_enface = abs(temp_enface_shift + inv_temp_enface_shift)
                        enface_wraps.append((error_enface, temp_enface_shift))
                except Exception as e:
                    enface_wraps = [(float('inf'), 0.0)]
        all_warps = [*cell_warps,*enface_wraps]
        all_warps = sorted(all_warps, key=lambda x: x[0])  # Sort by error
        temp_tform_manual = AffineTransform(translation=(all_warps[0][1],0))
        transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
    return transforms_all
