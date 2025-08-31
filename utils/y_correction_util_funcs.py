from skimage.transform import warp, AffineTransform
from tqdm import tqdm
import numpy as np
from utils.util_funcs import ncc
from scipy.optimize import minimize as minz
from utils.util_funcs import warp_image_affine

## Y-Motion Functions
def err_fun_y(shif, x, y , past_shift):
    x = warp_image_affine(x, [0,-past_shift])
    y = warp_image_affine(y, [0,past_shift])
    warped_x_stat = warp_image_affine(x, [0,-shif[0]])
    warped_y_mov = warp_image_affine(y, [0,shif[0]])
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)

def all_trans_y(data,static_y_motion,disable_tqdm,scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='Y-motion Correction',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            stat = data[static_y_motion][:,::20].copy()
            temp_img = data[i][:,::20].copy()
            # MANUAL
            past_shift = 0
            for _ in range(10):
                move = minz(method='powell',fun = err_fun_y,x0 = np.array([0.0]), bounds=[(-5,5)],
                            args = (stat
                                    ,temp_img
                                    ,past_shift))['x']
                past_shift += move[0]
            temp_tform_manual = AffineTransform(matrix = AffineTransform(translation=(0,past_shift*2)))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        except Exception as e:
            # with open(f'debugs/debug{scan_num}.txt', 'a') as f:
            #     f.write(f'Y motion EVERYTHIN FAILED HERE\n')
            #     f.write(f'NAME: {scan_num}\n')
            #     f.write(f'Ith: {i}\n')
            # raise Exception(e)
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def y_motion_correcting(data,slice_coords,top_surf,partition_coord,disable_tqdm,scan_num):
    temp_sliced_data = data[:, np.r_[tuple(np.r_[start:end] for start, end in slice_coords)], :].copy()
    static_y_motion = np.argmax(np.sum(temp_sliced_data,axis=(1,2)))
    tr_all_y = all_trans_y(temp_sliced_data,static_y_motion,disable_tqdm,scan_num)
    if partition_coord is None:
        for i in tqdm(range(data.shape[0]),desc='Y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i]  = warp(data[i],AffineTransform(matrix=tr_all_y[i]),order=3)
        return data
    
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='Y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,:partition_coord]  = warp(data[i,:partition_coord],AffineTransform(matrix=tr_all_y[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='Y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,partition_coord:]  = warp(data[i,partition_coord:],AffineTransform(matrix=tr_all_y[i]),order=3)
    return data
