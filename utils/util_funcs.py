# from pydicom import dcmread
# import matplotlib.pylab as plt
import numpy as np
import os
# from skimage.transform import warp, AffineTransform
import cv2
# import h5py
# from natsort import natsorted
# from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft
# from skimage.filters import threshold_otsu
# from skimage.metrics import normalized_mutual_information as nmi
# from scipy.signal import correlate2d
import sys

def ncc(array1, array2):
    a1 = array1.flatten()-array1.mean()
    a2 = array2.flatten()-array2.mean()
    numerator = np.dot(a1, a2)
    denominator = np.linalg.norm(a1) * np.linalg.norm(a2)
    return numerator / denominator if denominator != 0 else 0.0

def min_max(data1, global_min=None, global_max=None):    
    min_val = np.min(data1) if global_min is None else global_min
    max_val = np.max(data1) if global_max is None else global_max
    if min_val == max_val:
        return data1 
    return (data1 - min_val) / (max_val - min_val)

def non_zero_crop(a,b):
    mini = max(np.min(np.where(a[0]!=0)),np.min(np.where(b[0]!=0)))
    maxi = min(np.max(np.where(a[0]!=0)),np.max(np.where(b[0]!=0)))
    return mini, maxi

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlap
            last[1] = max(last[1], current[1])  # merge
        else:
            merged.append(current)
    return merged

def warp_image_affine(image, shifts):
    mat = np.float32([[1, 0, -shifts[0]], [0, 1, -shifts[1]]])
    rows, cols = image.shape
    warped_image = cv2.warpAffine(image, mat, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_image

def resource_path(relative_path):
    """ Get absolute path to resource for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)