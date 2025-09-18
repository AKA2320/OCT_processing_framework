import h5py
from pydicom import dcmread
from natsort import natsorted
import os
import numpy as np

def load_h5_data(dirname, scan_num):
    if dirname.endswith(('.h5','.hdf5')):
        with h5py.File(dirname, 'r') as hf:
            data = hf['volume'][:,20:-20,:] # remove 20 pixels from top and bottom to avoid bottom refleaction artifacts
        return data
    else:
        if not dirname.endswith('/'):
            dirname = dirname + '/'
        path = f'{dirname}{scan_num}/'
        pic_paths = [i for i in os.listdir(path) if i.endswith('.h5')]
        with h5py.File(path+pic_paths[0], 'r') as hf:
            original_data = hf['volume'][:,20:-20,:] # remove 20 pixels from top and bottom to avoid bottom refleaction artifacts
        return original_data

def load_data_dcm(dirname, scan_num):
    if not dirname.endswith('/'):
        dirname = dirname+'/'
    if os.listdir(dirname)[0].endswith(('.dcm','.DCM')):
        pic_paths = [i for i in os.listdir(dirname) if i.endswith('.dcm') or i.endswith('.DCM')]
        pic_paths = natsorted(pic_paths)
        temp_img = dcmread(os.path.join(dirname,pic_paths[0])).pixel_array
        imgs_from_folder = np.zeros((len(pic_paths),*temp_img.shape))
        for i,j in enumerate(pic_paths):
            imgs_from_folder[i] = dcmread(os.path.join(dirname, j)).pixel_array
        imgs_from_folder = imgs_from_folder[:,20:-20,:] # remove 20 pixels from top and bottom to avoid bottom refleaction artifacts
        return imgs_from_folder
    else:
        current_scan_path = os.path.join(dirname, scan_num)
        pic_paths = [i for i in os.listdir(current_scan_path) if i.endswith('.dcm') or i.endswith('.DCM')]
        pic_paths = natsorted(pic_paths)
        temp_img = dcmread(os.path.join(current_scan_path, pic_paths[0])).pixel_array
        imgs_from_folder = np.zeros((len(pic_paths),*temp_img.shape))
        for i,j in enumerate(pic_paths):
            imgs_from_folder[i] = dcmread(os.path.join(current_scan_path, j)).pixel_array
        imgs_from_folder = imgs_from_folder[:,20:-20,:] # remove 20 pixels from top and bottom to avoid bottom refleaction artifacts
        return imgs_from_folder
    

def GUI_load_dcm(path_dir):
    # path = path_num
    if not path_dir.endswith('/'):
        path_dir = path_dir+'/'
    pic_paths = []
    for i in os.listdir(path_dir):
        if i.endswith('.dcm') or  i.endswith('.DCM'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    temp_img = dcmread(path_dir+pic_paths[0]).pixel_array
    imgs_from_folder = np.zeros((len(pic_paths),*temp_img.shape))
    for i,j in enumerate(pic_paths):
        aa = dcmread(path_dir+j)
        imgs_from_folder[i] = aa.pixel_array
    imgs_from_folder = imgs_from_folder[:,:,:]
    return imgs_from_folder

def GUI_load_h5(path_h5):
    if not path_h5.endswith('.h5'):
        raise Exception ("Not HDF5 data format")
    with h5py.File(path_h5, 'r') as hf:
        original_data = hf['volume'][:,:,:]
    return original_data

