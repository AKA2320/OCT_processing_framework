import h5py
from pydicom import dcmread
from natsort import natsorted
import os
from utils.util_funcs import min_max, resource_path
from utils.transmorph_helper_funcs import preprocess_img, detect_areas
import napari
import numpy as np
import yaml
import logging
import sys

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

def load_napari_viewer(data):
    config_path = 'datapaths.yaml'
    try:
        with open(resource_path(config_path), 'r') as f:
            config = yaml.safe_load(f)
    except:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    try:
        from ultralytics import YOLO
        MODEL_FEATURE_DETECT_PATH = config['PATHS']['MODEL_FEATURE_DETECT_PATH']
        MODEL_FEATURE_DETECT = YOLO(MODEL_FEATURE_DETECT_PATH)
        logging.info("YOLO Model Loaded Succesfully.")
        # if self.is_gui_flag:
        #     self.output_queue.put("YOLO Model Loaded Succesfully.\n")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}", exc_info=True)
        # sys.exit("Failed to load YOLO model. Exiting.")

    # Detection part
    static_flat = np.argmax(np.sum(data[:,:,:], axis=(0,1)))
    test_detect_img = preprocess_img(data[:,:,static_flat])
    res_surface = MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.2, save=False, max_det = 100, # consider 100 as max
                                                verbose=False, classes = 0, device='cpu', agnostic_nms=True, augment=True)
    surface_crop_coords = detect_areas(res_surface[0].summary(), pad_val = 30, # 30 pixels padding, can be changed
                                        img_shape=test_detect_img.shape[0], expected_num=100) # consider 100 as max
    # print(surface_crop_coords)
    view_data = (min_max(data)*255).astype(np.uint8)
    viewer = napari.Viewer()
    viewer.add_image(data = view_data, name = 'whole data')
    for idx, (i_cord, j_cord) in enumerate(surface_crop_coords):
        temp_crop_data = data[:, i_cord:j_cord, :]
        temp_crop_data = (min_max(temp_crop_data)*255).astype(np.uint8)
        viewer.add_image(data = temp_crop_data, name = f'crop {idx}', visible = False)
    del data, view_data, temp_crop_data
    return viewer