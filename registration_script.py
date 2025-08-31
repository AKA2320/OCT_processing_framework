import sys
# import matplotlib.pylab as plt
# import numpy as np
import os
# from skimage.transform import warp, AffineTransform
from natsort import natsorted
from tqdm import tqdm
# import h5py
# import shutil
from ultralytics import YOLO
from utils.util_funcs import resource_path
from registration_scripts.reg_worker import RegistrationWorker
import yaml
import torch
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RegistrationMaster:
    def __init__(self, config_path='datapaths.yaml', is_gui=False, ENABLE_MULTIPROC_SLURM = False,
                 DISABLE_TQDM = True, EXPECTED_SURFACES = 2, EXPECTED_CELLS = 2, BATCH_FLAG=False,
                 DATA_LOAD_DIR = None, DATA_SAVE_DIR = None, USE_MODEL_LATERAL_TRANSLATION = False, 
                 SAVE_DETECTIONS = False,):
        """Initialize the registration pipeline with configuration and models"""
        self.is_gui_flag = is_gui
        if not self.is_gui_flag:
            self.config = self._load_config(config_path)
        else:
            self.config = self._set_config(config_path, DATA_LOAD_DIR, DATA_SAVE_DIR,
                                            EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG, DISABLE_TQDM,
                                            ENABLE_MULTIPROC_SLURM, USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BATCH_FLAG = self.config['FLAGS']['BATCH_FLAG']
        self.DATA_LOAD_DIR = self.config['PATHS']['DATA_LOAD_DIR']
        self.DATA_SAVE_DIR = self.config['PATHS']['DATA_SAVE_DIR']
        self.MODEL_FEATURE_DETECT_PATH = self.config['PATHS']['MODEL_FEATURE_DETECT_PATH']
        self.MODEL_X_TRANSLATION_PATH = self.config['PATHS']['MODEL_X_TRANSLATION_PATH']
        self.DISABLE_TQDM = self.config['FLAGS']['DISABLE_TQDM']
        self.USE_MODEL_LATERAL_TRANSLATION = self.config['FLAGS']['USE_MODEL_LATERAL_TRANSLATION']
        self.ENABLE_MULTIPROC_SLURM = self.config['FLAGS']['ENABLE_MULTIPROC_SLURM']
        self.save_detections = self.config['FLAGS']['SAVE_DETECTIONS']
        self.models = self._load_models()
        self.EXPECTED_SURFACES = self.config['VALUES']['EXPECTED_SURFACES']
        self.EXPECTED_CELLS = self.config['VALUES']['EXPECTED_CELLS']
        self.data_type = None

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _set_config(self, config_path, DATA_LOAD_DIR, DATA_SAVE_DIR,
                    EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG, DISABLE_TQDM,
                    ENABLE_MULTIPROC_SLURM, USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS):
        """Set the configuration from YAML file, as we take inputs from GUI"""
        try:
            with open(resource_path(config_path), 'r') as f:
                config = yaml.safe_load(f)
        except:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        # Paths
        config['PATHS']['DATA_LOAD_DIR'] = DATA_LOAD_DIR
        config['PATHS']['DATA_SAVE_DIR'] = DATA_SAVE_DIR
        # Valuues
        config['VALUES']['EXPECTED_SURFACES'] = EXPECTED_SURFACES
        config['VALUES']['EXPECTED_SURFACES'] = EXPECTED_CELLS
        # Flags
        config['FLAGS']['BATCH_FLAG'] = BATCH_FLAG
        config['FLAGS']['DISABLE_TQDM'] = DISABLE_TQDM
        config['FLAGS']['ENABLE_MULTIPROC_SLURM'] = ENABLE_MULTIPROC_SLURM
        config['FLAGS']['USE_MODEL_LATERAL_TRANSLATION'] = USE_MODEL_LATERAL_TRANSLATION
        config['FLAGS']['SAVE_DETECTIONS'] = SAVE_DETECTIONS
        try:
            config['PATHS']['MODEL_FEATURE_DETECT_PATH'] = resource_path(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
            config['PATHS']['MODEL_X_TRANSLATION_PATH'] = resource_path(config['PATHS']['MODEL_X_TRANSLATION_PATH'])
        except:
            config['PATHS']['MODEL_FEATURE_DETECT_PATH'] = config['PATHS']['MODEL_FEATURE_DETECT_PATH']
            config['PATHS']['MODEL_X_TRANSLATION_PATH'] = config['PATHS']['MODEL_X_TRANSLATION_PATH']
        return config

    def _load_models(self):
        """Load models based on configuration"""
        models = {}
        try:
            models['feature_yolo'] = YOLO(self.MODEL_FEATURE_DETECT_PATH)
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}", exc_info=True)
            sys.exit("Failed to load YOLO model. Exiting.")
        if self.USE_MODEL_LATERAL_TRANSLATION:
            try:
                MODEL_X_TRANSLATION = torch.load(self.MODEL_X_TRANSLATION_PATH, map_location=self.DEVICE, weights_only=False)
                MODEL_X_TRANSLATION.eval()
                logging.info("Model X loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading Model X: {e}", exc_info=True)
                logging.info("Proceeding without Model X translation.")
                MODEL_X_TRANSLATION = None
        else:
            MODEL_X_TRANSLATION = None
        models['x_translation'] = MODEL_X_TRANSLATION
        return models
        
    def spawn_worker_and_run_pipeline(self):
        self.DATA_LOAD_DIR = self.DATA_LOAD_DIR
        if self.DATA_LOAD_DIR.endswith('/'):
            self.DATA_LOAD_DIR = self.DATA_LOAD_DIR[:-1]
        if not self.BATCH_FLAG:
            if self.DATA_LOAD_DIR.lower().endswith('.h5'):
                self.data_type = 'h5'
                scans = [self.DATA_LOAD_DIR.split('/')[-1].removesuffix('.h5')]
            else:
                self.data_type = 'dcm'
                scans = [self.DATA_LOAD_DIR.split('/')[-1]]
        elif self.BATCH_FLAG:
            scans = [i for i in os.listdir(self.DATA_LOAD_DIR) if (i.startswith('scan'))]
            scans = natsorted(scans)
            first_path = os.listdir(os.path.join(self.DATA_LOAD_DIR,scans[0]))[0]
            if first_path.endswith('.h5'):
                self.data_type = 'h5'
            else:
                self.data_type = 'dcm'

        pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable = self.DISABLE_TQDM)
        if not self.ENABLE_MULTIPROC_SLURM:
            for scan_num in pbar:
                pbar.set_description(desc = f'Processing {scan_num}')
                start = time.time()
                self._launch_process_wrapper(scan_num, pbar)
                # scan_worker = RegistrationWorker()
                # scan_worker.process_scan()
                logging.info(f'Time taken: {time.time() - start:.2f} seconds')

        elif self.ENABLE_MULTIPROC_SLURM:
            self.DISABLE_TQDM = True
            pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable = self.DISABLE_TQDM)
            self._dask_slurm_spawner(pbar, scans)

    def init_dask_worker(self):
        self.MODEL_FEATURE_DETECT = self.models['feature_yolo']
        self.MODEL_X_TRANSLATION = self.models['x_translation']

    def _dask_slurm_spawner(self, pbar, scans):
        try:
            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client, wait
            from dask import delayed
        except ImportError as e:
            raise ImportError("Dask and dask_jobqueue modules are required for multiprocessing with SLURM") from e
        multiproc_args_list = [(scan_num, pbar) for scan_num in scans]
        logging.info("Setting up Dask SLURM cluster...")
        cluster = SLURMCluster(
            queue='general',
            account='ACC_NUMBER',
            cores=1, 
            processes=1,
            memory='10GB',
            walltime='03:00:00',
            job_extra_directives=[
                "--cpus-per-task=1",
                "--nodes=1",
                "--job-name=oct_reg",
                "--output=my_job.out",
                "--error=my_job.err"
            ],
            python=sys.executable,
        )
        cluster.scale(jobs=len(scans))
        # Attach client
        client = Client(cluster)
        client.run(self.init_dask_worker)
        tasks = [delayed(self._launch_process_wrapper)(*args) for args in multiproc_args_list]
        logging.info("Submitting tasks to the cluster...")
        results = client.compute(*tasks)
        wait(results)
        logging.info("Jobs DONE")
        # results = client.gather(results)
        try:
            client.close()
            cluster.close()
        except:
            pass
    
    def _launch_process_wrapper(self, scan_num, pbar):
        scan_worker = RegistrationWorker(self.config, self.models, scan_num, pbar, self.DATA_LOAD_DIR, 
                                        self.data_type, self.save_detections, self.DEVICE, self.DISABLE_TQDM,
                                        self.EXPECTED_SURFACES, self.EXPECTED_CELLS)
        scan_worker.process_scan()

'''
# class RegistrationWorker:
#     def __init__(self, config, models, scan_num, pbar, DATA_LOAD_DIR,
#                 data_type, save_detections, DEVICE, DISABLE_TQDM,
#                 EXPECTED_SURFACES, EXPECTED_CELLS):
#         self.config = config
#         self.MODEL_FEATURE_DETECT = models['feature_yolo']
#         self.MODEL_X_TRANSLATION = models['x_translation']
#         self.scan_num = scan_num
#         self.pbar = pbar
#         self.DATA_LOAD_DIR = DATA_LOAD_DIR
#         self.data_type = data_type
#         self.save_detections = save_detections
#         self.DEVICE = DEVICE
#         self.DISABLE_TQDM = DISABLE_TQDM
#         # Paths
#         self.DATA_SAVE_DIR = self.config['PATHS']['DATA_SAVE_DIR']
#         # Values
#         self.SURFACE_Y_PAD = self.config['VALUES']['SURFACE_Y_PAD']
#         self.SURFACE_X_PAD = self.config['VALUES']['SURFACE_X_PAD'] 
#         self.CELLS_X_PAD = self.config['VALUES']['CELLS_X_PAD'] 
#         self.EXPECTED_SURFACES = EXPECTED_SURFACES
#         self.EXPECTED_CELLS = EXPECTED_CELLS

#     def _load_data(self):
#         """Load data based on type (h5 or dcm)"""
#         if self.data_type == 'h5':
#             if not os.path.exists(self.DATA_LOAD_DIR):
#                 raise FileNotFoundError(f"Scan {self.DATA_LOAD_DIR} not found")
#             return load_h5_data(self.DATA_LOAD_DIR, self.scan_num)
#         elif self.data_type == 'dcm':
#             if not os.path.exists(self.DATA_LOAD_DIR):
#                 raise FileNotFoundError(f"Directory {self.DATA_LOAD_DIR} not found")
#             return load_data_dcm(self.DATA_LOAD_DIR, self.scan_num)
#         else:
#             raise ValueError("Unsupported data type. Use 'h5' or 'dcm'.")
    
#     def process_scan(self):
#         """Process a single scan through the full registration pipeline"""
#         try:
#             # Load data
#             self.pbar.set_description(desc = f'Loading data for {self.scan_num}')
#             original_data = self._load_data()
#             if original_data is None:
#                 raise ValueError("Data loading returned None, check the input paths.")

#             # Feature detection and initial cropping
#             self.pbar.set_description(desc = f'Cropping data for {self.scan_num}')
#             cropped_data = self._detect_features_crop_data(original_data)
#             if cropped_data is None:
#                 return None
#             del original_data  # Free memory, we now only use cropped_data
#             # Save unregistered data
#             self._save_data(cropped_data, '_unregistered')

#             # Get surface coordinates
#             self.surface_coords, self.partition_coord = self._get_surface_coords_cropped(cropped_data)
#             if self.surface_coords is None:
#                 logging.warning(f'No surface detected in cropped data, cropping error: {self.scan_num}')
#                 return None

#             # Data processing pipeline
#             cropped_data = self._process_data_pipeline(cropped_data)

#             # Save final registered data
#             self._save_data(cropped_data, '_registered')
#             return cropped_data

#         except Exception as e:
#             logging.error(f"Error processing scan {self.scan_num}: {e}", exc_info=True)
#             return None

    
#     def _detect_features_crop_data(self, original_data):
#         """Detect features using YOLO model and crop data"""
#         static_flat = np.argmax(np.sum(original_data[:,:,:], axis=(0,1)))
#         test_detect_img = preprocess_img(original_data[:,:,static_flat])
#         detections_save_dir = os.path.join(self.DATA_SAVE_DIR, 'detections')
#         if os.path.exists(os.path.join(detections_save_dir,self.scan_num)):
#             try:
#                 shutil.rmtree(os.path.join(detections_save_dir,self.scan_num))
#             except OSError as e:
#                 print(f"Error removing existing detection directory: {e}")
#         res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=self.save_detections, max_det = self.EXPECTED_SURFACES,
#                                                         project = detections_save_dir, name=self.scan_num, verbose=False,
#                                                         classes=[0,1], device=self.DEVICE, agnostic_nms=True, augment=True)
        
#         surface_crop_coords = [i for i in res_surface[0].summary() if i['name']=='surface']
#         cells_crop_coords = [i for i in res_surface[0].summary() if i['name']=='cells']
        
#         surface_crop_coords = detect_areas(surface_crop_coords, pad_val = self.SURFACE_Y_PAD,
#                                            img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_SURFACES)
#         cells_crop_coords = detect_areas(cells_crop_coords, pad_val = self.SURFACE_Y_PAD,
#                                          img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_CELLS)
#         if surface_crop_coords is None:
#             logging.warning(f'No surface detected in orignal data for cropping: {self.scan_num}')
#             return None
#         return crop_data(original_data, surface_crop_coords, cells_crop_coords, original_data.shape[1])

#     def _get_surface_coords_cropped(self, cropped_data):
#         """Get surface coordinates and partition coordinates from cropped data"""
#         static_flat = np.argmax(np.sum(cropped_data[:,:,:], axis=(0,1)))
#         test_detect_img = preprocess_img(cropped_data[:,:,static_flat])
        
#         res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False,max_det = self.EXPECTED_SURFACES,
#                                                         classes=0, device=self.DEVICE, agnostic_nms=True, augment=True)
        
#         surface_coords = detect_areas(res_surface[0].summary(), pad_val= self.SURFACE_Y_PAD,
#                                     img_shape= test_detect_img.shape[0], expected_num= self.EXPECTED_SURFACES)
        
#         if surface_coords is None:
#             return None
            
#         if self.EXPECTED_SURFACES > 1:
#             # We only flatten/correct Y motion in standard interference
#             # partition_coord tells us where to split the data
#             partition_coord = np.ceil(np.mean(np.mean(surface_coords[-2:], axis=1))).astype(int) 
#         else:
#             partition_coord = None
#         return surface_coords, partition_coord

#     def _process_data_pipeline(self, cropped_data):
#         """Process data through flattening and motion correction pipeline"""

#         # Flatten data
#         self.pbar.set_description(desc = f'Flattening {self.scan_num} surfaces.....')
#         cropped_data = self._flatten_data(cropped_data)
        
#         # Correct Y motion
#         self.pbar.set_description(desc = f'Correcting {self.scan_num} Y-Motion.....')
#         cropped_data = self._correct_y_motion(cropped_data)
        
#         # Correct X motion
#         self.pbar.set_description(desc = f'Correcting {self.scan_num} X-Motion.....')
#         cropped_data = self._correct_x_motion(cropped_data)
        
#         return cropped_data

#     def _flatten_data(self, data):
#         """Flatten the data based on surface coordinates"""
#         top_surf = True
        
#         if self.surface_coords.shape[0] > 1:
#             for _ in range(2):
#                 if top_surf:
#                     data = flatten_data(data, self.surface_coords[:-1], top_surf,
#                                          self.partition_coord, self.DISABLE_TQDM, self.scan_num)
#                 else:
#                     data = flatten_data(data, self.surface_coords[-1:], top_surf,
#                                          self.partition_coord, self.DISABLE_TQDM, self.scan_num)
#                 top_surf = False
#         else:
#             data = flatten_data(data, self.surface_coords, top_surf, self.partition_coord, self.DISABLE_TQDM, self.scan_num)
            
#         return data

#     def _correct_y_motion(self, data):
#         """Correct Y-axis motion in the data"""
#         top_surf = True
#         if self.surface_coords.shape[0] > 1:
#             for _ in range(2):
#                 if top_surf:
#                     data = y_motion_correcting(data, self.surface_coords[:-1], top_surf,
#                                                 self.partition_coord, self.DISABLE_TQDM, self.scan_num)
#                 else:
#                     data = y_motion_correcting(data, self.surface_coords[-1:], top_surf,
#                                                 self.partition_coord, self.DISABLE_TQDM, self.scan_num)
#                 top_surf = False
#         else:
#             data = y_motion_correcting(data, self.surface_coords, top_surf, self.partition_coord,
#                                         self.DISABLE_TQDM, self.scan_num)
#         return data

#     def _correct_x_motion(self, data):
#         """Correct X-axis motion in the data"""
#         static_flat = np.argmax(np.sum(data[:,self.surface_coords[0,0]:self.surface_coords[0,1],:], axis=(0,1)))
#         # static_flat = np.argmax(np.sum(data[:,:,:], axis=(0,1)))
#         test_detect_img = preprocess_img(data[:,:,static_flat])
        
#         res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False, max_det = self.EXPECTED_SURFACES,
#                                                         classes=0, device=self.DEVICE, agnostic_nms=True, augment=True)
#         res_cells = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False, max_det = self.EXPECTED_CELLS,
#                                                         classes=1, device=self.DEVICE, agnostic_nms=True, augment=True)
        
#         surface_coords_for_x = detect_areas(res_surface[0].summary(), pad_val=self.SURFACE_X_PAD,
#                                             img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_SURFACES)
#         cells_coords_for_x = detect_areas(res_cells[0].summary(), pad_val=self.CELLS_X_PAD,
#                                             img_shape=test_detect_img.shape[0],expected_num=self.EXPECTED_CELLS)
        
#         if (cells_coords_for_x is None) and (surface_coords_for_x is None):
#             logging.warning(f'No surface or cells detected: {self.scan_num}')
#             return None
            
#         enface_extraction_rows = []
#         if surface_coords_for_x is not None:
#             static_y_motion = np.argmax(np.sum(data[:,surface_coords_for_x[0,0]:surface_coords_for_x[0,1],:], axis=(1,2)))    
#             errs = []
#             for i in range(data.shape[0]):
#                 errs.append(ncc(data[static_y_motion,:,:], data[i,:,:]))
#             errs = np.squeeze(errs)
#             valid_args = np.squeeze(np.argwhere(errs>0.7))
#             for i in range(surface_coords_for_x.shape[0]):
#                 val = np.argmax(np.sum(np.max(data[:,surface_coords_for_x[i,0]:surface_coords_for_x[i,1],:], axis=0), axis=1))
#                 enface_extraction_rows.append(surface_coords_for_x[i,0]+val)
#         else:
#             valid_args = np.arange(data.shape[0])

#         tr_all = all_trans_x(data, cells_coords_for_x, valid_args, enface_extraction_rows,
#                             self.DISABLE_TQDM, self.scan_num, self.MODEL_X_TRANSLATION)
        
#         for i in tqdm(range(1, data.shape[0], 2),desc='X-motion warping',
#                       disable=self.DISABLE_TQDM,ascii="░▖▘▝▗▚▞█", leave=False):
#             data[i] = warp(data[i], AffineTransform(matrix=tr_all[i]), order=3)
#         return data

#     def _save_data(self, data, suffix=''):
#         """Save processed data to HDF5 file"""
#         if data.dtype != np.float64:
#             data = data.astype(np.float64)
#         folder_save = self.DATA_SAVE_DIR
#         if not folder_save.endswith('/'):
#             folder_save = folder_save + '/'
            
#         os.makedirs(folder_save, exist_ok=True)
#         hdf5_filename = f'{folder_save}{self.scan_num}{suffix}.h5'
#         with h5py.File(hdf5_filename, 'w') as hf:
#             hf.create_dataset('volume', data=data, compression='gzip', compression_opts=5)
'''

from multiprocessing import Process

def start_reg_cli():
    registration_pipeline = RegistrationMaster(config_path = 'datapaths.yaml', is_gui = False)
    registration_pipeline.spawn_worker_and_run_pipeline()

def start_reg_gui(DATA_LOAD_DIR, DATA_SAVE_DIR, EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG,
                 USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS):
    registration_pipeline = RegistrationMaster(config_path = 'datapaths.yaml', is_gui = True, DATA_LOAD_DIR = DATA_LOAD_DIR,
                            DATA_SAVE_DIR = DATA_SAVE_DIR, EXPECTED_SURFACES = EXPECTED_SURFACES, 
                            EXPECTED_CELLS = EXPECTED_CELLS, BATCH_FLAG = BATCH_FLAG, DISABLE_TQDM = True,
                            ENABLE_MULTIPROC_SLURM = False, USE_MODEL_LATERAL_TRANSLATION = USE_MODEL_LATERAL_TRANSLATION,
                            SAVE_DETECTIONS = SAVE_DETECTIONS) 
    registration_pipeline.spawn_worker_and_run_pipeline()

if __name__ == "__main__":
    # Initialize RegistrationMaster class
    p = Process(target = start_reg_cli)
    p.start()
    p.join()
    