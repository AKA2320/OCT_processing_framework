# -*- mode: python ; coding: utf-8 -*-

import sys
import os
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

# Get the site-packages path dynamically
site_path = [i for i in sys.path if "site-packages" in i][0]

# Construct paths using os.path.join for better cross-platform compatibility
napari_path = os.path.join(site_path, "napari")
vispy_path = os.path.join(site_path, "vispy")

config_py_path = 'config_transmorph.py'
funcs_transmorph_py_path = 'funcs_transmorph.py'
datapaths_yaml_path = 'datapaths.yaml'
models_path = 'models'
pyside_gui_py_path = 'pyside_gui.py'

a = Analysis(
    [pyside_gui_py_path],
    pathex=[],
    binaries=[],
    # Include your project's specific data files and now also napari/vispy
    datas=[
        (models_path, 'models'),
        (config_py_path, '.'),
        (funcs_transmorph_py_path, '.'),
        (datapaths_yaml_path, '.'),
        (napari_path, 'napari'), # Re-added napari
        (vispy_path, 'vispy')    # Re-added vispy
    ],
    hiddenimports=[
        'GUI_scripts', 'utils', 'pydicom', 'skimage', 'scipy', 'napari',
        'napari.view_layers', 'torch', 'PySide6', 'ultralytics', 'h5py',
        'natsort', 'torchvision', 'tqdm', 'timm', 'ml_collections'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['ipykernel','jupyter_client','IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries, # Include binaries from analysis (e.g., PySide6 DLLs)
    a.datas,    # Include data files from analysis
    [],
    name='OCT_windows_app', # Changed name for Windows executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Keep False for a GUI application
    disable_windowed_traceback=False,
    target_arch=None,
)
