# -*- mode: python ; coding: utf-8 -*-

import os
import sys
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
    [],
    exclude_binaries=True,
    name='OCT_mac_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCT_mac_app',
)
app = BUNDLE(
    coll,
    name='OCT_mac_app.app',
    icon=None,
    bundle_identifier=None,
)
