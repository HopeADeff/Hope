# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

block_cipher = None

hidden_imports = []
hidden_imports += collect_submodules('diffusers')
hidden_imports += collect_submodules('transformers')
hidden_imports += collect_submodules('accelerate')
hidden_imports += collect_submodules('scipy')
hidden_imports += ['scipy.special', 'cython', 'sklearn.utils._typedefs', 'huggingface_hub']

datas = []
datas += collect_data_files('diffusers')
datas += collect_data_files('transformers')
datas += collect_data_files('accelerate')

if os.path.exists('assets/models'):
    datas += [('assets/models', 'assets/models')]
else:
    print("WARNING: 'assets/models' not found. You must run download_models.py first!")

a = Analysis(
    ['Hope/Hope/engine.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='engine',
)
