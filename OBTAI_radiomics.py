#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Process, Queue, freeze_support

import radiomics
from radiomics import featureextractor

# ==============================================================================
#### STEP 1: Setting routes and outfiles
# ==============================================================================
# 
BASE_PATH_IMG = "/DIR/IMGS_NORM/"
BASE_PATH_MSK = "/DIR/SEG_MASK/"

OUTPUT_CSV  = "Dataset_Radiomics.csv"
LOG_ERRORS = "Log_Errors_PyRadiomics.txt"

# ==============================================================================
# STEP 2:  IBSI COMPLIANT: Setting Parameters
# ==============================================================================
SETTINGS_ZSCORE = {
    'binWidth': 0.06,              
    'resampledPixelSpacing': None, 
    'interpolator': 'sitkBSpline',
    'geometryTolerance': 0.01,
    'additionalInfo': True,
    'force2D': False,              
    'label': 1                    
}

def register_log(message):
    print(f"{message}")
    with open(LOG_ERRORS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# ==============================================================================
# STEP 3: Geometric 3D Validation MASK 
# ==============================================================================
def is_mask_3d_valid(arr_binary):
    coords = np.argwhere(arr_binary > 0)
    if len(coords) < 2:
        return False
    
    extension_axis = coords.max(axis=0) - coords.min(axis=0) + 1
    
    axis_in_space = np.sum(extension_axis >= 2)
    return axis_in_space >= 2


# ==============================================================================
# STEP 4: Reading lesion sub regions and id: Reading pair nm
# ==============================================================================
def process_case_lesions(img_path, msk_path):
    taks = []
    
    img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    msk_sitk = sitk.ReadImage(msk_path, sitk.sitkInt16)
    
    msk_array = sitk.GetArrayFromImage(msk_sitk)
    labels_in = np.unique(msk_array[msk_array > 0])
    
    if len(labels_in) == 0:
        return taks

    lesion_ids = sorted(list(set([int(l // 10) for l in labels_in])))
    
    for n in lesion_ids:
        val_contrast = n * 10 + 1  # Ej: 11 para lesión 1
        val_necrosis  = n * 10 + 2  # Ej: 12 para lesión 1
        
        is_contrast = val_contrast in labels_in
        is_necrosis  = val_necrosis in labels_in
        
    
        if is_necrosis and not is_contrast:
            register_log(f"Message")
        
        # ---------------------------------------------------------------------
        # ----------------------------------------------------------------------
        if is_contrast:
            arr_c = (msk_array == val_contrast).astype(np.uint8)
            cant_vox = np.count_nonzero(arr_c)
            
            if cant_vox > 1 and is_mask_3d_valid(arr_c):
                img_c = sitk.GetImageFromArray(arr_c)
                img_c = sitk.Cast(img_c, sitk.sitkUInt8)
                img_c.CopyInformation(img_sitk)
                
                taks.append({
                    'ID_Lesion': n,
                    'Region': 'contrast',
                    'img_sitk': img_sitk,
                    'mask_sitk': img_c
                })
            else:
                register_log(f"Message")

        # ----------------------------------------------------------------------
        #
        # ----------------------------------------------------------------------
        if is_necrosis:
            arr_n = (msk_array == val_necrosis).astype(np.uint8)
            cant_vox = np.count_nonzero(arr_n)
            
            if cant_vox > 1 and is_mask_3d_valid(arr_n):
                img_n = sitk.GetImageFromArray(arr_n)
                img_n = sitk.Cast(img_n, sitk.sitkUInt8)
                img_n.CopyInformation(img_sitk)
                
                taks.append({
                    'ID_Lesion': n,
                    'Region': 'necrosis',
                    'img_sitk': img_sitk,
                    'mask_sitk': img_n
                })
            else:
                register_log(f"Message")

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        arr_all = ((msk_array // 10 == n) | (msk_array == n)).astype(np.uint8)
        cant_vox_all = np.count_nonzero(arr_all)
        
        if cant_vox_all > 1 and is_mask_3d_valid(arr_all):
            img_all = sitk.GetImageFromArray(arr_all)
            img_all = sitk.Cast(img_all, sitk.sitkUInt8)
            img_all.CopyInformation(img_sitk)
            
            taks.append({
                'ID_Lesion': n,
                'Region': 'all',
                'img_sitk': img_sitk,
                'mask_sitk': img_all
            })

    return taks

# ==============================================================================
# STEP 5 + 6: MAIN
# ==============================================================================
if __name__ == '__main__':
    
    
    if os.path.exists(OUTPUT_CSV):
        print("Resume")
        df_previous = pd.read_csv(OUTPUT_CSV)
        processed = set(zip(df_previous['ID_Patient'], df_previous['File'], df_previous['ID_Lesion'], df_previous['Region']))
        results_master = df_previous.to_dict('records')
    else:
        print("Message CSV")
        processed, results_master = set(), []

    cases = sorted([d for d in os.listdir(BASE_PATH_IMG) if d.startswith("OBTAI")])

    for pac in cases:
        path_pac_img = os.path.join(BASE_PATH_IMG, pac)
        path_pac_msk = os.path.join(BASE_PATH_MSK, pac)
        
        if not os.path.isdir(path_pac_img): 
            continue
            
        files_img = sorted([f for f in os.listdir(path_pac_img) if f.endswith("_img.nii") or f.endswith("_img.nii.gz")])
        
        for f_img in files_img:
            f_msk = f_img.replace("_img", "_msk")
            full_path_img = os.path.join(path_pac_img, f_img)
            full_path_msk = os.path.join(path_pac_msk, f_msk)
            
            if not os.path.exists(full_path_msk):
                register_log(f"Message")
                continue
            
            try:
                taks_case = process_case_lesions(full_path_img, full_path_msk)
                
                for t in taks_case:
                    n_id = t['ID_Lesion']
                    r_type = t['Region']
                    
                   
                    if (pac, f_img, n_id, r_type) in processed:
                        continue
                        
                    print(f"Message")
                    
                    try:
                        current_settings = SETTINGS_ZSCORE.copy()
                        current_settings['label'] = 1
                        
                        extractor = featureextractor.RadiomicsFeatureExtractor(**current_settings)
                        extractor.disableAllImageTypes()
                        extractor.enableImageTypeByName('Original')
                        
                        extractor.disableAllFeatures()
                        extractor.enableFeatureClassByName('shape')
                        extractor.enableFeaturesClassByName('firstorder')
                        extractor.enableFeaturesClassByName('glcm')
                        extractor.enableFeaturesClassByName('glrlm')
                        extractor.enableFeaturesClassByName('glszm')
                        extractor.enableFeaturesClassByName('gldm')
                        extractor.enableFeaturesClassByName('ngtdm')
                        
                        features = extractor.execute(['img_sitk'], t['mask_sitk'])
                        clean_features = {k: v for k, v in features.items() if not k.startswith('diagnostics_')}
                        
                        register = {
                            'ID_Patient': pac,
                            'File': f_img,
                            'ID_Lesion': n_id,
                            'Region': r_type
                        }
                        register.update(clean_features)
                        results_master.append(register)
                        
                        #
                        pd.DataFrame(results_master).to_csv(OUTPUT_CSV, index=False)
                        
                    except Exception as ext_err:
                        register_log(f"Message")
                        
                gc.collect()
                
            except Exception as e:
                register_log(f"Message")

    print(f"DONE")