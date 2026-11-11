# OBTAI_radiomics_project
A radiomic-based analysis pipeline on MRIs and their 3D-lesion masks of brain metastases


The feature extraction pipeline is implemented on a sequential structure designed to handle multi-focus tumour segmentations. Labels encode the lesion identifier (n) followed by a region digit (m): 1 denotes contrast-enhancing tissue and 2 denotes non-enhancing or necrotic tissue. Managing the "nm" schema. Lesion identity is retained across follow-up examinations.

The script reads the root input directories (BASE_PATH_IMG and BASE_PATH_MSK) to identify case-specific subdirectories designated by the OBTAI prefix. For each case, it reads the longitudinal time-point imaging files (_img.nii) matching them with their corresponding segmentation masks (_msk.nii). The longitudinal time-point imaging files are identified following the ISO 8601 basic format. 

The segmentation mask is loaded as an array. The script decodes the "nm" multi-focus labelling scheme. Unique lesion ids are programmatically extracted using the integer division.

A geometric 3D validation filtering is performed in order to avoid mathematical degeneracy in texture matrices, each region in the mask has a geometric validation before extraction. The bounding box of each mask is analysed across its spatial axes (Z,Y,X), requiring a minimum spatial extent of at least two voxels to support 3D-radiomic extraction.
