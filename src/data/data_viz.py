import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.join(os.getcwd(), os.pardir,os.pardir)
test_load = nib.load(os.path.abspath(parent_dir + '/data/features_CrossSect-5074_LongBLM12-802_LongBLM24-532/MALPEM-ADNI_002_S_0295_MR_MPR_GradWarp_B1_Correction_N3_Scaled_Br_S67612_I150177.nii.gz'))

print(test_load.shape)
