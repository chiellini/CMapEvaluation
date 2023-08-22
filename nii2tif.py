import os
from glob import glob
import pandas as pd
import math
import shutil
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from PIL import Image
from skimage.transform import resize
from utils.image_io import nib_load, nib_save
from utils.utils import scale2index, save_indexed_tif,save_tif

#  =====================
#  save cells at particular frames
#  =====================
# cell_name = "ABar"
# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\Code\Evaluation\Results\3DErrorVolume"
# dst_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\Code\Evaluation\Results\3DErrorVolumeTif"
#
# # read name dictionary
#
src_folder = r"D:\ProjectData\CMapEvaluation\3D"
dst_folder = r"D:\ProjectData\CMapEvaluation\3D\tif"
seg_uni_files = sorted(glob(os.path.join(src_folder, "*_segCell_uni.nii.gz")))
# seg_gt_files = sorted(glob(os.path.join(src_folder, "*segCell_G.nii.gz")))

# new labels file
label_file = os.path.join(os.path.dirname(dst_folder), os.path.basename(dst_folder) + ".txt")
# open(label_file, "w").close()

for idx in tqdm(range(len(seg_uni_files)), desc=f"Saving to {dst_folder}"):
    seg_file = seg_uni_files[idx]
    # seg_file = seg_gt_files[tp]

    seg = nib_load(seg_file)
    # target_shape = [int(x / 2) for x in seg.shape]
    # seg = resize(image=seg, output_shape=target_shape, preserve_range=True, order=0).astype(np.uint8)

    seg = scale2index(seg) # to 0-255
    save_file = os.path.join(dst_folder, os.path.basename(seg_file).split(".")[0] + ".tif")
    save_indexed_tif(save_file, seg)
    print(np.unique(seg))
    # save_tif(save_file, seg)

    # save label anchor
    # print('label anchor',np.unique(seg).tolist())
    label_anchor = np.unique(seg).tolist()[-1]
    with open(label_file, "a") as f:
        f.write(f"{label_anchor}\n")
