import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from utils.image_io import nib_load, nib_save, check_folder
from utils.heatmap import HeatMap

# ========================
# 2D error map
# ========================
embryo_this_name='200109plc1p1'
target_folder = r"F:\CMap_paper\CMapEvaluation\2Duni"
pred_folder = r"F:\CMap_paper\CMapEvaluation\2Duni"
target_files = glob.glob(os.path.join(target_folder, "{}_*_G.nii.gz".format(embryo_this_name)))
pred_files =glob.glob(os.path.join(pred_folder, "{}*_uni.nii.gz".format(embryo_this_name)))

target_files.sort()
pred_files.sort()

assert len(pred_files) == len(target_files), "#pred != #target"
bar = tqdm(range(len(pred_files)))
bar.set_description("Error map of 2D")
# sw, sh, sz = nib_load(pred_files[0]).shape
# sw_slice = int(sw / 2)
for pred_file, target_file in zip(pred_files, target_files):
    bar.update(1)
    pred_arr = nib_load(pred_file)
    target_arr = nib_load(target_file)
    # print(pred.shape,target.shape)
    # quit(0)
    pred_arr = np.transpose(pred_arr, (1, 0))
    target_arr = np.transpose(target_arr, (1, 0))
    error_arr = (pred_arr != target_arr) * 1.0
    background_arr=((pred_arr==0)&(target_arr==0))*1.0
    # error = gaussian_filter(error, sigma=4)
    if error_arr.max() > 0:
        error_arr = error_arr / 5.0
    # error = (error * 255).astype(np.uint8)

    heatmap = HeatMap((target_arr != 0).astype(np.uint8), error_arr,background_arr, gaussian_std=0)
    heatmap.save(filename=os.path.basename(pred_file).split(".")[0],
                 format="png",
                 save_path="Results/2DErrorMap/{}".format(embryo_this_name),
                 transparency=0.7,
                 color_map='seismic')


# # ========================
# # 3D error volume
# # ========================
# target_folder = r"F:\CMap_paper\CMapEvaluation\Ground\niigz"
# pred_folder = r"F:\CMap_paper\CMapEvaluation\CMap\niigz"
# target_files = glob.glob(os.path.join(target_folder, "*_G.nii.gz"))
# pred_files =glob.glob(os.path.join(pred_folder, "*_uni.nii.gz"))
# # print(target_files,pred_files)
#
# target_files.sort()
# target_files = target_files[4:5]
# pred_files.sort()
# pred_files = pred_files[4:5]
# print(target_files,pred_files)
# # quit(0)
#
# assert len(pred_files) == len(target_files), "#pred != #target"
# bar = tqdm(range(len(pred_files)))
# bar.set_description("Error map of 3D")
# sw, sh, sz = nib_load(pred_files[0]).shape
# for pred_file, target_file in zip(pred_files, target_files):
#     save_folder = os.path.join("./Results/3DErrorMap", os.path.basename(pred_file).split(".")[0])
#     check_folder(save_folder)
#
#     pred = nib_load(pred_file)
#     target = nib_load(target_file)
#     error = (pred != target) * 1.0
#     # print(error,error.shape)
#     # error = gaussian_filter(error, sigma=4)
#
#     if error.max() > 0:
#         error = error / 5.0
#     # error = (error * 255).astype(np.uint8)
#     depth = error.shape[0]
#     for i_depth in range(depth):
#         pred0=np.transpose(pred[i_depth], (1, 0))
#         error0 = np.transpose(error[i_depth], (1, 0))
#         target0 = np.transpose(target[i_depth], (1, 0))
#         background_arr = ((pred0 == 0) & (target0 == 0)) * 1.0
#         heatmap = HeatMap((target0!=0).astype(np.uint8), error0,background_arr, gaussian_std=0)
#         heatmap.save(filename=os.path.basename(pred_file).split(".")[0] + "_{}".format(str(i_depth).zfill(3)),
#                      format="png",
#                      save_path=save_folder,
#                      transparency=0.7,
#                      color_map='seismic',
#                      width_pad=0)
#     bar.update(1)

