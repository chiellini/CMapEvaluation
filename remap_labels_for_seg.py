import os
from tqdm import tqdm
import numpy as np
from glob import glob
import nibabel as nib
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from utils.image_io import check_folder
from utils.image_io import nib_load, nib_save
from utils.utils import relabel_instance, pair_labels

# ==================================
# Relocate data
# ==================================
# sub_folder_name = "2D"
# suffix = "**\*_G.nii.gz"
# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\MannualAnnotations\Finished\2DAnnotation"
# dst_folder = os.path.join(r"D:\ProjectData\CMapEvaluation", sub_folder_name)
# check_folder(dst_folder)
#
# src_files = glob(os.path.join(src_folder, suffix), recursive=True)
# for src_file in tqdm(src_files, desc="Saving to {}".format(dst_folder)):
#     data = nib_load(src_file)
#     data = relabel_instance(data).astype(np.uint16)
#     save_file = os.path.join(dst_folder, os.path.basename(src_file))
#     nib_save(save_file, data)

# =======================================
# Get 2D segmentations corresponding to 2D annotations
# =======================================
# sub_folder_name = "2D"
# suffix = "**\*.nii.gz"
# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\DatasetUpdated\200113plc1p2\SegCellTimeCombinedLabelUnified"
# dst_folder = os.path.join(r"D:\ProjectData\CMapEvaluation", sub_folder_name)
# check_folder(dst_folder)
#
# src_files = glob(os.path.join(src_folder, suffix), recursive=True)
# for src_file in tqdm(src_files, desc="Saving to {}".format(dst_folder)):
#     data = nib_load(src_file)
#     sw, sh, sz = data.shape
#     data[0: sw // 2, :, :] = 0
#     data[(sw // 2) + 1:, :, :] = 0
#     data = relabel_instance(data).astype(np.uint16)
#     save_file = os.path.join(dst_folder, os.path.basename(src_file).replace(".nii.gz", "_128.nii.gz"))
#     nib_save(save_file, data)

# ==================================
# Relocate data
# ==================================
# sub_folder_name = "3DCShaper"
# suffix = "*.nii.gz"
# src_folder = os.path.join(r"D:\ProjectData\CMapEvaluation", sub_folder_name)
# dst_folder = os.path.join(r"D:\ProjectData\CMapEvaluation", sub_folder_name)
# check_folder(dst_folder)
#
# src_files = glob(os.path.join(src_folder, suffix), recursive=False)
# for src_file in tqdm(src_files, desc="Saving to {}".format(dst_folder)):
#     data = nib_load(src_file)
#     data = relabel_instance(data).astype(np.uint16)
#     save_file = os.path.join(dst_folder, os.path.basename(src_file))
#     nib_save(save_file, data)


# ==================================
# Relocate segmentations corresponding to the annotations
# ==================================
# sub_folder_name = "3D"
# TPs = [90, 123, 132, 166, 178, 185]
# suffix = "**\*.nii.gz"
# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\DatasetUpdated\200113plc1p2\SegCellTimeCombinedLabelUnified"
# dst_folder = os.path.join(r"D:\ProjectData\CMapEvaluation", sub_folder_name)
# check_folder(dst_folder)
#
# src_files = glob(os.path.join(src_folder, suffix), recursive=True)
# src_files.sort()
# for tp in tqdm(TPs, desc="Saving to {}".format(dst_folder)):
#     src_file = src_files[tp-1]
#     data = nib_load(src_file)
#     data = relabel_instance(data).astype(np.uint16)
#     save_file = os.path.join(dst_folder, os.path.basename(src_file))
#     nib_save(save_file, data)

# ================================
# remap labels
# ================================



import json


def remap_label_matching_GT(pred_file_path,target_file_path):
    pred_embryo = nib_load(pred_file_path).astype(np.int16)
    target_embryo = nib_load(target_file_path).astype(np.int16)
    pred2target = pair_labels(pred_embryo, target_embryo)
    print('prediction seg to ground truth pair list ', pred2target)

    target_max = target_embryo.max()
    pred_id_list = list(np.unique(pred_embryo))[1:]
    target_id_list = list(np.unique(target_embryo))[1:]

    out = np.zeros_like(pred_embryo)
    left_labels = pred_id_list.copy()
    for pred_id, target_id in pred2target.items():
        overlap_mask = np.logical_and(pred_embryo == pred_id, target_embryo == target_id)
        # # check if all segmented in background
        # overlap_mask_with_0 = np.logical_and(pred_embryo == pred_id, target_embryo == 0)
        #
        #
        # if (overlap_mask_with_0.sum()+3)> (pred_embryo == pred_id).sum():
        #     print(pred_id,target_id)
        #     print(overlap_mask_with_0.sum())
        #     print((pred_embryo == pred_id).sum())
        #     left_labels.remove(pred_id)
        #     this_mapping_dict[int(0)] = int(pred_id)
        #
        #     continue

        if overlap_mask.sum() == 0:
            continue
        left_labels.remove(pred_id)
        out[pred_embryo == pred_id] = target_id
        this_mapping_dict[int(target_id)] = int(pred_id)
    if len(left_labels) > 0:
        for left_label in left_labels:
            target_max += 1
            out[pred_embryo == left_label] = target_max
            this_mapping_dict[int(target_max)] = int(left_label)

    with open(pred_file_path.replace(".nii.gz", ".txt"), 'w') as f:
        f.write(json.dumps(this_mapping_dict))
    save_file = pred_file_path.replace(".nii.gz", "_uni.nii.gz")
    nib_save(save_file, out.astype(np.int16))

source_folder = "F:\CMap_paper\CMapEvaluation"
# # change this when you are working on validation
pred_files_path = glob(os.path.join(source_folder, "Cellpose3d/niigz/*.nii.gz")) # change this when you are working on validation

# change this when you are working on validation
target_files_path = glob(os.path.join(source_folder, "Ground/niigz/*_G.nii.gz"))
pred_files_path.sort()
target_files_path.sort()

assert len(pred_files_path) == len(target_files_path), "#files not equal"

bar = tqdm(range(len(pred_files_path)))
bar.set_description("Uniform labels")
for pred_file_path, target_file_path in zip(pred_files_path, target_files_path):
    this_mapping_dict={}
    pred_embryo = nib_load(pred_file_path).astype(np.uint16)
    target_embryo = nib_load(target_file_path).astype(np.uint16)
    pred2target = pair_labels(pred_embryo, target_embryo)
    print('prediction seg to ground truth pair list ', pred2target)

    target_max = target_embryo.max()
    pred_id_list = list(np.unique(pred_embryo))[1:]
    target_id_list = list(np.unique(target_embryo))[1:]

    out = np.zeros_like(pred_embryo)
    left_labels = pred_id_list.copy()
    for pred_id, target_id in pred2target.items():
        overlap_mask = np.logical_and(pred_embryo == pred_id, target_embryo == target_id)
        # # check if all segmented in background
        # overlap_mask_with_0 = np.logical_and(pred_embryo == pred_id, target_embryo == 0)
        #
        #
        # if (overlap_mask_with_0.sum()+3)> (pred_embryo == pred_id).sum():
        #     print(pred_id,target_id)
        #     print(overlap_mask_with_0.sum())
        #     print((pred_embryo == pred_id).sum())
        #     left_labels.remove(pred_id)
        #     this_mapping_dict[int(0)] = int(pred_id)
        #
        #     continue

        if overlap_mask.sum() == 0:
            continue
        left_labels.remove(pred_id)
        out[pred_embryo == pred_id] = target_id
        this_mapping_dict[int(target_id)]=int(pred_id)
    if len(left_labels) > 0:
        for left_label in left_labels:

            target_max += 1
            out[pred_embryo == left_label] = target_max
            this_mapping_dict[int(target_max)] = int(left_label)

    with open(pred_file_path.replace(".nii.gz", ".txt"), 'w') as f:
        f.write(json.dumps(this_mapping_dict))
    save_file = pred_file_path.replace(".nii.gz", "_uni.nii.gz")
    nib_save(save_file, out.astype(np.uint16))
    bar.update(1)

bar.close()
