import os
from tqdm import tqdm
import numpy as np
from glob import glob
from utils.image_io import nib_load, nib_save, check_folder
from utils.utils import pair_labels

# ================================
# remap labels
# ================================
import json

method_name = 'CShaperPP'
embryo_name_this_g = '200113plc1p2'

source_folder_pred = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\ForTimelapseAnd2DEvaluation\{}_unified\{}".format(
    method_name, embryo_name_this_g)
# # change this when you are working on validation
pred_files_path = glob(
    os.path.join(source_folder_pred, "*segCell.nii.gz"))  # change this when you are working on validation

# change this when you are working on validation
source_folder_gt = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\Timelapse2DsliceEvaluationGT"
target_files_path = glob(os.path.join(source_folder_gt, embryo_name_this_g,"{}_*_128_G.nii.gz".format(embryo_name_this_g)))

pred_files_path.sort()
target_files_path.sort()

dst_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\ForTimelapseAnd2DEvaluation\2DTimelapse'
print(pred_files_path, target_files_path)
assert len(pred_files_path) == len(target_files_path), "#files not equal"

bar = tqdm(range(len(pred_files_path)))
bar.set_description("Uniform labels for 2D timelapse")
sw, sh, sz = nib_load(pred_files_path[0]).shape
sw_slice = int(sw / 2)
for pred_file_path, target_file_path in zip(pred_files_path, target_files_path):
    this_mapping_dict = {}
    pred_embryo = nib_load(pred_file_path).astype(np.uint16)[sw_slice]
    gt_embryo = nib_load(target_file_path).astype(np.uint16)[sw_slice]
    pred2target = pair_labels(pred_embryo, gt_embryo)
    # print('prediction seg to ground truth pair list ', pred2target)

    target_max = gt_embryo.max()
    pred_id_list = list(np.unique(pred_embryo))[1:]
    target_id_list = list(np.unique(gt_embryo))[1:]

    out = np.zeros_like(pred_embryo)
    left_labels = pred_id_list.copy()
    for pred_id, target_id in pred2target.items():
        overlap_mask = np.logical_and(pred_embryo == pred_id, gt_embryo == target_id)
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
    embryo_name_tp = '_'.join(os.path.basename(pred_file_path).split('.')[0].split('_')[:2])

    save_file = os.path.join(dst_path, method_name,embryo_name_tp + '_128_uni.nii.gz')

    check_folder(save_file)

    with open(os.path.join(dst_path,method_name, embryo_name_tp + '_128_replacing.txt'), 'w') as f:
        f.write(json.dumps(this_mapping_dict))
    nib_save(save_file, out.astype(np.uint16))
    # save_file = os.path.join(dst_path, embryo_name_tp + '_128_G.nii.gz')
    # nib_save(save_file, gt_embryo.astype(np.uint16))
    bar.update(1)

bar.close()
