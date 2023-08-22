import os
import math
import numpy as np
from PIL import Image
from tifffile import imwrite
from scipy.optimize import linear_sum_assignment

from utils.image_io import check_folder

# generate 768 tiff figure drawing put palette
P = [252, 233, 79, 114, 159, 207, 239, 41, 41, 173, 127, 168, 138, 226, 52,
     233, 185, 110, 252, 175, 62, 211, 215, 207, 196, 160, 0, 32, 74, 135, 164, 0, 0,
     92, 53, 102, 78, 154, 6, 143, 89, 2, 206, 92, 0, 136, 138, 133, 237, 212, 0, 52,
     101, 164, 204, 0, 0, 117, 80, 123, 115, 210, 22, 193, 125, 17, 245, 121, 0, 186,
     189, 182, 85, 87, 83, 46, 52, 54, 238, 238, 236, 0, 0, 10, 252, 233, 89, 114, 159,
     217, 239, 41, 51, 173, 127, 178, 138, 226, 62, 233, 185, 120, 252, 175, 72, 211, 215,
     217, 196, 160, 10, 32, 74, 145, 164, 0, 10, 92, 53, 112, 78, 154, 16, 143, 89, 12,
     206, 92, 10, 136, 138, 143, 237, 212, 10, 52, 101, 174, 204, 0, 10, 117, 80, 133, 115,
     210, 32, 193, 125, 27, 245, 121, 10, 186, 189, 192, 85, 87, 93, 46, 52, 64, 238, 238, 246]

P = P * math.floor(255*3/len(P))
l = int(255 - len(P)/3)
P = P + P[3:(l+1)*3]
P = [0,0,0] + P

print(len(P))

def pair_labels(pred, target):
    """Pairwise the labels between pred and target"""
    target = np.copy(target)  # ? do we need this
    pred = np.copy(pred)
    target_id_list = list(np.unique(target))
    pred_id_list = list(np.unique(pred))
    # print(np.unique(target,return_counts=True))
    # print(np.unique(pred,return_counts=True))

    target_masks = {}
    for t in target_id_list:
        if t==0:
            continue
        t_mask = np.array(target == t, np.uint8)
        target_masks[t]=t_mask

    pred_masks = {}
    pred_dict_id_to_list_order={}
    for tmp_idx_p,p in enumerate(pred_id_list):
        if p==0:
            continue
        p_mask = np.array(pred == p, np.uint8)
        pred_masks[p]=p_mask
        pred_dict_id_to_list_order[p]=tmp_idx_p


    # prefill with value
    pairwise_inter = np.zeros([len(target_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(target_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)


    # caching pairwise
    for t_idx,target_id in enumerate(target_id_list[1:]):  # 0-th is background
        # print(t_idx,target_id)
        t_mask = target_masks[target_id]
        pred_target_overlap = pred[t_mask > 0]
        pred_target_overlap_id = np.unique(pred_target_overlap)
        pred_target_overlap_id = list(pred_target_overlap_id)

        for pred_id in pred_target_overlap_id:
            if pred_id == 0:  # ignore
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            # overlaping background
            inter = (t_mask * p_mask).sum()
            p_idx=pred_dict_id_to_list_order[pred_id]-1 # t_idx has been -1 for target_id_list[1:]
            pairwise_inter[t_idx, p_idx] = inter
            pairwise_union[t_idx, p_idx] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # Munkres pairing to find maximal unique pairing
    paired_target, paired_pred = linear_sum_assignment(-pairwise_iou)
    # print(pairwise_iou)
    # print(paired_target)
    # print(paired_pred)
    pred_labels = pred_id_list[1:]
    target_labels = target_id_list[1:]
    pred2target_dict = {pred_labels[pred_idx]:target_labels[target_label_idx] for pred_idx, target_label_idx in zip(paired_pred, paired_target)}

    return pred2target_dict


def relabel_instance(inst):
    out_inst = inst.copy()
    labels = sorted(np.unique(inst).tolist())
    # labels.remove(0)
    for idx, label in enumerate(labels):
        out_inst[out_inst == label] = idx

    return out_inst

def save_indexed_tif(file_name, data):
    """Save matrix data as indexed images which can be rendered by ImageJ"""
    check_folder(file_name)
    # print(len(P))
    tif_imgs = []
    num_slices = data.shape[-1]
    for i_slice in range(num_slices):
        tif_img = Image.fromarray(data[..., i_slice], mode="P")
        print(type(tif_img))
        tif_img.putpalette(P)
        tif_imgs.append(tif_img)
    if os.path.isfile(file_name):
        os.remove(file_name)

    # save the 1th slice image, treat others slices as appending
    tif_imgs[0].save(file_name, save_all=True, append_images=tif_imgs[1:])

def scale2index(seg0):
    """Rescale all labels into range [0, 255]"""
    seg = seg0 % 255
    reduce_mask = np.logical_and(seg0!=0, seg==0)
    seg[reduce_mask] = 255  # Because only 255 colors are available, all cells should be numbered within [0, 255].
    seg = seg.astype(np.uint8)

    return seg

def read_tif(file_path):
    dataset = Image.open(file_path)
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w,dataset.n_frames))
    for i in range(dataset.n_frames):
       dataset.seek(i)
       tiffarray[:,:,i] = np.array(dataset)
    expim = tiffarray.astype(np.double)

    return expim

def save_tif(file_name, data):
    """Save matrix data as indexed images which can be rendered by ImageJ"""
    data = np.transpose(data.astype(np.float32), [2, 0, 1])
    imwrite(file_name, data)
