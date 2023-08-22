import os
import shutil
import numpy as np
import nibabel as nib


def check_folder(file_folder, overwrite=False):
    if "." in os.path.basename(file_folder):
        file_folder = os.path.dirname(file_folder)
    if os.path.isdir(file_folder) and overwrite:
        shutil.rmtree(file_folder)
    elif not os.path.isdir(file_folder):
        os.makedirs(file_folder)


def nib_save(file_name, data, overwrite=False):
    check_folder(file_name, overwrite)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, file_name)


def nib_load(file_name):
    assert os.path.isfile(file_name), "File {} not exist".format(file_name)

    return nib.load(file_name).get_fdata()

# check lost files
def find_lost_files(file_list, max_tps):
    # ===============================
    # Check the existance of files
    # ===============================
    name_tps = [os.path.basename(file).split(".")[0].split("_")[1] for file in file_list]
    tps = [str(i).zfill(3) for i in range(1, max_tps + 1, 1)]

    extra_files = []
    for i, name_tp in enumerate(name_tps):
        try:
            tps.remove(name_tp)
        except:
            extra_files.append(os.path.basename(file_list[i]))

    print("     Tps not in the folder :  {}\n".format(tps))
    print("     Extra files in the folder  :  {}\n".format(extra_files))

def highlight_vals(x, value, color='green'):
    if x == value:
        return 'background-color: %s' % color
    else:
        return ''

def have_checked(row):
    if row["Revised"] == "Y" or row["No Error"] == "Y" or row["Not Clear"] == "Y":
        return True
    return False


def get_boundary(seg):
    padded = np.pad(seg, 1, mode='edge')

    if padded.ndim == 2:
        border_pixels = np.logical_or(
            np.logical_or(seg != padded[:-2, 1:-1], seg != padded[2:, 1:-1]),
            np.logical_or(seg != padded[1:-1, :-2], seg != padded[1:-1, 2:])
        )

    elif padded.ndim == 3:
        border_pixels = np.logical_or(
            np.logical_or(np.logical_or(seg != padded[:-2, 1:-1, 1:-1], seg != padded[2:, 1:-1, 1:-1]),
                          np.logical_or(seg != padded[1:-1, :-2, 1:-1], seg != padded[1:-1, 2:, 1:-1])),
            np.logical_or(seg != padded[1:-1, 1:-1, :-2], seg != padded[1:-1, 1:-1, 2:])
        )

    return border_pixels.astype(np.uint8)


def get_all_surface_pixels(seg):
    bg = (seg == 0).astype(np.uint8)
    bg_pad = np.pad(bg, pad_width=1, mode="edge")

    if bg_pad.ndim == 2:
        border_pixels = np.logical_or(
            np.logical_or(bg != bg_pad[:-2, 1:-1], bg != bg_pad[2:, 1:-1]),
            np.logical_or(bg != bg_pad[1:-1, :-2], bg != bg_pad[1:-1, 2:])
        )

    elif bg_pad.ndim == 3:
        border_pixels = np.logical_or(
            np.logical_or(np.logical_or(bg != bg_pad[:-2, 1:-1, 1:-1], bg != bg_pad[2:, 1:-1, 1:-1]),
                          np.logical_or(bg != bg_pad[1:-1, :-2, 1:-1], bg != bg_pad[1:-1, 2:, 1:-1])),
            np.logical_or(bg != bg_pad[1:-1, 1:-1, :-2], bg != bg_pad[1:-1, 1:-1, 2:])
        )

    # delete_bg_border
    border_pixels[bg != 0] = 0
    label_border_mask = seg.copy()
    label_border_mask[border_pixels == 0] = 0

    return label_border_mask


def get_contact_pairs(seg, i_tp, label2name_dict):
    bg = (seg == 0).astype(np.uint8)
    seg = np.pad(seg, pad_width=1, mode="edge")

    pairs_all = []
    # up and down
    forward_seg = seg[:-2, 1:-1, 1:-1]
    backward_seg = seg[2:, 1:-1, 1:-1]
    paired_seg_mask = np.logical_and(np.logical_and(forward_seg != backward_seg, backward_seg != 0), forward_seg != 0)
    up_labels = forward_seg[paired_seg_mask]
    down_labels = backward_seg[paired_seg_mask]

    pairs = np.stack((np.minimum(up_labels, down_labels), np.maximum(up_labels, down_labels)), axis=-1).tolist()
    pairs_all += pairs

    # left right
    forward_seg = seg[1:-1, :-2, 1:-1]
    backward_seg = seg[1:-1, 2:, 1:-1]
    paired_seg_mask = np.logical_and(np.logical_and(forward_seg != backward_seg, backward_seg != 0), forward_seg != 0)
    up_labels = forward_seg[paired_seg_mask]
    down_labels = backward_seg[paired_seg_mask]

    pairs = np.stack((np.minimum(up_labels, down_labels), np.maximum(up_labels, down_labels)), axis=-1).tolist()
    pairs_all += pairs

    # forward backward
    forward_seg = seg[1:-1, 1:-1, :-2]
    backward_seg = seg[1:-1, 1:-1, 2:]
    paired_seg_mask = np.logical_and(np.logical_and(forward_seg != backward_seg, backward_seg != 0), forward_seg != 0)
    up_labels = forward_seg[paired_seg_mask]
    down_labels = backward_seg[paired_seg_mask]

    pairs = np.stack((np.minimum(up_labels, down_labels), np.maximum(up_labels, down_labels)), axis=-1).tolist()
    pairs_all += pairs

    unique_pairs = {}
    used_pairs = []
    for pair in pairs_all:
        if pair in used_pairs:
            continue
        used_pairs.append(pair)
        num_element = pairs_all.count(pair)
        if num_element < 5:
            continue
        pair_str = [label2name_dict[i_label] for i_label in pair]
        unique_pairs.update({tuple(pair_str + [i_tp]): num_element})

    return unique_pairs


