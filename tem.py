# import os
# import nibabel as nib
# import pandas as pd
# from tqdm import tqdm
# from skimage.transform import resize
# from glob import glob
# from utils.image_io import nib_load
# import numpy as np


# ===================
# retrival number of cells
# ===================
# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\DatasetUpdated\200113plc1p2\SegCellTimeCombinedLabelUnified"
# save_file = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\paper\7_AtlasCell\Code\Evaluation\Results\2DErrorMap\200113plc1p2.csv"
# src_files = glob(os.path.join(src_folder, "*.nii.gz"))
#
# nums = []
# names = []
# for src_file in tqdm(src_files):
#     seg = nib_load(src_file)
#     nums.append(np.unique(seg).shape[0] - 1)
#     names.append(os.path.basename(src_file).split(".")[0])
#
# pd_num = pd.DataFrame(data={"name":names, "num":nums})
# pd_num.to_csv(save_file, index=False)

# ================================
# save for rendering
# ================================
# src_folder = r"D:\ProjectData\CMapEvaluation\3D"
# src_files = glob(os.path.join(src_folder, "*uni.nii.gz"))
# for src_file in tqdm(src_files):
#     img = nib.load(src_file)
#     header = img.header
#     seg = img.get_fdata().astype(np.uint16)
#     target_shape = [int(x / 2) for x in seg.shape]
#     header["dim"] = [3] + target_shape + [1] * 4
#     seg = resize(image=seg, output_shape=target_shape, preserve_range=True, order=0).astype(np.uint8)
#     header["pixdim"]=[1.0, 0.36, 0.36, 0.36, 0., 0., 0., 0.]
#     header["xyzt_units"]=11
#     img = nib.Nifti1Image(seg, affine=np.eye(4), header=header)
#     save_file = src_file.replace(".nii.gz", "_render.nii.gz")
#     nib.save(img, save_file)

# =================================
# Save nii as tif
# =================================
# from PIL import Image
# from tifffile import imwrite
# from utils.utils import check_folder
# def save_tif(file_name, data):
#     """Save matrix data as indexed images which can be rendered by ImageJ"""
#     data = np.transpose(data, [2, 0, 1])
#     imwrite(file_name, data)

# src_folder = r"D:\OneDriveBackup\OneDrive - City University of Hong Kong\Dataset\FluorescentImaging\DenoiseStack\TestData"
# save_folder = r"C:\Users\bcc\Desktop\DeepDenoise\TifFiles"
# src_files = glob(os.path.join(src_folder, "*.nii.gz"))
# for src_file in tqdm(src_files, desc="Saving tifs to {}".format(save_folder)):
#     img = nib.load(src_file)
#     data = img.get_fdata()
#     data0 = (data).astype(np.uint8)
#     save_file = os.path.join(save_folder, os.path.basename(src_file).split(".")[0] + ".tif")
#     save_tif(save_file, data0)

# img_files = glob(os.path.join(r"D:\TemDownload\psf-gl\frames", "*.tif"))
# imgs = []
# for img_file in tqdm(img_files):
#     img = np.array(Image.open(img_file))
#     imgs.append(img)
#     print(img.max())
#
# imgs = np.stack(imgs, axis=-1)
# # imgs = (imgs * 255).astype(np.uint8)
# save_tif(r"C:\Users\bcc\Desktop\DeepDenoise\PSFExample.tif", imgs)

# -------------------------------------------------
# from utils.utils import read_tif
# file_path = r"D:\TemDownload\PSF-CElegans-CY3.tif"
# arr = read_tif(file_path)
# arr = (arr / 65535.0 * 255.0).astype(np.uint8)
# arr0 = (arr[340:340+32, 320:320 + 32, 36:36+32])
# arr0 = arr
# save_file = r"C:\Users\bcc\Desktop\DeepDenoise\ParallelDecon\PSF1.tif"
# save_tif(save_file, arr0)

