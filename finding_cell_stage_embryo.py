import os.path
import glob
from tqdm import tqdm

import nibabel as nib
import numpy as np


embryo_names=['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4']

embryo_preseg_path=r'E:\ProjectData\MembraneProject\AllRawDataPacked'


for embryo_name_this in embryo_names:

    nuc_files = sorted(glob.glob(os.path.join(embryo_preseg_path,'{}\SegNuc\*_segNuc.nii.gz'.format(embryo_name_this))))

    cell_list = {}
    for idx, seg_cell_file in tqdm(enumerate(nuc_files), desc="Get cell number list"):
        seg_cell = nib.load(seg_cell_file).get_fdata()
        num_cell = len(np.unique(seg_cell)) - 1
        cell_list[idx+1]=num_cell
    print(embryo_name_this,cell_list)




