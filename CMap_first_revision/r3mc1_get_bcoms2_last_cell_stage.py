import os.path
import glob
import pandas as pd
import numpy as np

from utils.image_io import nib_load

saving_seg_cell = r'C:\Users\zelinli6\Downloads\pack_azuma_seg'

pd_cell_name = pd.DataFrame(columns=['Embryo Name', 'First Cell Number','Last Cell Number'])

embryo_name_list = [name for name in os.listdir(saving_seg_cell) if os.path.isdir(os.path.join(saving_seg_cell, name))]

for embryo_name_this in embryo_name_list:
    the_first_emb_path = sorted(glob.glob(os.path.join(saving_seg_cell, embryo_name_this, 'SegCell', '*.nii.gz')))[0]

    cell_number_first_this = len(np.unique(nib_load(the_first_emb_path))) - 1

    the_last_emb_path = sorted(glob.glob(os.path.join(saving_seg_cell, embryo_name_this, 'SegCell','*.nii.gz')))[-1]

    cell_number_last_this = len(np.unique(nib_load(the_last_emb_path))) - 1

    pd_cell_name.loc[len(pd_cell_name)] = [embryo_name_this, cell_number_first_this,cell_number_last_this]

pd_cell_name.to_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    'bcoms2_cellstage.csv'))
