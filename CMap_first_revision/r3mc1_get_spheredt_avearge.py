import os.path

import pandas as pd
import numpy as np

volume_surface_file_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\spheresDT\geom_info_full_enriched.csv'

contact_file_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\spheresDT\R_7cell.xlsx'

saving_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\spheresDT'

# volume_suface_pd=pd.read_csv(volume_surface_file_path)
#
# cell_name_List=set(volume_suface_pd['cell_name'])
#
# average_pd=pd.DataFrame(columns=['cell name','average volume','average surface area'])
#
# for cell_name_this in cell_name_List:
#     this_cell_pd=volume_suface_pd.loc[volume_suface_pd['cell_name']==cell_name_this]
#     volume_list=list(this_cell_pd['volume'])
#     avearage_volume=sum(volume_list)/len(volume_list)
#
#     surface_list = list(this_cell_pd['volume'])
#     avearage_surface = sum(surface_list) / len(surface_list)
#
#     average_pd.loc[len(average_pd)]=[cell_name_this, avearage_volume,avearage_surface]
#
# average_pd.to_csv(os.path.join(saving_path,'average_volume_surface.csv'))

# ====================================================================
contact_area_pd=pd.read_excel(contact_file_path,index_col=[1,2])
print(set(contact_area_pd.index))
# cell_name_List=set(contact_area_pd['cell_name'])

average_pd=pd.DataFrame(columns=['cell-cell pair','average area'])

for cell_name_pair in set(contact_area_pd.index):
    # print(cell_name_pair)
    this_cell_pd_contact=contact_area_pd.loc[cell_name_pair]['contactarea_mean']
    list_average=[]
    for value in this_cell_pd_contact:
        if value>0:
            list_average.append(value)
    if len(list_average)>0:
        average_this=sum(list_average)/len(list_average)
        average_pd.loc[len(average_pd)]=[cell_name_pair,average_this]

average_pd.to_csv(os.path.join(saving_path,'average_contact.csv'))


