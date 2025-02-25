import os.path

import pandas as pd
import numpy as np

volume_surface_contact_file_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\CMapSubmission\Dataset Access\Dataset E'

embryo_names_list=['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4',
                   'WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8',]

saving_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\cmap'

# volume_dict={}
# surface_dict={}
#
# for embryo_name_this in embryo_names_list:
#     pd_this_volume=pd.read_csv(os.path.join(volume_surface_contact_file_path,embryo_name_this,'{}_volume.csv'.format(embryo_name_this)))
#     pd_this_surface=pd.read_csv(os.path.join(volume_surface_contact_file_path,embryo_name_this,'{}_surface.csv'.format(embryo_name_this)))
#
#     for cell_name in pd_this_volume.columns:
#         volume_list=list(pd_this_volume[cell_name][pd_this_volume[cell_name].notna()])
#         average_list=[]
#         for value in volume_list:
#             if value>0:
#                 average_list.append(value)
#         average_volume=sum(average_list)/len(average_list)
#         if cell_name in volume_dict.keys():
#             volume_dict[cell_name].append(average_volume)
#         else:
#             volume_dict[cell_name]=[average_volume]
#
#         surface_list = list(pd_this_surface[cell_name][pd_this_surface[cell_name].notna()])
#         average_list = []
#         for value in surface_list:
#             if value > 0:
#                 average_list.append(value)
#         average_surface = sum(average_list) / len(average_list)
#         if cell_name in surface_dict.keys():
#             surface_dict[cell_name].append(average_surface)
#         else:
#             surface_dict[cell_name]=[average_surface]
#
# average_pd_volume_surface=pd.DataFrame(columns=['cell name','average volume','average surface area'])
#
# for cell_name, volume_list in volume_dict.items():
#     last_average_volume=sum(volume_list)/len(volume_list)
#     last_average_surface=sum(surface_dict[cell_name])/len(surface_dict[cell_name])
#     average_pd_volume_surface.loc[len(average_pd_volume_surface)]=[cell_name,last_average_volume,last_average_surface]
#
# average_pd_volume_surface.to_csv(os.path.join(saving_path,'average_volume_surface.csv'))

# ====================================================================
contact_dict={}
for embryo_name_this in embryo_names_list:
    pd_this_contact=pd.read_csv(os.path.join(volume_surface_contact_file_path,embryo_name_this,'{}_Stat.csv'.format(embryo_name_this)),index_col=[0,1])

    for cell_cell_pair in pd_this_contact.index:
        this_pair_series=pd_this_contact.loc[cell_cell_pair]
        # contact_list=this_pair_series[cell_cell_pair.notna()]
        average_list=[]
        for value in this_pair_series:
            if value>0:
                average_list.append(value)
        if len(average_list)>0:
            average_contact=sum(average_list)/len(average_list)
            if cell_cell_pair in contact_dict.keys():
                contact_dict[cell_cell_pair].append(average_contact)
            else:
                contact_dict[cell_cell_pair]=[average_contact]

average_pd_contact=pd.DataFrame(columns=['cell-cell pair','average contact'])

for cell_cell_contact, contact_list in contact_dict.items():
    if len(contact_list)>0:
        last_average_volume=sum(contact_list)/len(contact_list)
        average_pd_contact.loc[len(average_pd_contact)]=[cell_cell_contact,last_average_volume]

average_pd_contact.to_csv(os.path.join(saving_path,'average_contact.csv'))

