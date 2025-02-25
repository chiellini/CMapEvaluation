import os.path

import pandas as pd
import numpy as np

volume_surface_contact_file_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\pack_azuma_seg\statistics'

saving_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\pack_azuma_seg'

# List all folders
embryo_names_list = [folder for folder in os.listdir(volume_surface_contact_file_path) if os.path.isdir(os.path.join(volume_surface_contact_file_path, folder))]

volume_dict = {}
surface_dict = {}

for embryo_name_this in embryo_names_list:
    pd_this_volume = pd.read_csv(
        os.path.join(volume_surface_contact_file_path, embryo_name_this, '{}_volume.csv'.format(embryo_name_this)))
    pd_this_surface = pd.read_csv(
        os.path.join(volume_surface_contact_file_path, embryo_name_this, '{}_surface.csv'.format(embryo_name_this)))

    for cell_name in pd_this_volume.columns:
        volume_list = list(pd_this_volume[cell_name][pd_this_volume[cell_name].notna()])
        average_list = []
        for value in volume_list:
            if value > 0:
                average_list.append(value)
        average_volume = sum(average_list) / len(average_list)
        if cell_name in volume_dict.keys():
            volume_dict[cell_name].append(average_volume)
        else:
            volume_dict[cell_name] = [average_volume]

        surface_list = list(pd_this_surface[cell_name][pd_this_surface[cell_name].notna()])
        average_list = []
        for value in surface_list:
            if value > 0:
                average_list.append(value)
        average_surface = sum(average_list) / len(average_list)
        if cell_name in surface_dict.keys():
            surface_dict[cell_name].append(average_surface)
        else:
            surface_dict[cell_name] = [average_surface]

average_pd_volume_surface = pd.DataFrame(columns=['cell name', 'average volume', 'average surface area'])

for cell_name, volume_list in volume_dict.items():
    last_average_volume = sum(volume_list) / len(volume_list)
    last_average_surface = sum(surface_dict[cell_name]) / len(surface_dict[cell_name])
    average_pd_volume_surface.loc[len(average_pd_volume_surface)] = [cell_name, last_average_volume,
                                                                     last_average_surface]

average_pd_volume_surface.to_csv(os.path.join(saving_path, 'average_volume_surface.csv'))
