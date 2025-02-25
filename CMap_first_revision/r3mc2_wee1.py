# E AND MS irregularity , along cells

import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu,wilcoxon

gui_data_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Temporary Sharing\wee24gui'

calculating_embryo_names_and_zero_timing_4cell = {'241106wee1plc1p2': 0,'241106wee1plc1p4': 0}

start_cells = ['E', 'MS']
#
# pd_volume_surface_irregularity = pd.DataFrame(
#     columns=['Embryo Name', 'Cell Name', 'Time', 'Cell Volume', 'Cell Surface', 'Irregularity', 'Precedent',
#              'Imaging TP'])
#
# for embryo_name, _ in calculating_embryo_names_and_zero_timing_4cell.items():
#     volume_file = pd.read_csv(os.path.join(gui_data_root, embryo_name, '{}_volume.csv'.format(embryo_name)))
#     surface_file = pd.read_csv(os.path.join(gui_data_root, embryo_name, '{}_surface.csv'.format(embryo_name)))
#
#     volume_tem = volume_file[start_cells[0]]
#     mask_tem = volume_tem.notna()
#     data_this_this = volume_tem.loc[mask_tem]
#     for tp_this, value_this in data_this_this.items():
#         if value_this > 0:
#             start_tp = tp_this
#             print(embryo_name,start_tp)
#             break
#
#     max_time_this = len(volume_file.index)
#
#     for cell_name_this in volume_file.columns:
#         for start_cell in start_cells:
#             if cell_name_this.startswith(start_cell) and cell_name_this != 'EMS':
#                 mask_tem = volume_file[cell_name_this].notna()
#
#                 volume_data_this_this = volume_file.loc[mask_tem][cell_name_this]
#                 surface_data_this_this = surface_file.loc[mask_tem][cell_name_this]
#
#                 for tp_this, cell_volume_this in volume_data_this_this.items():
#                     if cell_volume_this > 0:
#                         cell_surface_this = surface_data_this_this.loc[tp_this]
#
#                         irregularity_this = cell_surface_this ** (1 / 2) / cell_volume_this ** (1 / 3)
#
#                         pd_volume_surface_irregularity.loc[len(pd_volume_surface_irregularity.index)] = \
#                             [embryo_name, cell_name_this, (int(tp_this) - start_tp) * 1.43, cell_volume_this,
#                              cell_surface_this, irregularity_this, start_cell+' sublineage', int(tp_this)]

# pd_volume_surface_irregularity = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells_wee1.csv'))
#
# for precedent_cell in start_cells:
#     pd_this_precedent = pd_volume_surface_irregularity.loc[
#         pd_volume_surface_irregularity['Precedent'] == precedent_cell+' sublineage']
#     timings_this_emb = set(pd_this_precedent['Time'])
#     for timing_tem in timings_this_emb:
#         tem_pd=pd_this_precedent.loc[pd_this_precedent['Time'] == timing_tem]
#         irregularity_this_tp_this_pre = list(tem_pd['Irregularity'])
#         # tp_this_list=list(tem_pd['Imaging TP'])
#         all_embcell_name_this_tp_irregularity_avg = sum(irregularity_this_tp_this_pre) / len(
#             irregularity_this_tp_this_pre)
#         # tp_this_tem=sum(tp_this_list)/len(tp_this_list)
#         # print(tp_this_tem,tp_this_list)
#         pd_volume_surface_irregularity.loc[len(pd_volume_surface_irregularity.index)] = ['Average', None, timing_tem,
#                                                                                          None, None,
#                                                                                          all_embcell_name_this_tp_irregularity_avg,
#                                                                                          precedent_cell,None]
#
# pd_volume_surface_irregularity.to_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells_wee1_3.csv'))
#
# # =================================================================================
# def find_peaks_and_troughs(values_dict, window_size=3):
#     peaks = []
#     troughs = []
#
#     # Ensure the window size is at least 3 to compare properly
#     half_window = window_size // 2
#
#     # Extract the keys (indices) and values from the dictionary
#     keys = list(values_dict.keys())
#     values = list(values_dict.values())
#
#     # Iterate through the dictionary values, starting from (half_window) to (len(values) - half_window - 1)
#     for i in range(half_window, len(values) - half_window):
#         # Get the neighbors: 3 before, the element, and 3 after
#         neighbors = values[i - half_window:i + half_window + 1]
#
#         # Check if current value is a peak
#         if values[i] > max(neighbors[:half_window] + neighbors[half_window + 1:]):
#             peaks.append((keys[i], values[i]))  # (key, value)
#
#         # Check if current value is a trough
#         elif values[i] < min(neighbors[:half_window] + neighbors[half_window + 1:]):
#             troughs.append((keys[i], values[i]))  # (key, value)
#
#     return peaks, troughs
#
# pd_volume_surface_irregularity = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells_wee1_3.csv'))
#
# pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
#     pd_volume_surface_irregularity['Embryo Name'] == 'Average']
#
# for target_cell in start_cells:
#     pd_volume_surface_irregularity_this=pd_volume_surface_irregularity.loc[pd_volume_surface_irregularity['Precedent']==target_cell]
#     # Example usage with dictionary input:
#     values_dict = result_dict = dict(zip(list(pd_volume_surface_irregularity_this['Time']), list(pd_volume_surface_irregularity_this['Irregularity'])))
#
#     peaks, troughs = find_peaks_and_troughs(values_dict, window_size=10)
#
#     # Print peaks and troughs
#     print(target_cell,"  Peaks:")
#     for key, value in peaks:
#         print(f"Index: {key}, Value: {value}")
#
#     print(target_cell,"  Troughs:")
#     for key, value in troughs:
#         print(f"Index: {key}, Value: {value}")

# ====================================================================================
pd_volume_surface_irregularity = pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_some_precedent_cells_wee1.csv'))

pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
    pd_volume_surface_irregularity['Embryo Name'] != 'Average']

plt.figure(figsize=(16, 6))

ax=sns.lineplot(data=pd_volume_surface_irregularity, x='Time', y='Irregularity', hue='Precedent', hue_order=['MS sublineage', 'E sublineage'],
             palette={'MS sublineage':'blue', 'E sublineage':'red'},
             errorbar=None,
             # err_style="band"
             )

plt.arrow(
    x=7.15, y=2.643016+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=12.87, y=2.33156-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=25.74, y=2.62788+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=31.46, y=2.3710-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=40.04, y=2.411127-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=45.76, y=2.718+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=60.06, y=2.52688-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=75.79, y=2.7129+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=2.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)


# =======the std and range===================

plt.xticks(fontsize=26,family='Arial')
plt.yticks([2.3,2.4,2.5,2.6,2.7,2.8],fontsize=26,family='Arial')

plt.xlabel("Developmental time since appearance of MS and E (min)", size=26,family='Arial')
plt.ylabel(r'Cell irregularity', size=26,family='Arial')
ax.yaxis.set_label_coords(-0.06,0.62)
plt.legend(prop={'size': 16})

# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,prop={'size': 16})

plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'contact_calculating_r3mc2.pdf'), format="pdf", dpi=300)

plt.show()

