# E AND MS irregularity , along cells

import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu,wilcoxon

# gui_data_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\CMapSubmission\Dataset Access\Dataset E'
#
# calculating_embryo_names_and_zero_timing_4cell = {'WT_Sample1': 11, 'WT_Sample2': 3, 'WT_Sample3': 2, 'WT_Sample4': 13,
#                                                   'WT_Sample5': 7, 'WT_Sample6': 12, 'WT_Sample7': 1, 'WT_Sample8': 3}
#
# start_cells = ['E', 'MS']
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
#                                                                                          precedent_cell+' sublineage',None]
#
# pd_volume_surface_irregularity.to_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells.csv'))

# =================================================================================
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
#     'irregularity_of_some_precedent_cells.csv'))
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
    'irregularity_of_some_precedent_cells.csv'))

pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
    pd_volume_surface_irregularity['Embryo Name'] != 'Average']

plt.figure(figsize=(16, 6))

ax=sns.lineplot(data=pd_volume_surface_irregularity, x='Time', y='Irregularity', hue='Precedent', hue_order=['MS sublineage', 'E sublineage'],
             palette={'MS sublineage':'blue', 'E sublineage':'red'},
             errorbar=None,
             # err_style="band"
             )

height_x=1
star_symbol_size=128

left_main_x=-1.6
x = [left_main_x + height_x, left_main_x, left_main_x, left_main_x+height_x]
y = [2.51, 2.51, 2.56, 2.56]
plt.plot(x, y, linewidth=4, color='black')
# plt.scatter(left_main_x-6,2.53 ,marker='*', s=star_symbol_size, color='black')
# plt.text(left_main_x-5,2.52, '*', fontsize=28, ha='center', va='center')
plt.text(left_main_x-8,2.515, '**', fontsize=28, ha='center', va='center')

left_main_x=6
x = [left_main_x + height_x, left_main_x, left_main_x, left_main_x+height_x]
y = [2.31, 2.31, 2.33, 2.33]
plt.plot(x, y, linewidth=4, color='black')
# plt.scatter(left_main_x-6,2.53 ,marker='*', s=star_symbol_size, color='black')
# plt.text(left_main_x-5,2.31, '**', fontsize=28, ha='center', va='center')
plt.text(left_main_x-8,2.295, '**', fontsize=28, ha='center', va='center')

x_here=0.72
y_here=0.26
# plt.gcf().text(x_here, y_here, "Student's $t$-test\n (one sided)", fontsize=18)
plt.gcf().text(x_here, y_here, "Wilcoxon Signed-Rank\nTest(one-sided)", fontsize=18)

plt.gcf().text(x_here, y_here-0.1, '**', fontsize=28)
plt.gcf().text(x_here+0.04, y_here-0.08, r'$p \leq 0.05 $', fontsize=16)

# ======================MS Arrow=======================
plt.arrow(
    x=7.15, y=2.5667+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=12.87, y=2.307914-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=22.88, y=2.61925+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=32.89, y=2.34783-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=47.19, y=2.6961+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=58.63, y=2.50317-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=75.7899, y=2.6285+0.05,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=87.229999, y=2.5277-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=107.25, y=2.631751+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=125.84, y=2.5546-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

plt.arrow(
    x=151.57999, y=2.62898+0.04,          # Starting point (x, y)
    dx=0, dy=-0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)


# ======E gastrulation finished====
plt.arrow(
    x=48.62, y=2.4082265-0.04,          # Starting point (x, y)
    dx=0, dy=0.01,        # Change in x (dx) and y (dy) - defines length and direction
    head_width=3.6,    # Width of the arrowhead
    head_length=0.02,   # Length of the arrowhead
    fc='black',         # Fill color for arrow (e.g., face color)
    ec='black'          # Edge color for arrow
)

# ===================lower description====================
y_basic=2.14
y_bias=0.01
x = [0, 0, 12.8, 12.8]
y = [y_basic+y_bias, y_basic, y_basic, y_basic+y_bias]
plt.plot(x, y, linewidth=2, color='black')
plt.text(9,y_basic-0.082, 'Before E2\ninternalization', fontsize=16, ha='center', va='center')

y_basic=2.24
y_bias=0.01
x = [12.9, 12.9, 48, 48]
y = [y_basic+y_bias, y_basic, y_basic, y_basic+y_bias]
plt.plot(x, y, linewidth=2, color='black')
plt.text(32,y_basic-0.043, 'During E2 internalization', fontsize=16, ha='center', va='center')

y_basic=2.3
y_bias=0.01
x = [49, 49, 336, 336]
y = [y_basic+y_bias, y_basic, y_basic, y_basic+y_bias]
plt.plot(x, y, linewidth=2, color='black')
plt.text(200,y_basic-0.043, 'After E2 internalization', fontsize=16, ha='center', va='center')

# =======E and MS the std and range===================
y_basic=1.96
y_bias=0.01
x = [12.9, 12.9, 48, 48]
y = [y_basic+y_bias, y_basic, y_basic, y_basic+y_bias]
plt.plot(x, y, linewidth=2, color='red')
plt.text(32,y_basic-0.126, 'E sublineage:\nSTD = 0.06\nMAX-MIN = 0.34', fontsize=16, ha='center', va='center')
plt.plot(0, y_basic-0.196,c='white')


y_basic=2.76
y_bias=0.01
x = [12.9, 12.9, 47, 47]
y = [y_basic-y_bias, y_basic, y_basic, y_basic-y_bias]
plt.plot(x, y, linewidth=2, color='blue')
plt.text(32,y_basic+0.106, 'MS sublineage:\nSTD = 0.13\nMAX-MIN = 0.55', fontsize=16, ha='center', va='center')
plt.plot(0, 2.93,c='white')


# =======the std and range===================

plt.xticks(fontsize=26,family='Arial')
plt.yticks([2.2,2.3,2.4,2.5,2.6,2.7,2.8],fontsize=26,family='Arial')

plt.xlabel("Developmental time starting from birth of MS and E (min)", size=26,family='Arial')
plt.ylabel(r'Cell irregularity', size=26,family='Arial')
ax.yaxis.set_label_coords(-0.06,0.62)
plt.legend(prop={'size': 16})

# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,prop={'size': 16})

plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'contact_calculating_r3mc2.pdf'), format="pdf", dpi=300)

plt.show()

# ===================================================================================
# pd_volume_surface_irregularity = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells.csv'))
# pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
#     pd_volume_surface_irregularity['Embryo Name'] != 'Average']
#
#
# def t_test_comparison(list1, list2):
#     # Calculate the means and standard deviations
#     mean1, std1 = np.mean(list1), np.std(list1, ddof=1)  # ddof=1 for sample standard deviation
#     mean2, std2 = np.mean(list2), np.std(list2, ddof=1)
#
#     # Perform an independent two-sample t-test
#     t_stat, p_value = stats.ttest_ind(list1, list2, equal_var=False)  # Welch's t-test (unequal variances)
#
#     # Print the results
#     print(f"Mean of List 1: {mean1:.2f}, Standard Deviation: {std1:.2f}")
#     print(f"Mean of List 2: {mean2:.2f}, Standard Deviation: {std2:.2f}")
#     print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
#
#     # Determine significance
#     alpha = 0.05  # significance level (5%)
#     if p_value < alpha:
#         print("There is a significant difference between the two datasets (reject null hypothesis).")
#     else:
#         print("There is no significant difference between the two datasets (fail to reject null hypothesis).")
#
# target_time=[7.15,8.58]
#
# # target_time=[12.87]
#
#
# MS_E_irregularity_pd = pd_volume_surface_irregularity.loc[pd_volume_surface_irregularity['Time'].isin(target_time)]
#
# E_irregularity = MS_E_irregularity_pd.loc[
#     MS_E_irregularity_pd['Precedent'] == 'E sublineage']['Irregularity']
# MS_irregularity = MS_E_irregularity_pd.loc[
#     MS_E_irregularity_pd['Precedent'] == 'MS sublineage']['Irregularity']
#
# t_test_comparison(list(E_irregularity), list(MS_irregularity))
#
# statistic, p_value = wilcoxon(list(E_irregularity), list(MS_irregularity))
# print('Wilcosum-Ranksum Test, statistic, p_value:  ', statistic,p_value)
#
# # ===================================================================================
# pd_volume_surface_irregularity = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells.csv'))
# pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
#     pd_volume_surface_irregularity['Embryo Name'] != 'Average']
#
# target_time_range=[12.87,48.62]
#
# MS_E_irregularity_pd = pd_volume_surface_irregularity.loc[(pd_volume_surface_irregularity['Time']>=target_time_range[0])
#                                                           & (pd_volume_surface_irregularity['Time']<=target_time_range[1])]
#
# E_irregularity_list = list(MS_E_irregularity_pd.loc[
#     MS_E_irregularity_pd['Precedent'] == 'E sublineage']['Irregularity'])
# MS_irregularity_list = list(MS_E_irregularity_pd.loc[
#     MS_E_irregularity_pd['Precedent'] == 'MS sublineage']['Irregularity'])
#
# meanE, stdE = np.mean(E_irregularity_list), np.std(E_irregularity_list, ddof=1)
# meanMS, stdMS = np.mean(MS_irregularity_list), np.std(MS_irregularity_list, ddof=1)
#
# print('E:  ',max(E_irregularity_list),min(E_irregularity_list), stdE,
#       'MS:  ',max(MS_irregularity_list), min(MS_irregularity_list), stdMS)
#
