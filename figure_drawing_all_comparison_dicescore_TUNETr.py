# ===============================================================================================
# plot bar , direct comparison between different segmentation methods
# ===============================================================================================
import os.path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Replace these with the names of your CSV files
evaluation_path = r'Results/Comparison3DDiceIoU'
evaluation_file_paths = glob.glob(os.path.join(evaluation_path, '*_score.csv'))

# Replace these with the names of the columns you want to use for the x and y axes
x_data1 = ['<50','<50', '50-100', '100-300']
x_data2=['300-500', '>500']
embryo_name1='170704plc1p1'
embryo_name2='181210plc1p3'
embryo_name3='190314plc1p3'
embryo_name4 = '200109plc1p1'
embryo_name5 = '200113plc1p2'

embryo_times1 = [24, 44,64,84]
embryo_times2 = [24, 44,64,84]
embryo_times3 = [24, 44,64,84]
embryo_times4 = [123, 181]
embryo_times5 = [132,  185]
embryo_namess = {}
for tpidx, tp in enumerate(embryo_times1):
    embryo_namess[embryo_name1 + '_' + str(tp).zfill(3) + '_rawMemb_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times2):
    embryo_namess[embryo_name2 + '_' + str(tp).zfill(3) + '_rawMemb_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times3):
    embryo_namess[embryo_name3 + '_' + str(tp).zfill(3) + '_rawMemb_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times4):
    embryo_namess[embryo_name4 + '_' + str(tp).zfill(3) + '_rawMemb_segCell_uni'] = x_data2[tpidx]
for tpidx, tp in enumerate(embryo_times5):
    embryo_namess[embryo_name5 + '_' + str(tp).zfill(3) + '_rawMemb_segCell_uni'] = x_data2[tpidx]

for tpidx, tp in enumerate(embryo_times1):
    embryo_namess[embryo_name1 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times2):
    embryo_namess[embryo_name2 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times3):
    embryo_namess[embryo_name3 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data1[tpidx]
for tpidx, tp in enumerate(embryo_times4):
    embryo_namess[embryo_name4 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data2[tpidx]
for tpidx, tp in enumerate(embryo_times5):
    embryo_namess[embryo_name5 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data2[tpidx]
print(embryo_namess)

y_column_name = 'DiceScore'
# y_column_name = 'IoU'

dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])

# Loop through the CSV files and plot the data
hue_order_list = [
    '3DUNet++',
    'VNet',
    'SwinUNETR',
    # 'Cellpose3D',
    'StarDist3D',
    'CShaper',
    'CShaper++',
    'CTransformer']
hue_palette = {'3DUNet++': '#ff7c00',
               'VNet': '#1ac938',
               'SwinUNETR': '#f14cc1',
               'StarDist3D': '#ffc400',
               'CShaper': '#00d7ff',
               'CShaper++': '#74fff8',
               'CTransformer': '#e8000b'}

for file_path in evaluation_file_paths:
    print(file_path)
    # plt.close()
    # fig = plt.figure(figsize=(16, 9), dpi=80)
    sns.set_theme()
    # Create a figure to hold the plots
    score_this_data = pd.read_csv(file_path)

    # score_this_data = score_this_data[score_this_data[y_column_name] > 0.1]
    # score_this_data = score_this_data[score_this_data[y_column_name] < 0.9]
    print(score_this_data)

    score_this_data['RealEmbryoName'] = score_this_data['EmbryoName']

    for key, value in embryo_namess.items():
        score_this_data['EmbryoName'] = score_this_data['EmbryoName'].replace(key, value)
    method_name = os.path.basename(file_path).split('_')[0]
    score_this_data['Method'] = method_name
    # print(score_this_data)
    if method_name in hue_order_list:
        dataframe_list.append(score_this_data)
result_dataframe = pd.concat(dataframe_list)
print(set(result_dataframe['EmbryoName']))
print(result_dataframe,result_dataframe.columns)


# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 99),
                  hue_order=hue_order_list, palette=hue_palette)

# # ======================drawing the fucking p -value confidence====================================

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot

star_symbol_size = 18
star_interval_coe_hei = 0.035
star_interval_coe_wid = 0.04
font_dict_tmp = {'size': 10}

# =====================<50=====================================================

height = 0.05
width = 0.115

this_start_x=0.225
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width] # CShaper++ 0.018934763880916493
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2 - 0.12, y[0] + 0.02, 'n.s.', fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')  # CShaper 0.61310019983598
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')  # stardist 3d  0.06263100827713756
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')  # swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + + 0.02, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')  # VNet
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')  # swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + + 0.02, 'n.s.',fontdict=font_dict_tmp)

# =-=============50-100===========================================================
this_start_x=1.23
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + + 0.02, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # stardist3d
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + + 0.02, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #vnet
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3dunet++
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

# =-================100-300========================================================
this_start_x=2.23
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + + 0.02, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # cshaper
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # stardist3d
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # swinunetr
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #vnet
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #3dunet++
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

# =================300-500=================================================================
this_start_x=3.23
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black') #CSHAPER++
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')#CSHAPER
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')#STARDIST3D
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

# =-================500========================================================
this_start_x=4.23
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2 - star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2 + star_interval_coe_wid * 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size,
            color='black')


plt.xticks(fontsize=16)

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)

plt.xlabel("Cell number", size=20)
plt.ylabel('Dice score', size=20)

plt.gcf().text(0.85, 0.85, r"Student's $t$-test (one sided)", fontsize=12)

# plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)

plt.gcf().text(0.85, 0.805, ' n.s.', fontsize=10)
plt.gcf().text(0.88, 0.805, r' $ p > 0.10 $ (non-significant)', fontsize=10)

plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
plt.gcf().text(0.88, 0.765, r' $ p \leq 0.10 $', fontsize=10)

plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
plt.gcf().text(0.88, 0.723, r' $ p \leq 0.05 $', fontsize=10)

plt.gcf().text(0.85, 0.66, '***', fontsize=14)
plt.gcf().text(0.88, 0.68, r' $ p \leq 0.001 $', fontsize=10)
# ========================================================================================

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )
out.savefig(f'{y_column_name}_comparison.pdf', dpi=300)



def calculate_ttest():
    # # ======================calculate the fucking p -value confidence====================================
    # x_data1 = ['<50','<50', '50-100', '100-300']
    # x_data2=['300-500', '>500']
    from scipy.stats import ttest_ind
    our_method = 'CTransformer'
    print(result_dataframe)
    significance_stat = pd.DataFrame(
        columns=['Comparing Method', 'Cell Num of CTransformer', 'Cell Num of Another', 'p-value'])

    method_name_this_here = 'CShaper++'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======CShaper cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here = 'CShaper'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNetCShaper cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here = 'StarDist3D'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>===== cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here = 'SwinUNETR'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here = 'VNet'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNet cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here = '3DUNet++'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNet cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    class2 = \
    result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '<50')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '50-100')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '100-300')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = \
    result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '300-500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    class2 = result_dataframe[
        (result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName'] == '>500')][
        y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    significance_stat.to_csv(r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\middle materials\{}_ttest.csv'.format(y_column_name))


# calculate_ttest()
