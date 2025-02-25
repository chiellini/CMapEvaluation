
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
# Loop through the CSV files and plot the data
evaluation_path = r'Results/ComparisonHausdorff'
evaluation_file_paths = glob.glob(os.path.join(evaluation_path, '*_cell_wise_evaluation.csv'))

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

y_column_name = 'HausdorffDistance95'
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
    hdistance_this_data = pd.read_csv(file_path)

    # hdistance_this_data = hdistance_this_data[hdistance_this_data[y_column_name] > 0.01]
    hdistance_this_data = hdistance_this_data[hdistance_this_data[y_column_name] < 20]

    print(hdistance_this_data)

    hdistance_this_data[y_column_name] = hdistance_this_data[y_column_name] * 0.18
    hdistance_this_data['RealEmbryoName'] = hdistance_this_data['EmbryoName']

    for key, value in embryo_namess.items():
        hdistance_this_data['EmbryoName'] = hdistance_this_data['EmbryoName'].replace(key, value)
    method_name = os.path.basename(file_path).split('_')[0]
    hdistance_this_data['Method'] = method_name
    # print(score_this_data)
    if method_name in hue_order_list:
        dataframe_list.append(hdistance_this_data)

result_dataframe = pd.concat(dataframe_list)
result_dataframe.to_csv('test.csv')
# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=result_dataframe, kind="box", x="EmbryoName", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 95),
                  showfliers=False,
                  hue_order=hue_order_list,palette=hue_palette)

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot

# ======================plotting the fucking p -value confidence====================================

star_symbol_size=18
star_interval_coe_hei=0.12
star_interval_coe_wid=0.02
font_dict_tmp={'size':10}
# =====================<50 cell stage =====================================================
height = 0.15
width = 0.1165

this_start_x=0.23
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1.8, 1.85, 1.85, 1.8]
plt.plot(x, y, linewidth=1, color='black')  # CShaper++ 0.115037131
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]] # CShaper 0.5
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # stardist 3d  0.007906047211452813
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # SwinUNETR
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # VNet
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3DUNet
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*2, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)


# =-==================50-100 cell stage======================================================
this_start_x=1.21
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1.4, 1.45, 1.45, 1.4] # cshaper++
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height] # cshaper
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # stardist3d
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # vnet
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3dunet++
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

# =-==================100-300 cell stage======================================================
this_start_x=2.22
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [1.1, 1.15, 1.15, 1.1]
plt.plot(x, y, linewidth=1, color='black') # cshaper++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #  cshaper 0.030110975972713123
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #stardist3d
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')# swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # vnet
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3dunet++
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

# =-============================300-500 cell stage==========================================================
this_start_x=3.22
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [0.9, 0.95, 0.95, 0.9]
plt.plot(x, y, linewidth=1, color='black') # cshaper ++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


x = [x[0] - width, x[1] - width, x[2], x[3]] # cshaper
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # Vnet 0.015519773669590321
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #stardist3d
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # swinunetr
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # vnet
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3dunet++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


# =-=====================>500===================================================
this_start_x=4.24
x = [this_start_x, this_start_x, this_start_x+width, this_start_x+width]
y = [2.7, 2.75, 2.75, 2.7]
plt.plot(x, y, linewidth=1, color='black') #cshaper++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # cshaper++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #stardist3d
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # swinunetr
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # vnet
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3dunet++
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


plt.xticks(fontsize=16)
plt.yticks([0.5,1.0,1.5,2.0,2.5,3.0],fontsize=16)

plt.xlabel("Cell number", size=20)
plt.ylabel(r'Hausdorff Distance ($\mu$m)', size=20)

plt.gcf().text(0.85, 0.85, r"Student's $t$-test (two sided)", fontsize=12)

# plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)

plt.gcf().text(0.85, 0.805, ' n.s.', fontsize=10)
plt.gcf().text(0.88, 0.805, r' $ p > 0.10 $ (non-significant)', fontsize=10)

plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
plt.gcf().text(0.88, 0.765, r' $ p \leq 0.10 $', fontsize=10)

plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
plt.gcf().text(0.88, 0.723, r' $ p \leq 0.05 $', fontsize=10)


plt.gcf().text(0.85, 0.66, '***', fontsize=14)
plt.gcf().text(0.88, 0.68, r' $ p \leq 0.01 $', fontsize=10)

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )

# plt.show()

# out.savefig('text.eps', dpi=300)
# out.savefig('text.svg', dpi=300)
out.savefig(f'{y_column_name}_comparison.pdf', dpi=300)

# ==================================================================================================

def calculate_ttest():
    # # ======================calculate the fucking p -value confidence====================================
    # x_data1 = ['<50','<50', '50-100', '100-300']
    # x_data2=['300-500', '>500']
    from scipy.stats import ttest_ind
    our_method='CTransformer'
    print(result_dataframe)
    significance_stat=pd.DataFrame(columns=['Comparing Method','Cell Num of CTransformer', 'Cell Num of Another','p-value'])

    method_name_this_here='CShaper++'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======CShaper cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    print(class1,class2)
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)


    method_name_this_here='CShaper'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNetCShaper cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here='StarDist3D'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>===== cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here='SwinUNETR'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here='VNet'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNet cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    method_name_this_here='3DUNet++'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNet cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='<50')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='50-100')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='100-300')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='300-500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    class1 = result_dataframe[(result_dataframe['Method'] == our_method) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']=='>500')][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)]=[method_name_this_here,len(class1),len(class2), p_value]
    print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

    significance_stat.to_csv(r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\middle materials\{}_ttest.csv'.format(y_column_name))


# calculate_ttest()