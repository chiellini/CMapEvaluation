
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
evaluation_path = r'F:\CMap_paper\Code\Evaluation\Results\Comparison'
evaluation_file_paths = glob.glob(os.path.join(evaluation_path, '*_hausdorff_distance.csv'))

# Replace these with the names of the columns you want to use for the x and y axes
x_data = [100, 200, 300, 400, 500, 550]
embryo_name1 = '200109plc1p1'
embryo_name2 = '200113plc1p2'
embryo_times1 = [78, 114, 123, 157, 172, 181]
embryo_times2 = [90, 123, 132, 166, 178, 185]
embryo_namess = {}
for tpidx, tp in enumerate(embryo_times1):
    embryo_namess[embryo_name1 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data[tpidx]
for tpidx, tp in enumerate(embryo_times2):
    embryo_namess[embryo_name2 + '_' + str(tp).zfill(3) + '_segCell_uni'] = x_data[tpidx]
print(embryo_namess)

y_column_name = 'HausdorffDistance'
# y_column_name = 'IoU'

dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])

# Loop through the CSV files and plot the data
hue_order_list = [
    '3DCellSeg',
    'CellPose3D',
    'StarDist3D',

    'VNetCShaper',
'CShaper',
    'CMap']
for file_path in evaluation_file_paths:
    print(file_path)
    # plt.close()
    # fig = plt.figure(figsize=(16, 9), dpi=80)
    sns.set_theme()
    # Create a figure to hold the plots
    hdistance_this_data = pd.read_csv(file_path)

    hdistance_this_data = hdistance_this_data[hdistance_this_data[y_column_name] > 0.01]
    hdistance_this_data = hdistance_this_data[hdistance_this_data[y_column_name] < 2000]
    print(hdistance_this_data)

    hdistance_this_data[y_column_name] = hdistance_this_data[y_column_name] * 0.25 ** 3
    hdistance_this_data['RealEmbryoName'] = hdistance_this_data['EmbryoName']

    for key, value in embryo_namess.items():
        hdistance_this_data['EmbryoName'] = hdistance_this_data['EmbryoName'].replace(key, value)
    method_name = os.path.basename(file_path).split('_')[0]
    hdistance_this_data['Method'] = method_name
    # print(score_this_data)
    if method_name in hue_order_list:
        dataframe_list.append(hdistance_this_data)
result_dataframe = pd.concat(dataframe_list)
# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]
hue_palette ={'3DCellSeg':'#ff7c00',
    'CellPose3D':'#f14cc1',
    'StarDist3D':'#ffc400',
    'CShaper':'#00d7ff',
    'VNetCShaper':'#1ac938',
    'CMap': '#e8000b'}
# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=result_dataframe, kind="box", x="EmbryoName", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 95),
                  showfliers=False,
                  hue_order=hue_order_list,palette=hue_palette)

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot
# ======================save average information for each volumes=====================================
method_name_list = [
    '3DCellSeg',
    'CellPose3D',
    'StarDist3D',
    'CShaper',
    'VNetCShaper',
    'CMap']
embryo_names = ['200109plc1p1_078_segCell_uni', '200109plc1p1_114_segCell_uni', '200109plc1p1_123_segCell_uni',
                '200109plc1p1_157_segCell_uni','200109plc1p1_172_segCell_uni', '200109plc1p1_181_segCell_uni',
                '200113plc1p2_090_segCell_uni', '200113plc1p2_123_segCell_uni','200113plc1p2_132_segCell_uni',
                '200113plc1p2_166_segCell_uni', '200113plc1p2_178_segCell_uni', '200113plc1p2_185_segCell_uni']
evaluation_hd_table=pd.DataFrame(columns=['EmbryoName', 'MethodName', 'AvgHausdorffDistance'])
for embryo_name_tmp in embryo_names:
    for method_name_tmp in method_name_list:
        dice_score_pd=result_dataframe.loc[(result_dataframe['RealEmbryoName']==embryo_name_tmp) & (result_dataframe['Method']==method_name_tmp)]
        dice_score_list_tmp=list(dice_score_pd['HausdorffDistance'])
        if len(dice_score_list_tmp)==0:
            evaluation_hd_table.loc[len(evaluation_hd_table.index)] = [embryo_name_tmp, method_name_tmp, None]
        else:
            avg_hd_this= sum(dice_score_list_tmp) / len(dice_score_list_tmp)
            evaluation_hd_table.loc[len(evaluation_hd_table.index)]=[embryo_name_tmp, method_name_tmp, avg_hd_this]

evaluation_hd_table.to_csv('Table S - HausdorffDistanceEvaluation.csv', index=False)

star_symbol_size=18
star_interval_coe_hei=0.012
star_interval_coe_wid=0.03
font_dict_tmp={'size':13}
# =====================100 cell stage =====================================================



height = 0.02
width = 0.135
x = [0.2, 0.2, 0.35, 0.35] # VNet 0.056393318969177667
y = [0.3, 0.305, 0.305, 0.3]

plt.plot(x, y, linewidth=1, color='black')  # CShaper 0.343757250220425
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.0075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # stardist 3d  0.007906047211452813
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # cell pose 3d 0.002819925650792936
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # 3DCellSeg  5.458612510126308e-07
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

# =-==================200 cell stage======================================================
x = [1.18, 1.18, 1.35, 1.35]
y = [0.3, 0.305, 0.305, 0.3] # VNet 0.15688431772345812
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.0075, 'n.s.',fontdict=font_dict_tmp)



x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*6, y[0] + 0.0075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

# =-==================300 cell stage======================================================
x = [2.2, 2.2, 2.35, 2.35]
y = [0.3, 0.305, 0.305, 0.3]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.0075, 'n.s.',fontdict=font_dict_tmp)

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') #  VNet 0.030110975972713123
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-star_interval_coe_wid*4, y[0] + 0.0075, 'n.s.',fontdict=font_dict_tmp)


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

# =-==============400 cell stage==========================================================
x = [3.2, 3.2, 3.35, 3.35]
y = [0.3, 0.305, 0.305, 0.3]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black') # Vnet 0.015519773669590321
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*1.5, y[0]+star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


# =-=====================500===================================================
x = [4.2, 4.2, 4.35, 4.35]
y = [0.3, 0.305, 0.305, 0.3]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')


# =-========================================================================
x = [5.2, 5.2, 5.35, 5.35]
y = [0.3, 0.305, 0.305, 0.3]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

x = [x[0] - width, x[1] - width, x[2], x[3]]
y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] +star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')
plt.scatter((x[1] + x[2]) / 2+star_interval_coe_wid*3, y[0] + star_interval_coe_hei, marker='*', s=star_symbol_size, color='black')

plt.xticks(fontsize=16)

plt.yticks([0.0,0.1,0.2,0.3],fontsize=16)

plt.xlabel("Cell number", size=18)
plt.ylabel(r'Hausdorff distance ($\mu$m)', size=18)

plt.gcf().text(0.85, 0.85, r"Student's $t$-test (two sided)", fontsize=12)

# plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)

plt.gcf().text(0.85, 0.805, ' n.s.', fontsize=10)
plt.gcf().text(0.88, 0.805, r' $ p > 0.10 $ (non-significant)', fontsize=10)

plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
plt.gcf().text(0.88, 0.765, r' $ p \leq 0.10 $', fontsize=10)

plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
plt.gcf().text(0.88, 0.723, r' $ p \leq 0.05 $', fontsize=10)


plt.gcf().text(0.85, 0.66, '***', fontsize=14)
plt.gcf().text(0.88, 0.68, r' $ p \leq 0.001 $', fontsize=10)

# plt.title(' Segmentation Comparison', size = 24 )

# plt.show()

# out.savefig('text.eps', dpi=300)
# out.savefig('text.svg', dpi=300)
out.savefig('text.pdf', dpi=300)



# # ======================calculate the fucking p -value confidence====================================
# from scipy.stats import ttest_ind
# print(result_dataframe)
# method_name_this_here='CShaper'
# # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======CShaper cell evaluation===============================
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# method_name_this_here='VNetCShaper'
# # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNetCShaper cell evaluation===============================
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# method_name_this_here='3DCellSeg'
# # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======3DCellSeg cell evaluation===============================
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# method_name_this_here='CellPose3D'
# # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======CellPose3D cell evaluation===============================
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# method_name_this_here='StarDist3D'
# # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======StarDist3D cell evaluation===============================
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)
#
# class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
# class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
# t_statistic, p_value = ttest_ind(class1, class2)
# print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)


