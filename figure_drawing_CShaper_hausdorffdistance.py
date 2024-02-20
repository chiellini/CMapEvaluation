
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
    # '3DCellSeg',
    # 'CellPose3D',
    # 'StarDist3D',
    'CShaper',
    # 'VNetCShaper',
    'CMap']
hue_palette ={'3DCellSeg':'#ff7c00',
    'CellPose3D':'#f14cc1',
    'StarDist3D':'#ffc400',
    'CShaper':'#00d7ff',
    'VNetCShaper':'#1ac938',
    'CMap': '#e8000b'}
for file_path in evaluation_file_paths:
    print(file_path)
    # plt.close()
    # fig = plt.figure(figsize=(16, 9), dpi=80)
    sns.set_theme()
    # Create a figure to hold the plots
    score_this_data = pd.read_csv(file_path)

    score_this_data = score_this_data[score_this_data[y_column_name] > 1]
    score_this_data = score_this_data[score_this_data[y_column_name] < 20]
    score_this_data[y_column_name] = score_this_data[y_column_name]*0.25**3
    print(score_this_data)

    for key, value in embryo_namess.items():
        score_this_data['EmbryoName'] = score_this_data['EmbryoName'].replace(key, value)
    method_name = os.path.basename(file_path).split('_')[0]
    score_this_data['Method'] = method_name
    # print(score_this_data)
    if method_name in hue_order_list:
        dataframe_list.append(score_this_data)
result_dataframe = pd.concat(dataframe_list)
print(len(result_dataframe[result_dataframe['Method']=='CMap']),result_dataframe[result_dataframe['Method']=='CMap'][y_column_name].mean())
print(len(result_dataframe[result_dataframe['Method']=='CShaper']),result_dataframe[result_dataframe['Method']=='CShaper'][y_column_name].mean())

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=result_dataframe, kind="box", x="EmbryoName", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 99),
                  showfliers=False,
                  hue_order=hue_order_list,palette=hue_palette)

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot


# =====================100 cell stage =====================================================
height = 0.008
width = 0.2
x = [-0.42, -0.42, 0.42, 0.42]
y = [0.08, 0.085, 0.085, 0.08]

plt.plot(x, y, linewidth=1, color='black')  # CShaper 0.343757250220425
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.0085, s=10,facecolor='none',edgecolors='black')
font_dict_tmp={'size':25,'weight':'bold'}
plt.text((x[1] + x[2]) / 2-0.25, y[0] + 0.0085,'n.s.',fontdict=font_dict_tmp)

# =-==================200 cell stage======================================================
x = [0.58, 0.58, 1.42, 1.42]
y = [0.08, 0.085, 0.085, 0.08]
plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-0.25, y[0] + 0.0085,'n.s.',fontdict=font_dict_tmp)

# =-==================300 cell stage======================================================
x = [1.58, 1.58, 2.42, 2.42]
y = [0.08, 0.085, 0.085, 0.08]

plt.plot(x, y, linewidth=1, color='black')
plt.text((x[1] + x[2]) / 2-0.25, y[0] + 0.0085,'n.s.',fontdict=font_dict_tmp)

# =-==============400 cell stage==========================================================
x = [2.58, 2.58, 3.42, 3.42]
y = [0.12, 0.125, 0.125, 0.12]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.25, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.25, y[0] + 0.019, marker='*', s=150, color='black')

# =-==========500 cell stage==============================================================
x = [3.58, 3.58, 4.42, 4.42]
y = [0.16, 0.165, 0.165, 0.16]

plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.25, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.25, y[0] + 0.019, marker='*', s=150, color='black')

# =-=============550 cell stage===========================================================
x = [4.58, 4.58, 5.42, 5.42]
y = [0.21, 0.215, 0.215, 0.21]

plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.25, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.019, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.25, y[0] + 0.019, marker='*', s=150, color='black')


plt.xticks(fontsize=16)

plt.yticks([0.0,0.1,0.2,0.3],fontsize=16)

plt.xlabel("Cell number", size=18)
plt.ylabel(r'Hausdorff distance ($\mu$m)', size=18)

plt.gcf().text(1.85, 0.85, r"Student's $t$-test (two sided)", fontsize=12)

# plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)

plt.gcf().text(1.85, 0.805, ' n.s.', fontsize=10)
plt.gcf().text(1.88, 0.805, r' $ p > 0.10 $ (non-significant)', fontsize=10)
#
# plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
# plt.gcf().text(0.88, 0.765, r' $ p \leq 0.10 $', fontsize=10)
#
# plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
# plt.gcf().text(0.88, 0.723, r' $ p \leq 0.05 $', fontsize=10)


plt.gcf().text(1.85, 0.66, '***', fontsize=14)
plt.gcf().text(1.88, 0.68, r' $ p \leq 0.001 $', fontsize=10)

# plt.title(' Segmentation Comparison', size = 24 )



out.savefig('text.pdf', dpi=300)

