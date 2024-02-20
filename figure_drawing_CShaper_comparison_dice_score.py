
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
evaluation_file_paths = glob.glob(os.path.join(evaluation_path, '*_score.csv'))

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

y_column_name = 'DiceScore'
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

    score_this_data = score_this_data[score_this_data[y_column_name] > 0.1]
    score_this_data = score_this_data[score_this_data[y_column_name] < 0.99]
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
out = sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 99),
                  hue_order=hue_order_list,palette=hue_palette)

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot


# =====================100=====================================================

x = [-0.4, -0.4, 0.4, 0.4] # VNet 0.018934763880916493
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')  # CShaper 0.61310019983598
font_dict_tmp={'size':25,'weight':'bold'}
plt.text((x[1] + x[2]) / 2 -0.25, y[0] + 0.03, 'n.s.',fontdict=font_dict_tmp)

# =-========================================================================

x = [0.6, 0.6, 1.4, 1.4]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.22, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.22, y[0] + 0.04, marker='*', s=150, color='black')

# =-========================================================================

x = [1.6, 1.6, 2.4, 2.4]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.22, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.22, y[0] + 0.04, marker='*', s=150, color='black')



# =-========================================================================
x = [2.6, 2.6, 3.4, 3.4]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.22, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.22, y[0] + 0.04, marker='*', s=150, color='black')



# =-========================================================================
x = [3.6, 3.6, 4.4, 4.4]
y = [1, 1.01, 1.01, 1]
plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.22, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.22, y[0] + 0.04, marker='*', s=150, color='black')




# =-========================================================================
x = [4.6, 4.6, 5.4, 5.4]
y = [1, 1.01, 1.01, 1]

plt.plot(x, y, linewidth=1, color='black')
plt.scatter((x[1] + x[2]) / 2-0.22, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2, y[0] + 0.04, marker='*', s=150, color='black')
plt.scatter((x[1] + x[2]) / 2+0.22, y[0] + 0.04, marker='*', s=150, color='black')



plt.xticks(fontsize=16)

plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)

plt.xlabel("Cell number", size=20)
plt.ylabel('Dice score', size=20)

plt.gcf().text(1.85, 0.85, r"Student's $t$-test (two sided)", fontsize=12)
plt.gcf().text(1.85, 0.78, ' n.s.', fontsize=10)
plt.gcf().text(1.88, 0.78, r' $ p > 0.10 $ (non-significant)', fontsize=10)
plt.gcf().text(1.85, 0.72, '***', fontsize=14)
plt.gcf().text(1.88, 0.735, r'$ p \leq 0.001 $', fontsize=10)


# plt.title(' Segmentation Comparison', size = 24 )
# plt.rcParams['figure.figsize'] = (16, 9)  # 6，8分别对应宽和高
# plt.ylim(0, 1)
out.savefig('text.pdf', dpi=300)

# ======================calculate the fucking p -value confidence====================================
from scipy.stats import ttest_ind
print(result_dataframe)
method_name_this_here='CShaper'
# =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======CShaper cell evaluation===============================
class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==100)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==100)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"CShaper t-statistic: {t_statistic:.2f}, p-value: ", p_value)

class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==200)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==200)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==300)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==300)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==400)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==400)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==500)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==500)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)

class1 = result_dataframe[(result_dataframe['Method'] == 'CMap(Ours)') & (result_dataframe['EmbryoName']==550)][y_column_name]
class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here) & (result_dataframe['EmbryoName']==550)][y_column_name]
t_statistic, p_value = ttest_ind(class1, class2)
print(method_name_this_here,len(class1),len(class2),f"t-statistic: {t_statistic:.2f}, p-value: ", p_value)


