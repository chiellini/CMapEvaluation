# ===============================================================================================
# plot bar , direct comparison between different segmentation methods
# ===============================================================================================
import os.path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

# Replace these with the names of your CSV files
evaluation_file_path = os.path.join(r'F:\CMap_paper\Code\Evaluation\Results\Comparison','CMap_score.csv')

cell_fate_dict={}
cell_fate_pd=pd.read_csv(os.path.join(r'F:\CMap_paper\Figures\Comparison','cell_fate_dictionary.csv'),index_col=0,names=['Fate'])
for index in cell_fate_pd.index:
    cell_fate_dict[index[:-1]]=cell_fate_pd.loc[index]['Fate'][:-1]
print(cell_fate_dict)

# Replace these with the names of the columns you want to use for the x and y axes
x_data = [100, 200, 300, 400, 500, 550]
embryo_name1 = '200109plc1p1'
embryo_name2 = '200113plc1p2'
embryo_times1 = [78, 114, 123, 157, 172, 181]
embryo_times2 = [90, 123, 132, 166, 178, 185]
embryo_namess = {}
embryo_name_file_txt_map={}
for tpidx, tp in enumerate(embryo_times1):
    embryo_tp_file_name_this=embryo_name1 + '_' + str(tp).zfill(3)
    embryo_namess[embryo_tp_file_name_this+ '_segCell_uni'] = x_data[tpidx]
    with open(os.path.join(r'F:\CMap_paper\CMapEvaluation\CMap\niigz',embryo_tp_file_name_this+'_segCell.txt'), 'r') as f:
        embryo_name_file_txt_map[embryo_tp_file_name_this+ '_segCell_uni'] = json.loads(f.read())
for tpidx, tp in enumerate(embryo_times2):
    embryo_tp_file_name_this = embryo_name2 + '_' + str(tp).zfill(3)
    embryo_namess[embryo_tp_file_name_this + '_segCell_uni'] = x_data[tpidx]
    with open(os.path.join(r'F:\CMap_paper\CMapEvaluation\CMap\niigz', embryo_tp_file_name_this + '_segCell.txt'),
              'r') as f:
        embryo_name_file_txt_map[embryo_tp_file_name_this + '_segCell_uni'] = json.loads(f.read())
print(embryo_namess)

y_column_name = 'DiceScore'
data_folder = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\GUIData\WebData_CMap_cell_label_v3"
name_file_path = data_folder + "/name_dictionary.csv"
label_name_dict = pd.read_csv(name_file_path, index_col=0).to_dict()['0']


dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])

# Loop through the CSV files and plot the data
# hue_order_list = [
#     'AB',
#     'C',
#     'D',
#     'E',
#     'MS']
hue_order_list=[
    'Neuron',
    'Pharynx',
    'Skin',
    'Muscle',
    'Intestine',
    'Germline',
    'Death',
    'Others',
    'Unspecified'
]
def celllabel_to_cellname(row):
    # print(row)
    uni_celllabel=row['CellLabel']
    embryo_name_this=row['EmbryoName']
    # print(embryo_name_file_txt_map[embryo_name_this])
    celllabel=embryo_name_file_txt_map[embryo_name_this].get(str(uni_celllabel),0)
    return label_name_dict.get(celllabel,'LOSS')


def lineage_defined(cellname):
    if cellname[:1] in hue_order_list:
        return cellname[:1]
    else:
        return cellname[:2]

def fate_defined(cellname):
    cell_fate=cell_fate_dict.get(cellname,'Unspecified')
    return cell_fate

# plt.close()
# fig = plt.figure(figsize=(16, 9), dpi=80)
sns.set_theme()
# Create a figure to hold the plots
score_this_data = pd.read_csv(evaluation_file_path)

# score_this_data = score_this_data[score_this_data[y_column_name] > 0.1]
score_this_data = score_this_data[score_this_data[y_column_name] < 0.99]

score_this_data['CellName'] = score_this_data.apply(celllabel_to_cellname,axis=1)

for key, value in embryo_namess.items():
    score_this_data['EmbryoName'] = score_this_data['EmbryoName'].replace(key, value)
score_this_data['Fate'] = score_this_data['CellName'].map(fate_defined)
score_this_data['Lineage'] = score_this_data['CellName'].map(lineage_defined)

score_this_data.to_csv('text.csv')
# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=score_this_data, kind="bar", x="EmbryoName", y=y_column_name, hue='Fate', height=4, aspect=2,
                  # errorbar=('ci', 99),
                  hue_order=hue_order_list)

# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot
plt.show()

# # =====================100 cell stageeeeee=====================================================
# height = 0.03
# width = 0.13
# x = [0.2, 0.2, 0.35, 0.35] # VNet 0.018934763880916493
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')  # CShaper 0.61310019983598
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, s=10,facecolor='none',edgecolors='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black') # stardist 3d  0.06263100827713756
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black') # cell pose 3d 0.09477472023042054
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black') # 3DCellSeg  0.004422245159720624
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
#
# # =-========================================================================
# height = 0.03
# width = 0.13
# x = [1.18, 1.18, 1.35, 1.35]
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# # =-========================================================================
# height = 0.03
# width = 0.13
# x = [2.2, 2.2, 2.35, 2.35]
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# # =-========================================================================
# height = 0.03
# width = 0.13
# x = [3.2, 3.2, 3.35, 3.35]
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
#
# # =-========================================================================
# height = 0.03
# width = 0.13
# x = [4.2, 4.2, 4.35, 4.35]
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
#
# # =-========================================================================
# height = 0.03
# width = 0.13
# x = [5.2, 5.2, 5.35, 5.35]
# y = [1, 1.01, 1.01, 1]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# x = [x[0] - width, x[1] - width, x[2], x[3]]
# y = [y[0] + height, y[1] + height, y[2] + height, y[3] + height]
# plt.plot(x, y, linewidth=1, color='black')
# plt.scatter((x[1] + x[2]) / 2-0.05, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2, y[0] + 0.02, marker='*', s=10, color='black')
# plt.scatter((x[1] + x[2]) / 2+0.05, y[0] + 0.02, marker='*', s=10, color='black')
#
# plt.xlabel("Cell Number", size=20)
# plt.ylabel('Dice Score', size=20)
#
# plt.gcf().text(0.85, 0.85, 'Student\'s t-test \n (two sided)', fontsize=12)
#
# # plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)
#
# plt.gcf().text(0.85, 0.80, '  o', fontsize=12)
# plt.gcf().text(0.88, 0.80, r'$ p > 0.10 $', fontsize=10)
#
# plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
# plt.gcf().text(0.88, 0.765, r'$ p \leq 0.10 $', fontsize=10)
#
# plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
# plt.gcf().text(0.88, 0.723, r'$ p \leq 0.05 $', fontsize=10)
#
#
# plt.gcf().text(0.85, 0.66, '***', fontsize=14)
# plt.gcf().text(0.88, 0.68, r'$ p \leq 0.001 $', fontsize=10)

# plt.title(' Segmentation Comparison', size = 24 )



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

