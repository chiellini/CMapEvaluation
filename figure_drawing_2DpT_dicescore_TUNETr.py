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
evaluation_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\Comparison2DTimelapseDiceIoU'
evaluation_file_paths = glob.glob(os.path.join(evaluation_path, '*_score.csv'))

# Replace these with the names of the columns you want to use for the x and y axes

embryo_name = '200113plc1p2'
max_time = 255

embryo_namess = {}
for tp in range(1, max_time + 1):
    embryo_namess[embryo_name + '_' + str(tp).zfill(3) + '_128_uni'] = tp

y_column_name = 'IoU'
# y_column_name = 'IoU'

dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])

# Loop through the CSV files and plot the data
hue_order_list = [
    'StarDist3D',
    'CShaper++',
    'CTransformer']
hue_palette = {
    'StarDist3D': '#ffc400',
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
print(result_dataframe, result_dataframe.columns)

# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.lineplot(data=result_dataframe, x="EmbryoName", y=y_column_name, hue='Method',
                  errorbar=('ci', 99),
                  hue_order=hue_order_list, palette=hue_palette)

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )
# out.savefig(f'{y_column_name}_comparison.pdf', dpi=300)

plt.xticks([0,50,100,150,200,250],fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Developmental Time Point", size=20)
plt.ylabel('Jaccard Index', size=20)
plt.savefig("2d EVALUATION.pdf", format="pdf",dpi=300)
plt.show()

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
    class1 = result_dataframe[(result_dataframe['Method'] == our_method)][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here)][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"CShaper t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    method_name_this_here = 'StarDist3D'
    # =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=======VNetCShaper cell evaluation===============================
    class1 = result_dataframe[(result_dataframe['Method'] == our_method)][y_column_name]
    class2 = result_dataframe[(result_dataframe['Method'] == method_name_this_here)][y_column_name]
    t_statistic, p_value = ttest_ind(class1, class2)
    significance_stat.loc[len(significance_stat)] = [method_name_this_here, len(class1), len(class2), p_value]
    print(method_name_this_here, len(class1), len(class2), f"StarDist3D t-statistic: {t_statistic:.2f}, p-value: ",
          p_value)

    significance_stat.to_csv(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\middle materials\2d_JI_{}_ttest.csv'.format(
            y_column_name))

calculate_ttest()
