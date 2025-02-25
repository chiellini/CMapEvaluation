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

column_name1 = 'Cell Stage'
column_name2 = 'Cell Volume Relative Error'

# pd_gt_seg_cell_volume = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
#     'gt_seg_cell.csv'))

#
# pd_gt_seg_cell_volume[column_name1] = pd_gt_seg_cell_volume['Time Point']
# pd_gt_seg_cell_volume[column_name2] = pd_gt_seg_cell_volume['Variation Ratio']
#
# tp_and_cell_stage_dict = {
#     'WT_Sample1': {'90': '~100', '123': '~200', '132': '~300', '166': '~400', '178': '~500',
#                    '185': '~550'},
#     'WT_Sample7': {'78': '~100', '114': '~200', '123': '~300', '157': '~400', '172': '~500',
#                    '181': '~550'}
#     }
#
# df_tem_list=[]
#
# for embryo_name_key, value in tp_and_cell_stage_dict.items():
#     new_tem_df=pd_gt_seg_cell_volume.loc[pd_gt_seg_cell_volume['Embryo Name'] == embryo_name_key]
#     for tp_this, cell_stage_this in value.items():
#         new_tem_df['Cell Stage'] = new_tem_df['Cell Stage'].replace(int(tp_this), cell_stage_this)
#     df_tem_list.append(new_tem_df)
# result_dataframe = pd.concat(df_tem_list)
#
# # result_dataframe = pd.concat(dataframe_list)
# result_dataframe.to_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
#     'gt_seg_cell_cell_stage.csv'))


result_dataframe=pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
    'gt_seg_cell_cell_stage.csv'))


for cell_stage_this in set(result_dataframe['Cell Stage']):
    series_this = result_dataframe.loc[result_dataframe['Cell Stage']==cell_stage_this]['Variation Ratio']
    print(cell_stage_this,series_this.mean())
# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]
f, ax = plt.subplots(figsize=(7, 6))

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
# out = sns.catplot(data=result_dataframe, kind="box", x=column_name1, y=column_name2, height=4, aspect=2,
#                   errorbar=('ci', 95),
#                   showfliers=False)

# sns.boxplot(
#     result_dataframe, x=column_name2, y=column_name1,
#     whis=[0, 100], width=.6,
# )
# Add in points to show each observation
# sns.stripplot(result_dataframe[result_dataframe[column_name2]<0.66], x=column_name1, y=column_name2, size=1, color='green')
sns.pointplot(data=result_dataframe, x=column_name1, y=column_name2,
    dodge=False, join=False, errorbar='sd',capsize=.16,hue='Cell Number')
# Tweak the visual presentation
plt.legend(loc='upper right', bbox_to_anchor=(1.29, 1),prop={'size': 16})


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel("Cell number", size=20)
plt.ylabel(r'Relative variation of cell volume', size=20)

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )


# out.savefig('text.eps', dpi=300)
# out.savefig('text.svg', dpi=300)
plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
    'figr2.pdf'), dpi=300)

plt.show()


