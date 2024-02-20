
# ===============================================================================================
# plot bar , direct comparison between different segmentation methods
# ===============================================================================================
import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib import rcParams


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

# y_column_name = 'HausdorffDistance'
y_column_name1 = 'IoU'
y_column_name2 = 'DiceScore'


dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])
down_boundary_list=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
high_boundary=0.99

# Loop through the CSV files and plot the data
hue_order_list = [
    '3DCellSeg',
    'CellPose3D',
    'StarDist3D',
    'CShaper',
    'VNetCShaper',
    'CMap(Ours)']
df_iou_vs_avgdicescore=pd.DataFrame(columns=['method','avg_dice','iou_threshold'])
for iou_threshold_this in down_boundary_list:
    for file_path in evaluation_file_paths:
        print(file_path)
        method_name = os.path.basename(file_path).split('_')[0]
        if method_name in hue_order_list:
            # Create a figure to hold the plots
            score_this_data = pd.read_csv(file_path)

            score_this_data = score_this_data[score_this_data[y_column_name1] > iou_threshold_this]
            score_this_data = score_this_data[score_this_data[y_column_name1] < high_boundary]
            if len(score_this_data)==0:
                print(method_name)
            else:
                dice_score_list_tmp=list(score_this_data[y_column_name2])
                average_dice_score_for_this_iou_threshold=sum(dice_score_list_tmp)/len(dice_score_list_tmp)
                df_iou_vs_avgdicescore.loc[len(df_iou_vs_avgdicescore)] = [method_name,average_dice_score_for_this_iou_threshold,iou_threshold_this]

# result_dataframe = result_dataframe[result_dataframe['Method'] in hue_order_list]

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
rcParams['figure.figsize'] = 16,9
sns.set_theme()

out = sns.lineplot(data=df_iou_vs_avgdicescore, x="iou_threshold", y='avg_dice', hue='method',hue_order=hue_order_list)
plt.show()
# sns.lineplot(data=score_this_data, x="EmbryoName", y=y_column_name,ax=ax)
# Show the plot



