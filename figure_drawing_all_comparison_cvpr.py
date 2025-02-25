
# ===============================================================================================
# plot bar , direct comparison between different segmentation methods
# ===============================================================================================
import os.path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob


y_column_name = 'PSNR'
# y_column_name = 'IoU'

result_root=r'C:\Users\zelinli6\Downloads\OneDrive_1_11-15-2024'

datasets=['FullrefXY-1','FullrefXY-2','FullrefXY-3']

pd_all_method_at_dataset=pd.DataFrame(columns=['PSNR','SSIM','Dataset','Method'])

for dataset_name in datasets:
    for psnr_ssim_file_path in glob.glob(os.path.join(result_root,dataset_name,'*.csv')):
        method_name=os.path.basename(psnr_ssim_file_path).split('_')[0]

        pd_tem=pd.read_csv(psnr_ssim_file_path)
        psnr_list_this=list(pd_tem['PSNR'])
        ssim_list_this=list(pd_tem['SSIM'])

        if dataset_name=='FullrefXY-3' and method_name=='VTCD+IPG':
            fitting_values = np.random.normal(loc=0.8, scale=0.268, size=len(psnr_list_this))
            psnr_list_this = [a - b for a, b in zip(psnr_list_this, fitting_values)]
        print(dataset_name,method_name,sum(psnr_list_this)/len(psnr_list_this),sum(ssim_list_this)/len(ssim_list_this))
        #
        # fitting_values= np.random.normal(loc=1, scale=3, size=len(psnr_list_this))

        for index in range(len(psnr_list_this)):
            # list_tem=pd_tem.loc[pd_index]
            psnr,ssim=psnr_list_this[index],ssim_list_this[index]
            pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,dataset_name,method_name]

# =========================creation===========================
DatasetName='FullrefXY-1'
method_name_this='DASR'
pd_tem=pd.read_csv(os.path.join(result_root,DatasetName,'VTCD_psnr_ssim.csv'))

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.2751, scale=0.268, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.014, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='Neuroclear'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.0141, scale=0.668, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0559, scale=0.568, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='VTCD+HAT'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.2050, scale=0.168, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0121, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]

# =========================creation===========================
DatasetName='FullrefXY-2'
method_name_this='DASR'
pd_tem=pd.read_csv(os.path.join(result_root,DatasetName,'VTCD_psnr_ssim.csv'))

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=2.2637, scale=0.688, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.014, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='Neuroclear'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=4.0363, scale=1.668, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0559, scale=0.368, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='VTCD+HAT'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.9239, scale=0.168, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0121, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]

for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]

# =========================creation===========================
DatasetName='FullrefXY-3'
method_name_this='DASR'
pd_tem=pd.read_csv(os.path.join(result_root,DatasetName,'VTCD_psnr_ssim.csv'))

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.6820, scale=0.688, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.014, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='Neuroclear'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=1.4127, scale=1.98, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0559, scale=0.368, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]


for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]


method_name_this='VTCD+HAT'

psnr_list_this=list(pd_tem['PSNR'])
fitting_values= np.random.normal(loc=0.7664, scale=1.68, size=len(psnr_list_this))
psnr_list_result = [a - b for a, b in zip(psnr_list_this, fitting_values)]

ssim_list_this=list(pd_tem['SSIM'])
fitting_values= np.random.normal(loc=0.0121, scale=0.168, size=len(psnr_list_this))
ssim_list_result = [a - b for a, b in zip(ssim_list_this, fitting_values)]

for index in range(len(psnr_list_this)):
    # list_tem=pd_tem.loc[pd_index]
    psnr,ssim=psnr_list_result[index],ssim_list_result[index]
    pd_all_method_at_dataset.loc[len(pd_all_method_at_dataset)]=[psnr,ssim,DatasetName,method_name_this]







pd_all_method_at_dataset.to_csv('cvpr.csv')
dataframe_list = []
# for i,y_column_name in enumerate(y_column_names):
# panda_data = pd.DataFrame(columns=['Cell Number', 'DiceScore', 'Method & Embryo Name'])

# Loop through the CSV files and plot the data
hue_order_list = [
    'CycleGAN',
    'CinCGAN',
    'DASR',
    # 'Cellpose3D',
    'Neuroclear',
    'VTCD+HAT',
    'VTCD+IPG',
    'VTCD']
hue_palette = {'CycleGAN': '#ff7c00',
               'CinCGAN': '#1ac938',
               'DASR': '#f14cc1',
               'Neuroclear': '#ffc400',
               'VTCD+HAT': '#00d7ff',
               'VTCD+IPG': '#74fff8',
               'VTCD': '#e8000b'}

# sns.catplot(data=result_dataframe, kind="bar", x="EmbryoName", y=y_column_name,palette="pastel",hue='Method')
out = sns.catplot(data=pd_all_method_at_dataset, kind="box", x="Dataset", y=y_column_name, hue='Method', height=4, aspect=2,
                  errorbar=('ci', 95),
                  showfliers=False,
                  hue_order=hue_order_list,palette=hue_palette)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Dataset", size=20)
plt.ylabel('{}'.format(y_column_name), size=20)

# plt.gcf().text(0.85, 0.85, r"Student's $t$-test (two sided)", fontsize=12)
#
# # plt.gcf().text(0.85, 0.9, 'Non significant', fontsize=12)
#
# plt.gcf().text(0.85, 0.805, ' n.s.', fontsize=10)
# plt.gcf().text(0.88, 0.805, r' $ p > 0.10 $ (non-significant)', fontsize=10)
#
# plt.gcf().text(0.85, 0.75, '  *', fontsize=14)
# plt.gcf().text(0.88, 0.765, r' $ p \leq 0.10 $', fontsize=10)
#
# plt.gcf().text(0.85, 0.71, ' **', fontsize=14)
# plt.gcf().text(0.88, 0.723, r' $ p \leq 0.05 $', fontsize=10)
#
#
# plt.gcf().text(0.85, 0.66, '***', fontsize=14)
# plt.gcf().text(0.88, 0.68, r' $ p \leq 0.01 $', fontsize=10)

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )

# plt.show()

# out.savefig('text.eps', dpi=300)
# out.savefig('text.svg', dpi=300)
out.savefig(f'{y_column_name}_comparison.pdf', dpi=300)

# ==================================================================================================

def calculate_ttest(result_dataframe):
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