import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

gui_data_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\CMapSubmission\Dataset Access\Dataset E'

# 4-cell
calculating_embryo_names_and_zero_timing_4cell = {'WT_Sample1': 11, 'WT_Sample2': 3, 'WT_Sample3': 2, 'WT_Sample4': 13,
                                            'WT_Sample5': 7, 'WT_Sample6': 12, 'WT_Sample7': 1, 'WT_Sample8': 3}

cells_1_list = ['MSappp', 'MSappa', 'MSapap']
cells_2_list = ['ABplpapa','ABplpapp']
# cells_2='ABplpapp'
#
# embryo_names_and_zero_timing_this_cell_pairs={}
#
# pd_contact_time_middle = pd.DataFrame(columns=['Embryo Name', 'Contact Area', 'Ligand Cell', 'Time','Receptor Cell'])
#
# for embryo_name, start_tp in calculating_embryo_names_and_zero_timing_4cell.items():
#     volume_file = pd.read_csv(os.path.join(gui_data_root, embryo_name, '{}_volume.csv'.format(embryo_name)))
#     volume_tem=volume_file[cells_2_list[0]]
#     mask_tem = volume_tem.notna()
#     data_this_this = volume_tem.loc[mask_tem]
#     for tp_this,value_this in data_this_this.items():
#         if value_this>0:
#             embryo_names_and_zero_timing_this_cell_pairs[embryo_name]=int(tp_this)
#             break
#
# for embryo_name, start_tp in embryo_names_and_zero_timing_this_cell_pairs.items():
#     contact_file = pd.read_csv(os.path.join(gui_data_root, embryo_name, '{}_Stat.csv'.format(embryo_name)),
#                                index_col=['cell1', 'cell2'])
#     # contact_sum=0
#     for ligand_cell in cells_1_list:
#         for receptor_cell in cells_2_list:
#             if (ligand_cell, receptor_cell) in contact_file.index:
#                 data_this = contact_file.loc[(ligand_cell, receptor_cell)]
#                 mask_tem = data_this.notna()
#                 data_this_this = data_this.loc[mask_tem]
#             elif (receptor_cell, ligand_cell) in contact_file.index:
#                 data_this = contact_file.loc[(receptor_cell, ligand_cell)]
#                 mask_tem = data_this.notna()
#                 data_this_this = data_this.loc[mask_tem]
#             else:
#                 continue
#
#             for tp_this, value_this in data_this_this.items():
#                 if value_this > 0:
#                     pd_contact_time_middle.loc[len(pd_contact_time_middle.index)] = [embryo_name, value_this, ligand_cell,
#                                                                                      (int(tp_this) - start_tp) * 1.43,receptor_cell]
#
# pd_contact_time_for_plotting = pd.DataFrame(columns=['Embryo Name', 'Contact Area Sum', 'Time','Receptor Cell'])
# for embryo_name, start_tp in embryo_names_and_zero_timing_this_cell_pairs.items():
#     this_embryo_name_pd=pd_contact_time_middle.loc[pd_contact_time_middle['Embryo Name']==embryo_name]
#     for receptor_cell in cells_2_list:
#         this_embryo_name_recep_pd=this_embryo_name_pd.loc[this_embryo_name_pd['Receptor Cell']==receptor_cell]
#         timings_this_emb=set(this_embryo_name_recep_pd['Time'])
#         for timing_tem in timings_this_emb:
#             this_embryo_name_this_tp_contact_sum=sum(this_embryo_name_recep_pd.loc[this_embryo_name_recep_pd['Time']==timing_tem]['Contact Area'])
#             pd_contact_time_for_plotting.loc[len(pd_contact_time_for_plotting.index)]=[embryo_name,this_embryo_name_this_tp_contact_sum,timing_tem,receptor_cell]
#
# # timings_this_emb=set(pd_contact_time_for_plotting['Time'])
# # for timing_tem in timings_this_emb:
# #     embs_contact_this_tp_list=list(pd_contact_time_for_plotting.loc[pd_contact_time_for_plotting['Time']==timing_tem]['Contact Area Sum'])
# #     all_embryo_name_this_tp_contact_avg=sum(embs_contact_this_tp_list)/len(embs_contact_this_tp_list)
# #     pd_contact_time_for_plotting.loc[len(pd_contact_time_for_plotting.index)]=['Average',all_embryo_name_this_tp_contact_avg,timing_tem]
# pd_contact_time_for_plotting.to_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r1mc4',
#     'contact_calculating_r1mc4.csv'))

# =================================================================================
pd_contact_time_for_plotting=pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r1mc4',
    'contact_calculating_r1mc4.csv'))

pd_contact_time_for_plotting_ABplpapa=pd_contact_time_for_plotting.loc[pd_contact_time_for_plotting['Receptor Cell']=='']

# ax = sns.lineplot(data=pd_contact_time_for_plotting, x='Time', y='Contact Area Sum', hue='Embryo Name',
#                   # style='Cell Name',
#                   markers=True, dashes=False)
# h, l = ax.get_legend_handles_labels()
# ax.legend(h, l, ncol=2)
# plt.show()

sns.lineplot(data=pd_contact_time_for_plotting,x='Time', y='Contact Area Sum', hue='Receptor Cell',errorbar='sd', err_style="band")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel("Developmental time since appearance of \n ABplpapa and ABplpapp (min)", size=14)
plt.ylabel('Total cell-cell contact area with \n MSapap, MSappa, and MSappp ($\mu$m$^2$)', size=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,prop={'size': 14})

plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r1mc4',
    'contact_calculating_r1mc4.pdf'), format="pdf",dpi=300)

plt.show()
