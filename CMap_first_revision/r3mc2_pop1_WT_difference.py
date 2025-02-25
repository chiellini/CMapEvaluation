# E AND MS irregularity , along cells

import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu,wilcoxon

gui_data_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\CMapSubmission\Dataset Access\Dataset E'

sub_lineages=['E sublineage','MS sublineage']

pd_volume_surface_irregularity_wt = pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_some_precedent_cells.csv'))
pd_volume_surface_irregularity_wt = pd_volume_surface_irregularity_wt.loc[
    pd_volume_surface_irregularity_wt['Embryo Name'] == 'Average']

pd_volume_surface_irregularity_pop1 = pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_some_precedent_cells_pop1.csv'))
pd_volume_surface_irregularity_pop1 = pd_volume_surface_irregularity_pop1.loc[
    pd_volume_surface_irregularity_pop1['Embryo Name'] == 'Average']

time_list= pd_volume_surface_irregularity_pop1['Time']

pd_wt_pop1_irregularity_difference = pd.DataFrame(columns=[ 'Time', 'Irregularity Difference', 'Comparing Lineages','Embryo Type'])

for single_time in time_list:
    irregularity_wt_dict={}
    irregularity_pop1_dict={}
    for target_lineage in sub_lineages:
        irregularity_wt_dict[target_lineage]=list(pd_volume_surface_irregularity_wt.loc[(pd_volume_surface_irregularity_wt['Precedent'] == target_lineage) &
                                                                                   (pd_volume_surface_irregularity_wt['Time']==single_time)]['Irregularity'])[0]
        irregularity_pop1_dict[target_lineage]=list(pd_volume_surface_irregularity_pop1.loc[(pd_volume_surface_irregularity_pop1['Precedent'] == target_lineage) &
                                                                                       (pd_volume_surface_irregularity_pop1['Time'] == single_time)]['Irregularity'])[0]
    pd_wt_pop1_irregularity_difference.loc[len(pd_wt_pop1_irregularity_difference)]=[single_time,abs(irregularity_wt_dict['E sublineage']-irregularity_wt_dict['MS sublineage']),'|MS-E| of WT','WT']
    pd_wt_pop1_irregularity_difference.loc[len(pd_wt_pop1_irregularity_difference)]=[single_time,abs(irregularity_pop1_dict['E sublineage']-irregularity_pop1_dict['MS sublineage']),'|MS-E| of pop1$^-$','pop1$^-$']
    pd_wt_pop1_irregularity_difference.loc[len(pd_wt_pop1_irregularity_difference)]=[single_time,abs(irregularity_wt_dict['E sublineage']-irregularity_pop1_dict['E sublineage']),'|E-E| of WT-pop1$^-$','WT-pop1$^-$']

pd_wt_pop1_irregularity_difference.to_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_pop1_wt_difference.csv'))

plt.figure(figsize=(16, 6))

ax=sns.lineplot(data=pd_wt_pop1_irregularity_difference, x='Time', y='Irregularity Difference',
                hue='Comparing Lineages', hue_order=['|MS-E| of WT', '|MS-E| of pop1$^-$','|E-E| of WT-pop1$^-$'],
             palette={'|MS-E| of WT':'red', '|MS-E| of pop1$^-$':'green','|E-E| of WT-pop1$^-$':'blue'},
             errorbar=None,
             # err_style="band"
             )

plt.xticks(fontsize=26,family='Arial')
plt.yticks(fontsize=26,family='Arial')

plt.xlabel("Developmental time since appearance of MS and E (min)", size=26,family='Arial')
plt.ylabel(r'Cell irregularity', size=26,family='Arial')
ax.yaxis.set_label_coords(-0.06,0.62)
plt.legend(prop={'size': 16})

# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,prop={'size': 16})

plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'contact_calculating_r3mc2.pdf'), format="pdf", dpi=300)

plt.show()
