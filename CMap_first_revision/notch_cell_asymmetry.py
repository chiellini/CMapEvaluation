import os.path

import pandas as pd
# import numpy as np

# ligand_embryos_root= r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Temporary Sharing\notch6to10KO'
# ligand_embryos_root=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Temporary Sharing\KO10th'
# ligand_embryos_root= r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\CMapSubmission\Dataset Access\Dataset E'
ligand_embryos_root=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Temporary Sharing\notch10_15_2emb'

saving_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\notch_cell_asymmetry'
embryo_name_cell_pair={
    # 'ABplpapp':['241025plc1KOp1'], # 6th
    #                    'ABplpappa':['241112plc1KOp2','241112plc1KOp1'], # 7th, 8th
    #
                       # 'ABplpappaa':['241112plc1KOp1'], # 9th
    #
    #                 'ABprpapp':['241007plc1KOp1','241107plc1KOp1','241022plc1KOp1'], # 10th
# 'ABprpapp':['241209plc1KOp1'], # 10th
'ABprpapp':['241215plc1KOp1','241215plc1KOp2'], # 10th

# 'ABprpapp':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8']
# 'ABplpapp':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8']

# 1th
# 'ABp':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8'],
# 'ABp':['241205plc1KOp2'],

# 2th 2 cells
# 'ABalp':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8']
# 'ABalp':['241206plc1KOp1'],
# 'ABara':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8']
# 'ABara':['241206plc1KOp1'],

# 3th
# 'ABplaaa':['WT_Sample1','WT_Sample2','WT_Sample3','WT_Sample4','WT_Sample5','WT_Sample6','WT_Sample7','WT_Sample8']
#       'ABplaaa':['241203plc1KOp3'],
                       }

suffixs=['a','p']
# suffixs=['l','r']

for receptor_cell, embryo_list in embryo_name_cell_pair.items():
    for embryo_name_this in embryo_list:
        pd_saving_vavb = pd.DataFrame(columns=['Embryo Name', 'Time Point', 'Va/Vb'])

        pd_volume_path=os.path.join(ligand_embryos_root, embryo_name_this, '{}_volume.csv'.format(embryo_name_this))
        pd_volume=pd.read_csv(pd_volume_path,header=0)
        if receptor_cell+suffixs[0] not in pd_volume.columns:
            print('NOT THE CELL,',receptor_cell+suffixs[0])
            continue
        a_volumes=pd_volume[receptor_cell+suffixs[0]]
        p_volumes = pd_volume[receptor_cell + suffixs[1]]

        for index, value in a_volumes.items():
            if value>3.6 and index in p_volumes.index:
                other_value=p_volumes.loc[index]
                if other_value>3.6:
                    variation_a_b=value/other_value

                    pd_saving_vavb.loc[len(pd_saving_vavb)]=[embryo_name_this,index,variation_a_b]
        pd_saving_vavb.to_csv(os.path.join(saving_path,'{}_{}.csv'.format(receptor_cell,embryo_name_this)))




