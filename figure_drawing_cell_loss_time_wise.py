import os.path
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.image_io import nib_load

def generate_dataframe_for_drawing():
    all_information_dataframe = pd.DataFrame(
        columns=['EmbryoName', 'MethodName', 'TimePoint', 'CellName', 'Fate', 'Normal'])

    method_names = ['StarDist3D', 'CShaper++', 'CTransformer']

    embryo_names = ['170704plc1p1', '200113plc1p2']
    # embryo_max_times = [240, 255]

    name_dictionary_path = r'F:\packed membrane nucleus 3d niigz\name_dictionary_TUNETr.csv'
    label_name_dict = pd.read_csv(name_dictionary_path, index_col=0).to_dict()['0']
    # name_label_dict = {value: key for key, value in label_name_dict.items()}
    # lineage_names = ['AB', 'C', 'D', 'E', 'MS', 'P']

    fate_file_path = r'F:\packed membrane nucleus 3d niigz\CellFate.xls'
    cell_fate = pd.read_excel(fate_file_path, names=["Cell", "Fate"], converters={"Cell": str, "Fate": str}, header=None)
    cell_fate = cell_fate.applymap(lambda x: x[:-1])
    cell2fate = dict(zip(cell_fate.Cell, cell_fate.Fate))

    cell_nucleus_annotation_path = r'F:\packed membrane nucleus 3d niigz'
    cell_evaluating_middle_materials_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\ForTimelapseAnd2DEvaluation'
    # \170704plc1p1\losing
    for embryo_name in embryo_names:
        this_embryo_nucleus_annotations = sorted(
            glob.glob(os.path.join(cell_nucleus_annotation_path, embryo_name,'AnnotatedNuc', '*.nii.gz')))
        for tp_index, this_embryo_tp_file in enumerate(this_embryo_nucleus_annotations):
            this_annotated_nucleus = nib_load(this_embryo_tp_file)
            nuclei_list = np.unique(this_annotated_nucleus)[1:]
            for method_this in method_names:
                loss_cell_list = []
                this_method_losing_path = os.path.join(cell_evaluating_middle_materials_path,
                                                       '{}_unified\middle_materials'.format(method_this), embryo_name,
                                                       'losing')
                loss_csv_path = os.path.join(this_method_losing_path,
                                             '{}_{}_losing.csv'.format(embryo_name, str(tp_index + 1).zfill(3)))
                if os.path.exists(loss_csv_path):
                    loss_cell_list = pd.read_csv(loss_csv_path, header=0, index_col=0)['0'].to_list()
                    # print(loss_cell_list)

                for cell_label_this in nuclei_list:
                    cell_name_this = label_name_dict[cell_label_this]
                    cell_fate_this=cell2fate.get(cell_name_this,False)
                    if not cell_fate_this:
                        cell_fate_this=cell2fate.get(cell_name_this[:-1], False)
                    if not cell_fate_this:
                        cell_fate_this = cell2fate.get(cell_name_this[:-2], 'Unspecified')

                    if len(loss_cell_list)==0:
                        all_information_dataframe.loc[len(all_information_dataframe)] = [embryo_name, method_this,
                                                                                         tp_index + 1, cell_name_this,
                                                                                         cell_fate_this, 1]
                    elif cell_name_this not in loss_cell_list:
                        all_information_dataframe.loc[len(all_information_dataframe)] = [embryo_name, method_this,
                                                                                         tp_index + 1, cell_name_this,
                                                                                         cell_fate_this, 1]
                    else:
                        all_information_dataframe.loc[len(all_information_dataframe)] = [embryo_name, method_this,
                                                                                         tp_index + 1, cell_name_this,
                                                                                         cell_fate_this, 0]
    all_information_dataframe.to_csv('testing.csv')

# def compose_embryo_wise

def plot_figures_time_lapse():
    data_frame_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\CellLossForEvaluation_cell_wise.csv'
    all_information_dataframe=pd.read_csv(data_frame_path)

    hue_order_list = [
        'StarDist3D',
        'CShaper++',
        'CTransformer']
    hue_palette = {
        'StarDist3D': '#ffc400',
        'CShaper++': '#74fff8',
        'CTransformer': '#e8000b'}

    embryo_wise_data=pd.DataFrame(columns=['TimePoint','LossRate','Method','Sample'])
    embryo_1_information=all_information_dataframe.loc[all_information_dataframe['EmbryoName']=='170704plc1p1']
    max_time=240
    for tp in range(1,max_time+1):
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this=embryo_1_information.loc[(embryo_1_information['MethodName']==method_this)&(embryo_1_information['TimePoint']==tp)]['Normal'].to_list()
            # print(method_this, tp,loss_rate_this)
            embryo_wise_data.loc[len(embryo_wise_data)]=[tp,1-sum(loss_rate_this)/len(loss_rate_this),method_this,'WT_C_Sample2']

    embryo_2_information = all_information_dataframe.loc[all_information_dataframe['EmbryoName'] == '200113plc1p2']
    max_time = 255
    for tp in range(1, max_time + 1):
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this = embryo_2_information.loc[
                (embryo_2_information['MethodName'] == method_this) & (embryo_2_information['TimePoint'] == tp)][
                'Normal'].to_list()
            # print(method_this, tp,loss_rate_this)
            embryo_wise_data.loc[len(embryo_wise_data)] = [tp, 1 - sum(loss_rate_this) / len(loss_rate_this),
                                                           method_this, 'WT_Sample1']
    out=sns.lineplot(data=embryo_wise_data, x="TimePoint", y='LossRate', hue='Method',style='Sample',
                       errorbar=('ci', 99),
                       hue_order=hue_order_list, palette=hue_palette)
    # plt.show()
    # out.savefig(f'time_lapse_cell_loss.pdf', dpi=300)
    plt.xticks([0, 50, 100, 150, 200, 250], fontsize=16)

    plt.yticks(fontsize=16)

    plt.xlabel("Developmental Time Point", size=20)
    plt.ylabel('Cell Loss Rate', size=20)
    plt.savefig("time_lapse_cell_loss.pdf", format="pdf", dpi=300)
    plt.show()


def lineage_mapping_func(cell_name):
    lineage_names = ['AB', 'C', 'D', 'E', 'MS', 'P']
    if cell_name[:2] in lineage_names:
        return cell_name[:2]
    else:
        return cell_name[0]

def plot_figures_lineage_wise():
    data_frame_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\CellLossForEvaluation_cell_wise.csv'
    all_information_dataframe=pd.read_csv(data_frame_path)
    all_information_dataframe['LineageName']=all_information_dataframe['CellName'].apply(lineage_mapping_func)


    lineage_names = ['AB', 'C', 'D', 'E', 'MS', 'P']
    hue_order_list = [
        'StarDist3D',
        'CShaper++',
        'CTransformer']
    hue_palette = {
        'StarDist3D': '#ffc400',
        'CShaper++': '#74fff8',
        'CTransformer': '#e8000b'}

    embryo_wise_data=pd.DataFrame(columns=['Lineage','LossRate','Method','Sample'])
    embryo_1_information=all_information_dataframe.loc[all_information_dataframe['EmbryoName']=='170704plc1p1']
    # max_time=240
    for lineage_this in lineage_names:
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this=embryo_1_information.loc[(embryo_1_information['MethodName']==method_this)&(embryo_1_information['LineageName']==lineage_this)]['Normal'].to_list()
            print(method_this, lineage_this)
            embryo_wise_data.loc[len(embryo_wise_data)]=[lineage_this,1-sum(loss_rate_this)/len(loss_rate_this),method_this,'WT_C_Sample2']

    embryo_2_information = all_information_dataframe.loc[all_information_dataframe['EmbryoName'] == '200113plc1p2']
    # max_time = 255
    for lineage_this in lineage_names:
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this = embryo_2_information.loc[
                (embryo_2_information['MethodName'] == method_this) & (embryo_2_information['LineageName'] == lineage_this)][
                'Normal'].to_list()
            print(method_this, lineage_this)
            embryo_wise_data.loc[len(embryo_wise_data)] = [lineage_this, 1 - sum(loss_rate_this) / len(loss_rate_this),
                                                           method_this, 'WT_Sample1']
    out=sns.scatterplot(data=embryo_wise_data, x="Lineage", y='LossRate', hue='Method',style='Sample',s=99,
                       hue_order=hue_order_list, palette=hue_palette)
    # plt.show()
    # out.savefig(f'time_lapse_cell_loss.pdf', dpi=300)
    ax=plt.subplot()
    ax.set_xticklabels(lineage_names, fontsize=16)

    plt.yticks(fontsize=16)

    plt.xlabel("Lineage Name", size=20)
    plt.ylabel('Cell Loss Rate', size=20)
    plt.savefig("time_lapse_cell_loss.pdf", format="pdf", dpi=300)
    plt.show()

def plot_figures_fate_wise():
    data_frame_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\CellLossForEvaluation_cell_wise.csv'
    all_information_dataframe=pd.read_csv(data_frame_path)


    fate_list=['Neuron',
               'Skin',
               'Muscle',
               'Pharynx',
               'Intestine',
               'Germ Cell',

               'Death',
'Unspecified',
               'Other',

               ]
    hue_order_list = [
        'StarDist3D',
        'CShaper++',
        'CTransformer']
    hue_palette = {
        'StarDist3D': '#ffc400',
        'CShaper++': '#74fff8',
        'CTransformer': '#e8000b'}

    embryo_wise_data=pd.DataFrame(columns=['CellFate','LossRate','Method','Sample'])
    embryo_1_information=all_information_dataframe.loc[all_information_dataframe['EmbryoName']=='170704plc1p1']
    # max_time=240
    for fate_this in fate_list:
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this=embryo_1_information.loc[(embryo_1_information['MethodName']==method_this)&(embryo_1_information['Fate']==fate_this)]['Normal'].to_list()
            print(method_this, fate_this)
            embryo_wise_data.loc[len(embryo_wise_data)]=[fate_this,1-sum(loss_rate_this)/len(loss_rate_this),method_this,'WT_C_Sample2']

    embryo_2_information = all_information_dataframe.loc[all_information_dataframe['EmbryoName'] == '200113plc1p2']
    # max_time = 255
    for fate_this in fate_list:
        for method_this in hue_order_list:
            # dice_score_pd = result_dataframe.loc[(result_dataframe['RealEmbryoName'] == embryo_name_tmp) & (
            #             result_dataframe['Method'] == method_name_tmp)]
            loss_rate_this = embryo_2_information.loc[
                (embryo_2_information['MethodName'] == method_this) & (embryo_2_information['Fate'] == fate_this)][
                'Normal'].to_list()
            print(method_this, fate_this)
            embryo_wise_data.loc[len(embryo_wise_data)] = [fate_this, 1 - sum(loss_rate_this) / len(loss_rate_this),
                                                           method_this, 'WT_Sample1']
    out=sns.scatterplot(data=embryo_wise_data, x="CellFate", y='LossRate', hue='Method',style='Sample',s=88,
                       hue_order=hue_order_list, palette=hue_palette)
    # plt.show()
    # out.savefig(f'time_lapse_cell_loss.pdf', dpi=300)
    plt.xticks(fate_list, fontsize=16, rotation=60)

    plt.yticks(fontsize=16)

    plt.xlabel("Tissue Type", size=20)
    plt.ylabel('Cell Loss Rate', size=20)
    plt.savefig("time_lapse_cell_loss.pdf", format="pdf", dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_figures_lineage_wise()