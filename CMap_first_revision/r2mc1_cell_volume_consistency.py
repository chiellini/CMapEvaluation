import os.path
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from utils.image_io import nib_load


def validation_single_tp_cell():
    # seg_cell_path_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification\filtered_seg_volumes'
    # gt_cell_path_root = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification\match_gt_cell_volumes'
    # name_dictionary_file_path = r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\Tables\name_dictionary.csv"
    # label_name_dict = pd.read_csv(name_dictionary_file_path, index_col=0).to_dict()['0']
    #
    # pd_gt_seg_cell_volume = pd.DataFrame(
    #     columns=['Embryo Name', 'Time Point', 'Cell Name', 'GT Cell Volume', 'Segmented Cell Volume',
    #              'Variation Ratio', 'Cell Number'])
    # resolution_per_voxel = 0.18 ** 3
    #
    # emb_name_tp_2_cell_number = {('WT_Sample1', 90): '~100', ('WT_Sample1', 123): '~200', ('WT_Sample1', 132): '~300',
    #                              ('WT_Sample1', 166): '~400', ('WT_Sample1', 178): '~500', ('WT_Sample1', 185): '~550',
    #                              ('WT_Sample7', 78): '~100', ('WT_Sample7', 114): '~200', ('WT_Sample7', 123): '~300',
    #                              ('WT_Sample7', 157): '~400', ('WT_Sample7', 172): '~500', ('WT_Sample7', 181): '~550'
    #                              }
    #
    # gt_cell_volume_paths = sorted(glob.glob(os.path.join(gt_cell_path_root, '*.nii.gz')))
    # seg_cell_volume_paths = sorted(glob.glob(os.path.join(seg_cell_path_root, '*.nii.gz')))
    #
    # for idx, gt_cell_volume_path in enumerate(gt_cell_volume_paths):
    #     emb_name, tp = os.path.basename(gt_cell_volume_path).split('_')[1:3]
    #
    #     this_gt_volume_arr = nib_load(gt_cell_volume_path)
    #     this_seg_volume_arr = nib_load(seg_cell_volume_paths[idx])
    #     cell_list = list(np.unique(this_gt_volume_arr)[1:])
    #     for cell_number in cell_list:
    #         cell_name = label_name_dict[cell_number]
    #         # print(emb_name, tp, cell_name)
    #
    #         voxel_number_this = sum(this_seg_volume_arr[this_seg_volume_arr == cell_number])
    #         if voxel_number_this == 0:
    #             continue
    #         seg_this_cell_volume = voxel_number_this * resolution_per_voxel
    #         # pd_seg_cell_volume.loc[len(pd_seg_cell_volume.index)] = [emb_name, tp, cell_name, seg_this_cell_volume]
    #
    #         gt_this_cell_volume = sum(this_gt_volume_arr[this_gt_volume_arr == cell_number]) * resolution_per_voxel
    #         variation_ratio = abs(seg_this_cell_volume - gt_this_cell_volume) / gt_this_cell_volume
    #         pd_gt_seg_cell_volume.loc[len(pd_gt_seg_cell_volume.index)] = ['WT_' + emb_name, tp, cell_name,
    #                                                                        gt_this_cell_volume,
    #                                                                        seg_this_cell_volume, variation_ratio,
    #                                                                        emb_name_tp_2_cell_number[
    #                                                                            ('WT_' + emb_name, int(tp))]]
    #
    # pd_gt_seg_cell_volume.to_csv(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
    #     'gt_seg_cell.csv'))

    # ================================================================================================
    plt.figure(figsize=(5, 5))

    pd_gt_seg_cell_volume = pd.read_csv(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
        'gt_seg_cell.csv'))

    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    mean_variation = pd_gt_seg_cell_volume['Variation Ratio'].mean()
    print(mean_variation)
    plt.plot([mean_variation, mean_variation], [0.96, 1.1], linestyle='--', color='black', zorder=1)

    sns.histplot(data=pd_gt_seg_cell_volume, x='Variation Ratio', binwidth=.1, stat='probability',zorder=2)

    plt.xticks(fontsize=20)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=20)


    plt.xlabel('Relative variation\nof cell volume', size=20)
    plt.ylabel('Fraction', size=20)
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
        'figr1_variation.pdf'), format='pdf')

    plt.cla()
    # =============================================================================================
    max_value = max(
        list(pd_gt_seg_cell_volume['GT Cell Volume']) + list(pd_gt_seg_cell_volume['Segmented Cell Volume']))
    min_value = min(
        list(pd_gt_seg_cell_volume['GT Cell Volume']) + list(pd_gt_seg_cell_volume['Segmented Cell Volume']))

    plt.plot([0, max_value+66666], [0, max_value+66666], linestyle='--', color='black',zorder=1)

    sns.scatterplot(data=pd_gt_seg_cell_volume, x='GT Cell Volume', y='Segmented Cell Volume', hue='Cell number',zorder=2)
    plt.xlim(min_value - 16, max_value + 66666)  # Set x-axis range from 0 to 5
    plt.ylim(min_value - 16, max_value + 66666)  # Set y-axis range from 0 to 50
    plt.xscale('log')
    plt.yscale('log')
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.xlabel("Cell volume\nin manual annotation ($\mathrm{\mu} \mathrm{m}^3$)", size=20)
    plt.ylabel('Cell volume in $CMap$ ($\mathrm{\mu} \mathrm{m}^3$)', size=20)
    # plt.show()
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
        'figr1.pdf'), format='pdf')
    plt.show()


if __name__ == "__main__":
    validation_single_tp_cell()
