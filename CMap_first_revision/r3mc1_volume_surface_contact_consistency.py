import os.path
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from utils.image_io import nib_load


def spheredt_cmap_validation_volume_surface_cell():
    # cmap_cell_volume_surface_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\cmap\average_volume_surface.csv'
    # spheredt_cell_volume_surface_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\spheresDT\average_volume_surface_spheredt.csv'
    #
    # pd_cmap_cell_volume_surface=pd.read_csv(cmap_cell_volume_surface_path)
    # pd_spheredt_cell_volume_surface=pd.read_csv(spheredt_cell_volume_surface_path)
    #
    # pd_two_dataset = pd.DataFrame(columns=['cell name','spheredt volume', 'cmap volume', 'spheredt surface','cmap surface','volume variance ratio','surface variance ratio'])
    #
    # # cell_name_list=set(pd_cmap_cell_volume_surface['cell name'])
    #
    # for idx_tem in pd_spheredt_cell_volume_surface.index:
    #     spheredt_series=pd_spheredt_cell_volume_surface.loc[idx_tem]
    #     cell_name_this=spheredt_series['cell name']
    #     spheredt_volume_this=spheredt_series['average volume']
    #     spheredt_surface_this=spheredt_series['average surface area']
    #
    #     this_cmap_series=pd_cmap_cell_volume_surface.loc[pd_cmap_cell_volume_surface['cell name']==cell_name_this]
    #     cmap_volume_this=this_cmap_series.iloc[0,2]
    #     cmap_surface_this=this_cmap_series.iloc[0,3]
    #
    #     volume_ratio=abs(cmap_volume_this-spheredt_volume_this)/cmap_volume_this
    #     surface_ratio=abs(cmap_surface_this-spheredt_surface_this)/cmap_surface_this
    #
    #     pd_two_dataset.loc[len(pd_two_dataset)]=[cell_name_this,spheredt_volume_this,cmap_volume_this,
    #                                              spheredt_surface_this,cmap_surface_this,
    #                                              volume_ratio,surface_ratio]
    #
    #
    # pd_two_dataset.to_csv(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'volume_surface_spheredt_cmap.csv'))

    # =============================================================================================
    plt.figure(figsize=(5, 5))
    pd_two_dataset = pd.read_csv(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'volume_surface_spheredt_cmap.csv'))

    plt.xlim(0, 1)
    plt.ylim(0, 0.8)
    mean_variation = pd_two_dataset['volume variance ratio'].mean()
    print(mean_variation)
    plt.plot([mean_variation, mean_variation], [0.73, 1], linestyle='--', color='black', zorder=1)
    sns.histplot(data=pd_two_dataset, x='volume variance ratio', binwidth=.1, stat='probability',zorder=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Relative variation\nof cell volume', size=20)
    plt.ylabel('Fraction', size=20)

    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'spheredt_cmap_volume_variance.pdf'), format='pdf')
    plt.cla()
    # -------------------------------

    max_value = max(
        list(pd_two_dataset['spheredt volume']) + list(pd_two_dataset['cmap volume']))
    min_value = min(
        list(pd_two_dataset['spheredt volume']) + list(pd_two_dataset['cmap volume']))

    plt.plot([0, max_value+66], [0, max_value+66], linestyle='--', color='black')

    sns.scatterplot(data=pd_two_dataset, x='cmap volume', y='spheredt volume')
    plt.xlim(min_value - 66, max_value+66)  # Set x-axis range from 0 to 5
    plt.ylim(min_value - 66, max_value+66)  # Set y-axis range from 0 to 50
    plt.xscale('log')
    plt.yscale('log')
    # ax.set_aspect('equal', adjustable='box')
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.xlabel("Cell volume in $CMap$ ($\mathrm{\mu} \mathrm{m}^3$)", size=20)
    plt.ylabel('Cell volume in\n$spheresDT/Mpacts-PiCS$ ($\mathrm{\mu} \mathrm{m}^3$)', size=20)
    # plt.show()
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'spheredt_cmap_volume.pdf'), format='pdf')
    plt.show()

    # =============================================================================================
    # plt.figure(figsize=(5, 5))
    # pd_two_dataset = pd.read_csv(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'volume_surface_spheredt_cmap.csv'))
    # plt.xlim(0, 1)
    # sns.histplot(data=pd_two_dataset, x='surface variance ratio', binwidth=.1)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    # plt.xlabel('Relative variation of cell surface', size=16)
    # plt.ylabel('Count', size=16)
    # ax = plt.gca()
    # # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    # plt.savefig(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'spheredt_cmap_surface_variance.pdf'), format='pdf')
    # plt.cla()
    # # -------------------------------
    # max_value = max(
    #     list(pd_two_dataset['spheredt volume']) + list(pd_two_dataset['cmap volume']))
    #
    # plt.plot([0, max_value], [0, max_value], linestyle='--', color='black')
    #
    # sns.scatterplot(data=pd_two_dataset, x='cmap surface', y='spheredt surface')
    #
    # plt.xscale('log')
    # plt.yscale('log')
    # # ax = plt.gca()
    # # use sci demical style
    # # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    #
    # plt.xticks(fontsize=16)
    #
    # plt.yticks(fontsize=16)
    #
    # plt.xlabel("Cell surface in SpheresDT", size=16)
    # plt.ylabel('Cell surface in $CMap$', size=16)
    # # plt.show()
    # plt.savefig(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'spheredt_cmap_surface.pdf'), format='pdf')
    # plt.show()


def spheredt_cmap_validation_cell_cell_contact():
    # cmap_cell_contact_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\cmap\average_contact.csv'
    # spheredt_cell_contact_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\spheresDT\average_contact.csv'
    #
    # pd_cmap_cell_contact = pd.read_csv(cmap_cell_contact_path, index_col=1)
    # pd_spheredt_cell_contact = pd.read_csv(spheredt_cell_contact_path, index_col=1)
    #
    # pd_two_dataset = pd.DataFrame(
    #     columns=['cell-cell contact pair', 'spheredt area', 'cmap area', 'area variance ratio'])
    #
    # # cell_name_list=set(pd_cmap_cell_volume_surface['cell name'])
    #
    # for cell_cell_contact_this in pd_spheredt_cell_contact.index:
    #     spheredt_series = pd_spheredt_cell_contact.loc[cell_cell_contact_this]
    #     spheredt_contact_this = spheredt_series['average area']
    #
    #     if cell_cell_contact_this in pd_cmap_cell_contact.index:
    #         this_cmap_series = pd_cmap_cell_contact.loc[cell_cell_contact_this]
    #         cmap_contact_area_this = this_cmap_series.iloc[1]
    #     else:
    #         cell1,cell2=cell_cell_contact_this.split(', ')
    #         order_changed_cell_cell_contact_this='('+cell2[:-1]+', '+cell1[1:]+')'
    #         if order_changed_cell_cell_contact_this in pd_cmap_cell_contact.index:
    #             this_cmap_series = pd_cmap_cell_contact.loc[order_changed_cell_cell_contact_this]
    #             cmap_contact_area_this = this_cmap_series.iloc[1]
    #         else:
    #             continue
    #
    #     volume_ratio = abs(spheredt_contact_this - cmap_contact_area_this) / cmap_contact_area_this
    #
    #     pd_two_dataset.loc[len(pd_two_dataset)] = [cell_cell_contact_this, spheredt_contact_this,
    #                                                cmap_contact_area_this, volume_ratio]
    #
    # pd_two_dataset.to_csv(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'contact_spheredt_cmap.csv'))

    # =============================================================================================
    plt.figure(figsize=(5, 5))
    pd_two_dataset = pd.read_csv(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'contact_spheredt_cmap.csv'))
    plt.xlim(0, 1)
    sns.histplot(data=pd_two_dataset, x='area variance ratio', binwidth=.1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel('Relative variation of cell volume', size=16)
    plt.ylabel('Count', size=16)
    ax = plt.gca()
    # use sci demical style
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'spheredt_cmap_contact_variance.pdf'), format='pdf')
    plt.cla()
    # -------------------------------
    max_value = max(
        list(pd_two_dataset['spheredt area']) + list(pd_two_dataset['cmap area']))

    plt.plot([0, max_value], [0, max_value], linestyle='--', color='black')

    sns.scatterplot(data=pd_two_dataset, x='cmap area', y='spheredt area')

    plt.xscale('log')
    plt.yscale('log')
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    plt.xlabel("Cell volume in $CMap$", size=16)
    plt.ylabel('Cell volume in SpheresDT', size=16)
    # plt.show()
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'spheredt_cmap_contact.pdf'), format='pdf')
    plt.show()

def bcoms2_cmap_validation_volume_surface_cell():
    # cmap_cell_volume_surface_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\cmap\average_volume_surface.csv'
    # bcoms2_cell_volume_surface_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1\bcoms2\average_volume_surface_bcoms2.csv'
    #
    # pd_cmap_cell_volume_surface=pd.read_csv(cmap_cell_volume_surface_path)
    # pd_bcmos2_cell_volume_surface=pd.read_csv(bcoms2_cell_volume_surface_path)
    #
    # pd_two_dataset = pd.DataFrame(columns=['cell name','bcoms2 volume', 'cmap volume','volume variance ratio'])
    #
    # # cell_name_list=set(pd_cmap_cell_volume_surface['cell name'])
    #
    # for idx_tem in pd_bcmos2_cell_volume_surface.index:
    #     bcoms2_series=pd_bcmos2_cell_volume_surface.loc[idx_tem]
    #     cell_name_this=bcoms2_series['cell name']
    #     spheredt_volume_this=bcoms2_series['average volume']
    #     # spheredt_surface_this=spheredt_series['average surface area']
    #
    #     this_cmap_series=pd_cmap_cell_volume_surface.loc[pd_cmap_cell_volume_surface['cell name']==cell_name_this]
    #     cmap_volume_this=this_cmap_series.iloc[0,2]
    #     # cmap_surface_this=this_cmap_series.iloc[0,3]
    #
    #     volume_ratio=abs(cmap_volume_this-spheredt_volume_this)/cmap_volume_this
    #     # surface_ratio=abs(cmap_surface_this-spheredt_surface_this)/cmap_surface_this
    #
    #     pd_two_dataset.loc[len(pd_two_dataset)]=[cell_name_this,spheredt_volume_this,cmap_volume_this,
    #                                              # spheredt_surface_this,cmap_surface_this,
    #                                              volume_ratio]
    #
    #
    # pd_two_dataset.to_csv(os.path.join(
    #     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
    #     'volume_surface_bcoms2_cmap.csv'))

    # =============================================================================================
    plt.figure(figsize=(5, 5))
    pd_two_dataset = pd.read_csv(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'volume_surface_bcoms2_cmap.csv'))
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)

    sns.histplot(data=pd_two_dataset, x='volume variance ratio', binwidth=.1,stat='probability')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    mean_variation=pd_two_dataset['volume variance ratio'].mean()
    print(mean_variation)
    plt.plot([mean_variation, mean_variation], [0.372, 1], linestyle='--', color='black')

    plt.xlabel('Relative variation\nof cell volume', size=20)
    plt.ylabel('Fraction', size=20)
    # plt.show()
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'bcoms2_cmap_volume_variance.pdf'), format='pdf')
    plt.cla()
    # -------------------------------
    max_value = max(
        list(pd_two_dataset['bcoms2 volume']) + list(pd_two_dataset['cmap volume']))

    min_value = min(
        list(pd_two_dataset['bcoms2 volume']) + list(pd_two_dataset['cmap volume']))

    plt.plot([0, max_value+966], [0, max_value+966], linestyle='--', color='black',zorder=1)

    ax=sns.scatterplot(data=pd_two_dataset, x='cmap volume', y='bcoms2 volume',zorder=2)

    plt.xlim(min_value-6, max_value+966)  # Set x-axis range from 0 to 5
    plt.ylim(min_value-6, max_value+966)  # Set y-axis range from 0 to 50
    plt.xscale('log')
    plt.yscale('log')
    # ax.set_aspect('equal', adjustable='box')
    # ax = plt.gca()
    # use sci demical style
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.xlabel("Cell volume in $CMap$ ($\mathrm{\mu} \mathrm{m}^3$)", size=20)
    plt.ylabel('Cell volume in $BCOMS2$ ($\mathrm{\mu} \mathrm{m}^3$)', size=20)
    # plt.show()
    plt.savefig(os.path.join(
        r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc1',
        'bcoms2_cmap_volume.pdf'), format='pdf')
    plt.show()


if __name__ == "__main__":
    spheredt_cmap_validation_volume_surface_cell()
