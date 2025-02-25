import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from utils.metrics import get_fast_aji, get_dice_1, get_dice_2, get_size_dice_and_iou
from utils.image_io import nib_load


def iou_dicescore_evaluate(args=None):
    gt_folder = args["gt_folder"]
    pred_folder = args["pred_folder"]
    # base_data = os.path.basename(gt_folder)
    print("Evaluation result of {} , method name {}".format(pred_folder, args['method_name']))

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*_uni.nii.gz")))
    assert len(pred_files) == len(gt_files), "#gt_files != #pred_files"

    embryo_using_pd_names = []
    all_ious = []
    all_dices = []
    all_cell_label = []
    all_cell_sizes = []

    embryo_ious = []  # average iou
    embryo_dices = []
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), desc=f"Evaltaing {pred_folder}", total=len(gt_files)):
        gt = nib_load(gt_file).astype(np.int16)
        pred = nib_load(pred_file).astype(np.int16)

        cells_label, cells_sizes, ious, dices = get_size_dice_and_iou(gt, pred)
        embryo_using_pd_names += [os.path.basename(pred_file).split(".")[0]] * len(cells_sizes)
        all_cell_label += cells_label
        all_cell_sizes += cells_sizes
        all_dices += dices
        all_ious += ious

        embryo_ious.append(sum(ious) / len(ious))
        embryo_dices.append(sum(dices) / len(dices))

    pd_each_cell_score = pd.DataFrame(
        data={"EmbryoName": embryo_using_pd_names, 'CellLabel': all_cell_label, "CellSize": all_cell_sizes,
              'IoU': all_ious, "DiceScore": all_dices})
    # save_file = os.path.join(args["save_folder"], base_data + "_" + os.path.basename(os.path.basename(gt_folder)) + "_score.csv")

    embryo_names = [os.path.basename(basename_path).split('.')[0] for basename_path in pred_files]
    pd_embryo_avg_score = pd.DataFrame(data={'EmbryoName': embryo_names, 'IoU': embryo_ious, 'DiceScore': embryo_dices})

    save_file = os.path.join(args["save_folder"], args['method_name'] + "_score.csv")
    if os.path.isfile(save_file):
        open(save_file, "w").close()
    pd_each_cell_score.to_csv(save_file, index=False)
    save_file = os.path.join(args["save_folder"], args['method_name'] + "_evaluation.csv")
    pd_embryo_avg_score.to_csv(save_file, index=False)


import SimpleITK as sitk
import seg_metrics.seg_metrics as sg


def four_metrics_evaluate(args=None):
    gt_folder = args["gt_folder"]
    pred_folder = args["pred_folder"]
    base_data = os.path.basename(gt_folder)
    print("Evaluation hausdorff distance of {} , method name {}".format(pred_folder, args['method_name']))

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*_uni.nii.gz")))
    print(gt_files, pred_files)
    assert len(pred_files) == len(gt_files), "#gt_files != #pred_files"
    embryo_wise_effective_mean_surface_dis_list = {}
    embryo_wise_effective_hausdorff_distance95_list = {}
    embryo_wise_effective_dice_score_list = {}
    embryo_wise_effective_jaccard_score_list = {}

    embryo_wise_mean_surface_dis_list = {}
    # embryo_wise_hausdorff_distance95_list = {}
    # embryo_wise_dice_score_list = {}
    # embryo_wise_jaccard_score_list = {}

    all_embryo_pred_cell_num_dict = {}
    all_embryo_gt_cell_num_dict = {}
    all_embryo_difference_list=[]

    for gt_file_name in gt_files:
        embryo_tp = '_'.join(os.path.basename(gt_file_name).split('_')[:2])
        embryo_wise_effective_mean_surface_dis_list[embryo_tp] = []
        embryo_wise_effective_hausdorff_distance95_list[embryo_tp] = []
        embryo_wise_effective_dice_score_list[embryo_tp] = []
        embryo_wise_effective_jaccard_score_list[embryo_tp] = []
        embryo_wise_mean_surface_dis_list[embryo_tp] = []

    embryo_using_pd_names = []

    all_mean_surface_dis_list = []
    all_hausdorff_distance95_list = []
    all_dice_score_list = []
    all_jaccard_score_list = []

    all_effective_mean_surface_dis_list = []
    all_effective_hausdorff_distance95_list = []
    all_effective_dice_score_list = []
    all_effective_jaccard_score_list = []
    all_cell_label_list = []
    for gdth_fpath, pred_fpath in tqdm(zip(gt_files, pred_files), desc=f"Calculating Dice and Hausdorff Distance",
                                       total=len(gt_files)):
        # gt = nib_load(gt_file).astype(np.int16)
        # pred = nib_load(pred_file).astype(np.int16)

        # cells_label, hausdorff_distances_list = get_hausdorff_distance(gt_file, pred_file)
        # Read images and convert it to numpy array.
        gdth_img = sitk.ReadImage(gdth_fpath)
        gdth_np = sitk.GetArrayFromImage(gdth_img)

        pred_img = sitk.ReadImage(pred_fpath)
        pred_np = sitk.GetArrayFromImage(pred_img)  # note: image shape order: (z,y,x)

        spacing = np.array(list(reversed(pred_img.GetSpacing())))  # note: after reverseing,  spacing order =(z,y,x)

        # Downsampling images to save cpu and memory utility, otherwirse the Colab may raise out of memory error.
        # print(pred_np.shape)
        gdth_np = gdth_np[::2, ::2, ::2]
        pred_np = pred_np[::2, ::2, ::2]
        # print(pred_np.shape)

        # gdth_labels = np.unique(gdth_np)
        # print(f"ground truth labels: {gdth_labels}")
        t1 = time.time()
        print("Start calculating ...")  # It will cost about 40 seconds
        labels = list(np.unique(pred_np))[1:]
        # csv_file = 'metrics.csv'  # or None if do not want to save metrics to csv file
        metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                                   gdth_img=gdth_np,
                                   pred_img=pred_np,
                                   # csv_file=csv_file,  # save results to the csv_file
                                   spacing=spacing,  # assign spacing
                                   metrics=['dice','jaccard','msd', 'hd'])
        # - dice: Dice(F - 1)
        # - jaccard: Jaccard
        # - hd95: Hausdorff distance 95 % percentile
        # - msd: Mean(Average) surface distance

        t2 = time.time()
        print(f"It cost {t2 - t1:.2f} seconds to finish the calculation.")
        print(os.path.basename(gdth_fpath), os.path.basename(pred_fpath))

        print(metrics)  # a list of dict which store metrics, if given only one pair of images, list length equals to 1.
        # print(metrics[0])  # a dict of metrics
        df_metrics = pd.DataFrame(metrics[0])
        print("=======================================")
        print(df_metrics)  # Better shown by pd.DataFrame

        embryo_using_pd_names += [os.path.basename(pred_fpath).split(".")[0]] * len(df_metrics)
        all_cell_label_list += list(df_metrics['label'])
        # if df_metrics['dice']>0.1:
        embryo_tp = '_'.join(os.path.basename(gdth_fpath).split('_')[:2])

        pred_cell_num = len(np.unique(pred_np)) - 1
        gt_cell_num = len(np.unique(gdth_np)) - 1
        all_embryo_pred_cell_num_dict[embryo_tp] = pred_cell_num
        all_embryo_gt_cell_num_dict[embryo_tp] = gt_cell_num
        all_embryo_difference_list.append(abs(pred_cell_num-gt_cell_num))

        embryo_wise_mean_surface_dis_list[embryo_tp] += list(df_metrics['msd'])

        embryo_wise_effective_mean_surface_dis_list[embryo_tp] += list(df_metrics[df_metrics['msd'] < 30]['msd'])
        embryo_wise_effective_hausdorff_distance95_list[embryo_tp] += list(df_metrics[df_metrics['hd'] < 30]['hd'])
        embryo_wise_effective_dice_score_list[embryo_tp] += list(df_metrics[df_metrics['dice'] > 0]['dice'])
        embryo_wise_effective_jaccard_score_list[embryo_tp] += list(df_metrics[df_metrics['jaccard'] > 0]['jaccard'])

        all_effective_mean_surface_dis_list += list(df_metrics[df_metrics['msd'] < 30]['msd'])
        all_effective_hausdorff_distance95_list += list(df_metrics[df_metrics['hd'] < 30]['hd'])
        all_effective_dice_score_list += list(df_metrics[df_metrics['dice'] > 0]['dice'])
        all_effective_jaccard_score_list += list(df_metrics[df_metrics['jaccard'] > 0]['jaccard'])

        all_mean_surface_dis_list += list(df_metrics['msd'])
        all_hausdorff_distance95_list += list(df_metrics['hd'])
        all_dice_score_list += list(df_metrics['dice'])
        all_jaccard_score_list += list(df_metrics['jaccard'])

    pd_fourmetrics_cell_wise = pd.DataFrame(data={"EmbryoName": embryo_using_pd_names, 'CellLabel': all_cell_label_list,
                                                  "MeanSurfaceDistance": all_mean_surface_dis_list,
                                                  "HausdorffDistance95": all_hausdorff_distance95_list,
                                                  'DiceScore': all_dice_score_list,
                                                  'JaccardIndex': all_jaccard_score_list})
    save_file = os.path.join(args["save_folder"], args['method_name'] + "_cell_wise_evaluation.csv")
    pd_fourmetrics_cell_wise.to_csv(save_file, index=False)

    pd_fourmetrics_embryo_wise = pd.DataFrame(
        columns=["EmbryoName", "MeanSurfaceDistance", "HausdorffDistance95", 'DiceScore', 'JaccardIndex',
                 'CellLossRate'])
    for embryo_tp in embryo_wise_effective_mean_surface_dis_list.keys():
        pd_fourmetrics_embryo_wise.loc[len(pd_fourmetrics_embryo_wise)] = [embryo_tp,
                                                                           sum(
                                                                               embryo_wise_effective_mean_surface_dis_list[
                                                                                   embryo_tp]) / len(
                                                                               embryo_wise_effective_mean_surface_dis_list[
                                                                                   embryo_tp]),
                                                                           sum(
                                                                               embryo_wise_effective_hausdorff_distance95_list[
                                                                                   embryo_tp]) / len(
                                                                               embryo_wise_effective_hausdorff_distance95_list[
                                                                                   embryo_tp]),
                                                                           sum(
                                                                               embryo_wise_effective_dice_score_list[
                                                                                   embryo_tp]) / len(
                                                                               embryo_wise_effective_dice_score_list[
                                                                                   embryo_tp]),
                                                                           sum(
                                                                               embryo_wise_effective_jaccard_score_list[
                                                                                   embryo_tp]) / len(
                                                                               embryo_wise_effective_jaccard_score_list[
                                                                                   embryo_tp]),
                                                                           abs(all_embryo_gt_cell_num_dict[embryo_tp] -
                                                                               all_embryo_pred_cell_num_dict[embryo_tp]) /
                                                                           all_embryo_gt_cell_num_dict[embryo_tp]
                                                                           ]
    pd_fourmetrics_embryo_wise.loc[len(pd_fourmetrics_embryo_wise)] = ['AllCellAverage',
                                                                       sum(
                                                                           all_effective_mean_surface_dis_list) / len(
                                                                           all_effective_mean_surface_dis_list),
                                                                       sum(
                                                                           all_effective_hausdorff_distance95_list) / len(
                                                                           all_effective_hausdorff_distance95_list),
                                                                       sum(
                                                                           all_effective_dice_score_list) / len(
                                                                           all_effective_dice_score_list),
                                                                       sum(
                                                                           all_effective_jaccard_score_list) / len(
                                                                           all_effective_jaccard_score_list),
                                                                       sum(all_embryo_difference_list) / sum(all_embryo_gt_cell_num_dict.values())
                                                                       ]
    print(all_embryo_pred_cell_num_dict,all_embryo_gt_cell_num_dict)
    save_file = os.path.join(args["save_folder"], args['method_name'] + "_volume_wise_evaluation.csv")

    pd_fourmetrics_embryo_wise.to_csv(save_file, index=False)


if __name__ == "__main__":
    args = dict(
        gt_folder=r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\3DEvaluation\GT\SegCell",
        pred_folder=r"C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\06paper TUNETr TMI LSA NC\TUNETr dataset\3DEvaluation\CShaper_validating\SegCell",
        method_name='CShaper',
        save_folder="./Results/Comparison")

    four_metrics_evaluate(args)
    # iou_dicescore_evaluate(args)