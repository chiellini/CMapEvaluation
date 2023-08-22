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
    base_data = os.path.basename(gt_folder)
    print("Evaluation result of {} , method name {}".format(pred_folder,args['method_name'] ))

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*_G.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*_uni.nii.gz")))
    assert len(pred_files) == len(gt_files), "#gt_files != #pred_files"

    embryo_using_pd_names = []
    all_ious=[]
    all_dices = []
    all_cell_label=[]
    all_cell_sizes = []

    embryo_ious=[] # average iou
    embryo_dices=[]
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), desc=f"Evaltaing {pred_folder}", total=len(gt_files)):
        gt = nib_load(gt_file).astype(np.int16)
        pred = nib_load(pred_file).astype(np.int16)

        cells_label,cells_sizes, ious, dices = get_size_dice_and_iou(gt, pred)
        embryo_using_pd_names += [os.path.basename(pred_file).split(".")[0]] * len(cells_sizes)
        all_cell_label+=cells_label
        all_cell_sizes += cells_sizes
        all_dices += dices
        all_ious+=ious

        embryo_ious.append(sum(ious)/len(ious))
        embryo_dices.append(sum(dices)/len(dices))


    pd_each_cell_score = pd.DataFrame(data={"EmbryoName": embryo_using_pd_names,'CellLabel':all_cell_label, "CellSize": all_cell_sizes, 'IoU':all_ious,"DiceScore": all_dices})
    # save_file = os.path.join(args["save_folder"], base_data + "_" + os.path.basename(os.path.basename(gt_folder)) + "_score.csv")

    embryo_names=[os.path.basename(basename_path).split('.')[0] for basename_path in pred_files]
    pd_embryo_avg_score=pd.DataFrame(data={'EmbryoName':embryo_names,'IoU':embryo_ious,'DiceScore':embryo_dices})

    save_file = os.path.join(args["save_folder"], args['method_name'] + "_score.csv")
    if os.path.isfile(save_file):
        open(save_file, "w").close()
    pd_each_cell_score.to_csv(save_file, index=False)
    save_file = os.path.join(args["save_folder"], args['method_name'] + "_evaluation.csv")
    pd_embryo_avg_score.to_csv(save_file, index=False)

import SimpleITK as sitk
import seg_metrics.seg_metrics as sg

def hausdorff_distance_evaluate(args=None):
    gt_folder = args["gt_folder"]
    pred_folder = args["pred_folder"]
    base_data = os.path.basename(gt_folder)
    print("Evaluation hausdorff distance of {} , method name {}".format(pred_folder,args['method_name'] ))

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*_G.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*_uni.nii.gz")))
    print(gt_files,pred_files)
    assert len(pred_files) == len(gt_files), "#gt_files != #pred_files"

    embryo_using_pd_names = []
    all_hausdorff_distance_list=[]
    all_hausdorff_distance95_list=[]
    all_dice_score_list=[]
    all_cell_label_list=[]
    for gdth_fpath, pred_fpath in tqdm(zip(gt_files, pred_files), desc=f"Calculating Hausdorff Distance {pred_folder}", total=len(gt_files)):
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
                                   metrics=['hd', 'hd95','dice'])
        t2 = time.time()
        print(f"It cost {t2 - t1:.2f} seconds to finish the calculation.")
        print(metrics)  # a list of dict which store metrics, if given only one pair of images, list length equals to 1.
        # print(metrics[0])  # a dict of metrics
        df_metrics = pd.DataFrame(metrics[0])
        print("=======================================")
        print(df_metrics)  # Better shown by pd.DataFrame

        embryo_using_pd_names += [os.path.basename(pred_fpath).split(".")[0]] * len(df_metrics)
        all_cell_label_list+=list(df_metrics['label'])
        all_hausdorff_distance_list+=list(df_metrics['hd'])
        all_hausdorff_distance95_list+=list(df_metrics['hd95'])
        all_dice_score_list+=list(df_metrics['dice'])




    pd_hausdorff_distance = pd.DataFrame(data={"EmbryoName": embryo_using_pd_names,'CellLabel':all_cell_label_list,
                                               "HausdorffDistance": all_hausdorff_distance_list,
                                               "HausdorffDistance95": all_hausdorff_distance95_list,
                                               'DiceScore':all_dice_score_list})
    save_file = os.path.join(args["save_folder"], args['method_name'] + "_hausdorff_distance.csv")
    pd_hausdorff_distance.to_csv(save_file, index=False)



if __name__ == "__main__":
    args = dict(gt_folder = r"F:\CMap_paper\CMapEvaluation\3D",
                pred_folder = r"F:\CMap_paper\CMapEvaluation\VNet-CShaper\niigz",
                method_name='VNet-CShaper',
                save_folder = "./Results/Comparison")

    hausdorff_distance_evaluate(args)