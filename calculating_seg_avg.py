import pandas as pd
import numpy as np

# file_name_evaluating = 'UNETR(without_nucleus)_hausdorff_distance.csv'
# y_column_name='HausdorffDistance95'
# small_threshold=0
# big_threshold=1000

# y_column_name='Jaccard'
# small_threshold=0.01
# big_threshold=0.99



# ========================================
file_name_evaluating = 'CellPose3D_hausdorff_distance_j.csv'
y_column_name='HausdorffDistance95'
small_threshold=0
big_threshold=1000

# y_column_name='JaccardIndex'
# small_threshold=0.01
# big_threshold=0.99


evaluation_file_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\07 ISBI2024 Deadline 2023Nov10\Comparison cell wise\{}'.format(
    file_name_evaluating)

score_this_data = pd.read_csv(evaluation_file_path)
print(score_this_data[y_column_name])

score_this_data = score_this_data[score_this_data[y_column_name] > small_threshold]
score_this_data = score_this_data[score_this_data[y_column_name] < big_threshold]

print(np.mean(score_this_data[y_column_name]))
