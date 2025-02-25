import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# pd_volume_surface_irregularity = pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_some_precedent_cells.csv'))
#
# pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
#     pd_volume_surface_irregularity['Embryo Name'] != 'Average']
#
# pd_std_time_wise=pd.DataFrame(columns=['Lineage','Time','STD'])
#
# lineage=['E sublineage','MS sublineage']
#
# for time_this in set(pd_volume_surface_irregularity['Time']):
#     for lineage_this in lineage:
#         pd_tem_this=pd_volume_surface_irregularity.loc[(pd_volume_surface_irregularity['Time']==time_this)&(pd_volume_surface_irregularity['Precedent']==lineage_this)]
#         std_this=np.std(list(pd_tem_this['Irregularity']), ddof=1)
#         print(lineage_this,time_this,std_this)
#         pd_std_time_wise.loc[len(pd_std_time_wise)]=[lineage_this,time_this,std_this]
#
# pd_std_time_wise.to_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_std.csv'))
#
# # ======================================================================================
#
# pd_std_time_wise=pd.read_csv(os.path.join(
#     r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
#     'irregularity_of_std.csv'))
#
# sns.lineplot(data=pd_std_time_wise,x='Time',y='STD',hue='Lineage')
#
# plt.show()

# ======================================================================
pd_volume_surface_irregularity = pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_some_precedent_cells.csv'))
pd_volume_surface_irregularity = pd_volume_surface_irregularity.loc[
    pd_volume_surface_irregularity['Embryo Name'] == 'Average']

pd_std_time_wise=pd.read_csv(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\r3mc2',
    'irregularity_of_std.csv'))

lineage=['E sublineage','MS sublineage']

pd_irregularity_std=pd.DataFrame(columns=['Time','Lineage','Irregularity','STD'])

for time_this in set(pd_volume_surface_irregularity['Time']):
    for lineage_this in lineage:
        pd_tem_this_irregularity=pd_volume_surface_irregularity.loc[(pd_volume_surface_irregularity['Time']==time_this)&(pd_volume_surface_irregularity['Precedent']==lineage_this)]
        irregularity_this=pd_tem_this_irregularity.iloc[0,6]

        pd_tem_this_std=pd_std_time_wise.loc[(pd_std_time_wise['Time']==time_this)&(pd_std_time_wise['Lineage']==lineage_this)]
        std_this=pd_tem_this_std.iloc[0,3]

        pd_irregularity_std.loc[len(pd_irregularity_std)]=[time_this,lineage_this,irregularity_this,std_this]

ax=sns.scatterplot(data=pd_irregularity_std,x='Irregularity',y='STD',hue='Lineage',hue_order=['MS sublineage','E sublineage'],palette=['blue','red'],zorder=2)

# Access the legend and adjust font size
# legend = ax.get_legend()
# legend.set_title("")  # Remove legend title
# for text in legend.get_texts():
#     text.set_fontsize(16)  # Set font size for labels
# legend.get_frame().set_linewidth(0.5)  # Optional: Adjust legend border thickness

# Set the legend position to bottom-right
# plt.scatter(y=0.14,x=2.23,color='white')
plt.legend(
    # loc='lower right',  # Position legend at bottom-right
    # bbox_to_anchor=(1, 0),  # Fine-tune the position
    fontsize=16  # Set font size
)
# ==============================================================================

# Convert to numpy arrays
array1 = np.array(list(pd_irregularity_std['Irregularity']))
array2 = np.array(list(pd_irregularity_std['STD']))

# Pearson Correlation Coefficient
pearson_corr, _ = pearsonr(array1, array2)
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")

# Interpretation of Correlation Strength (based on Pearson)
def interpret_correlation(corr):
    if abs(corr) > 0.8:
        return "Strong correlation"
    elif abs(corr) > 0.5:
        return "Moderate correlation"
    elif abs(corr) > 0.3:
        return "Weak correlation"
    else:
        return "Very weak or no correlation"

# Interpret the results
print(f"Pearson Correlation Strength: {interpret_correlation(pearson_corr)}")


# ================================================================================
# Define the model function: y = e^(a * x)
def exponential_model(x, a,b):
    return a * x+b


# Fit the model to the data
params, covariance = curve_fit(exponential_model, array1, array2)

# Extract the fitted parameter 'a'
a_fitted = params[0]
b_fitted=params[1]
print(f"Fitted parameter a: {a_fitted} and covariance  ",covariance)

# Generate fitted y values
plotting_x=np.linspace(min(array1), max(array1)+0.016, 100)
y_fitted = exponential_model(plotting_x, a_fitted,b_fitted)
# plt.legend(loc='upper right', bbox_to_anchor=(1.29, 1),prop={'size': 16})

# Plot the original data and the fitted curve
plt.plot(plotting_x, y_fitted, label=f"Fitted curve: e^({a_fitted:.2f}x)", zorder=1,linestyle='--', color='black')

plt.text(2.66,0.026,  '$R$ = 0.4353',
         fontsize=20, color='black',
         ha='center', va='center',
         # bbox=dict(facecolor='lightyellow', alpha=0.5)
         )

# plt.yticks([0.02,0.06,0.1,0.14],fontsize=20)
plt.xticks([2.3,2.4,2.5,2.6,2.7],fontsize=20)

plt.yticks(fontsize=20)

plt.ylabel("Cell irregularity\nvariability (STD)", size=20)
plt.xlabel(r'Cell irregularity (MEAN)', size=20)

# plt.title(' Segmentation ComparisonHausdorff', size = 24 )


# out.savefig('text.eps', dpi=300)
# out.savefig('text.svg', dpi=300)
plt.savefig(os.path.join(
    r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\04paper CMap coroperation\A first revision\gt_cell_volume_verification',
    'figr3mc1.pdf'), dpi=300)

plt.show()


