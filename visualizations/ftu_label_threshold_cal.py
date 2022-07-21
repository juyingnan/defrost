import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import io
import math
import os
import numpy as np
import pandas as pd
import unet_pos_def
from utils import tools, mapping_functions
from os.path import exists
from sklearn import metrics

dimension_reduction_types = ['tsne', 'umap', 'svd']
dr_index = 0
dr_type = dimension_reduction_types[dr_index]

ftu_label_dict = {
    1: "FTU",
    0: "FTU edge",
    -1: "non-FTU",
}

color_label_dict = {
    1: "blue",
    0: "azure",
    -1: "red",
}

root_path = r'X:\temp/'
file_name_list = [file_name for file_name in os.listdir(root_path) if file_name.endswith(".mat")]

use_unet_structure_pos = True
if use_unet_structure_pos:
    cols = unet_pos_def.tom_1_unet_def["cols"]
    rows = unet_pos_def.tom_1_unet_def["rows"]
else:
    N = len(file_name_list)
    cols = 5
    rows = int(math.ceil(N / cols))

silhouette_score_lists = [[] for _ in range(len(file_name_list))]
calinski_score_lists = [[] for _ in range(len(file_name_list))]
davies_score_lists = [[] for _ in range(len(file_name_list))]

horizontal_spacing = 0.03
layer_names = [file_name.split('_')[1].split('.')[0] for file_name in file_name_list]
fig = make_subplots(
    rows=2, cols=2,
    # column_widths=[1.0, 0],
    # row_heights=[0.7, 0.2, 0.1],
    subplot_titles=["silhouette", "calinski", "davies"]
)

# get FTU and non-FTU label
final_mat = io.loadmat(root_path + 'colon_final2.mat')
final_output = final_mat.get('feature_matrix')

# samples
final_img_ids = final_mat.get('image_id')[0]
final_batch_ids = final_mat.get('batch_id')[0]
final_patch_ids = final_mat.get('patch_id')[0]
_, non_zero_rates = tools.get_is_ftu_label(final_output, need_rates=True)
for threshold in np.arange(0.01, 0.16, 0.01):
    for i in range(len(non_zero_rates)):
        rate = non_zero_rates[i]
        if rate < threshold + 0.0025 and rate > threshold -0.0025:
            print(f"FTU rate: {rate}\timage{final_img_ids[i]}_batch{final_batch_ids[i]}"
                  f"_model0_final2_feature{final_patch_ids[i]}")
            break


threshold_lists = np.arange(0.01, 0.16, 0.001)

for t in range(len(file_name_list)):
    # for t in range(3):
    file_name = file_name_list[t]

    has_mapping_data = True
    mapping_mat_file_path = root_path + rf'colon_{dr_type}\{file_name.replace(".", f"_{dr_type}.")}'

    # try to read existing tsne data first
    if has_mapping_data and exists(mapping_mat_file_path):
        digits = io.loadmat(mapping_mat_file_path)
        X_mapping = digits.get('feature_matrix')
        # print(f"Loaded {dr_type} mapping data from existing mat.", X_mapping.shape)

    # do tsne calculation and save the tsne mat
    else:
        mat_path = root_path + file_name
        digits = io.loadmat(mat_path)

        X = digits.get('feature_matrix')
        # print(X.shape)
        mapping_function = getattr(mapping_functions, f"cal_{dr_type}")
        X = digits.get('feature_matrix')
        X_mapping = mapping_function(X)

        # save tsne raw result to mat
        digits['feature_matrix'] = X_mapping
        io.savemat(mapping_mat_file_path, mdict=digits)

    X_norm = X_mapping # tools.get_norm(X_mapping)
    # plt.figure(figsize=(8, 8))

    # x_norm_df = pd.DataFrame()
    # x_norm_df["X"] = X_norm[:, 0]
    # x_norm_df["Y"] = X_norm[:, 1]
    # x_norm_df["FTU"] = [ftu_label_dict[label] for label in non_zero_labels]
    # x_norm_df["color"] = [color_label_dict[label] for label in non_zero_labels]
    # # print(x_norm_df)
    #
    # row = unet_pos_def.tom_1_unet_def[layer_names[t]][0] if use_unet_structure_pos else (t // cols + 1)
    # col = unet_pos_def.tom_1_unet_def[layer_names[t]][1] if use_unet_structure_pos else (t % cols + 1)

    # for label in ftu_label_dict:
    #     legend = ftu_label_dict[label]
    #     select_df = x_norm_df[x_norm_df['FTU'] == legend]
    #     fig.add_trace(go.Scatter(x=select_df["X"], y=select_df["Y"],
    #                              mode='markers',
    #                              marker=dict(
    #                                  color=select_df["color"],
    #                                  line_width=1),
    #                              opacity=0.8,
    #                              name=legend,
    #                              legendgroup=legend,
    #                              showlegend=True if t == 0 else False,
    #                              ),
    #                   row=row,
    #                   col=col,
    #                   )
    #     fig.add_annotation(xref='x domain',
    #                        yref='y domain',
    #                        x=0.99,
    #                        y=0.01,
    #                        text="{:.2f}".format(metrics.calinski_harabasz_score(X_norm, y)),
    #                        showarrow=False,
    #                        row=row,
    #                        col=col,
    #                        )

    # calculate cluster silhouette coefficient
    # print(file_name,
    #       metrics.silhouette_score(X_norm, y, metric='euclidean'),
    #       metrics.calinski_harabasz_score(X_norm, y),
    #       metrics.davies_bouldin_score(X_norm, y),
    #       )

    _, non_zero_rates = tools.get_is_ftu_label(final_output, need_rates=True)
    for threshold in threshold_lists:
        non_zero_labels = [1 if rate >= threshold else (0 if rate >= 1e-4 else -1)
                           for rate in non_zero_rates]
        y = non_zero_labels

        silhouette_score_lists[t].append(metrics.silhouette_score(X_norm, y, metric='euclidean'))
        calinski_score_lists[t].append(metrics.calinski_harabasz_score(X_norm, y))
        davies_score_lists[t].append(metrics.davies_bouldin_score(X_norm, y))

# fig.update_xaxes(range=[-0.09, 1.09], showticklabels=False)
# fig.update_yaxes(range=[-0.09, 1.09], showticklabels=False)

# fig.write_html(fr".\result\{file_name_list[0].split('_')[0]}_{dr_type}.html")

for t in range(len(silhouette_score_lists)):
    fig.add_trace(go.Scatter(x=threshold_lists, y=silhouette_score_lists[t],
                             mode='lines',
                             name=file_name_list[t]), col=1, row=1)
    fig.add_trace(go.Scatter(x=threshold_lists, y=calinski_score_lists[t],
                             mode='lines',
                             name=file_name_list[t]), col=2, row=1)
    fig.add_trace(go.Scatter(x=threshold_lists, y=davies_score_lists[t],
                             mode='lines',
                             name=file_name_list[t]), col=1, row=2)

fig.show()
