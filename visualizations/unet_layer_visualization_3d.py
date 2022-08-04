import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import io
import math
import os
import pandas as pd
import numpy as np
import unet_pos_def
from utils import tools, mapping_functions
from os.path import exists
from sklearn import metrics

dimension_reduction_types = ['tsne', 'umap', 'svd']
dr_index = 1
dr_type = dimension_reduction_types[dr_index]

ftu_label_dict = {
    1: "FTU",
    0: "FTU edge",
    -1: "non-FTU",
}

color_label_dict = {
    1: "blue",
    0: "gray",
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

horizontal_spacing = 0.03
layer_names = [file_name.split('_')[1].split('.')[0] for file_name in file_name_list]
fig = make_subplots(
    rows=1, cols=1,
    # column_widths=[1.0, 0],
    # row_heights=[0.7, 0.2, 0.1],
    specs=[[{"type": "Scatter3d", }],
           ],
    print_grid=True,
    horizontal_spacing=horizontal_spacing, vertical_spacing=0.02, shared_xaxes=True,
    subplot_titles=["3D_test"]
)

# get FTU and non-FTU label
final_mat = io.loadmat(root_path + 'colon_final2.mat')
final_output = final_mat.get('feature_matrix')
non_zero_labels = tools.get_is_ftu_label(final_output)

line_temp = []
x_scale = 0.25
z_scale = 0.25

for t in range(len(file_name_list)):
    # for t in range(3):
    file_name = file_name_list[t]

    has_mapping_data = True
    mapping_mat_file_path = root_path + rf'colon_{dr_type}\{file_name.replace(".", f"_{dr_type}.")}'

    # try to read existing tsne data first
    if has_mapping_data and exists(mapping_mat_file_path):
        digits = io.loadmat(mapping_mat_file_path)
        X_mapping = digits.get('feature_matrix')
        print(f"Loaded {dr_type} mapping data from existing mat.", X_mapping.shape)

    # do tsne calculation and save the tsne mat
    else:
        mat_path = root_path + file_name
        digits = io.loadmat(mat_path)

        X = digits.get('feature_matrix')
        print(X.shape)
        mapping_function = getattr(mapping_functions, f"cal_{dr_type}")
        X = digits.get('feature_matrix')
        X_mapping = mapping_function(X)

        # save tsne raw result to mat
        digits['feature_matrix'] = X_mapping
        io.savemat(mapping_mat_file_path, mdict=digits)

    # y = digits.get('image_id')[0]
    y = non_zero_labels
    X_norm = tools.get_norm(X_mapping)
    # plt.figure(figsize=(8, 8))

    x_norm_df = pd.DataFrame()
    x_norm_df["X"] = X_norm[:, 0]
    x_norm_df["Y"] = X_norm[:, 1]
    if file_name.startswith("colon_encoder"):
        x_norm_df["X"] = [i + (5 - int(file_name[-5])) * x_scale for i in X_norm[:, 0]]
        x_norm_df["Y"] = X_norm[:, 1]
        x_norm_df["Z"] = [(5 - int(file_name[-5])) * z_scale for i in X_norm[:, 1]]
        x_norm_df["FTU"] = [ftu_label_dict[label] for label in non_zero_labels]
        x_norm_df["color"] = [color_label_dict[label] for label in non_zero_labels]
        print(x_norm_df)
        line_temp.append(x_norm_df)

    row = unet_pos_def.tom_1_unet_def[layer_names[t]][0] if use_unet_structure_pos else (t // cols + 1)
    col = unet_pos_def.tom_1_unet_def[layer_names[t]][1] if use_unet_structure_pos else (t % cols + 1)

    if file_name.startswith("colon_encoder"):
        for label in ftu_label_dict:
            legend = ftu_label_dict[label]
            select_df = x_norm_df[x_norm_df['FTU'] == legend]
            fig.add_trace(go.Scatter3d(x=select_df["X"], y=select_df["Y"], z=select_df["Z"],
                                       mode='markers',
                                       marker=dict(
                                           size=2,
                                           color=select_df["color"],
                                           line_width=1),
                                       opacity=0.5,
                                       name=legend,
                                       legendgroup=legend,
                                       showlegend=True if t == 0 else False,
                                       ),
                          row=1,
                          col=1,
                          )
            xi = np.linspace((5 - int(file_name[-5])) * x_scale, (5 - int(file_name[-5])) * x_scale + 1, 2)
            yi = np.linspace(0, 1, 2)
            zi = np.array([[(5 - int(file_name[-5])) * z_scale for i in xi] for j in xi])
            x_grid, y_grid = np.meshgrid(xi, yi)

            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=zi,
                colorscale=[[0, '#C2C5CC'], [1, '#C2C5CC']],
                opacity=0.05,
                showlegend=False,
                showscale=False
            ),
                row=1,
                col=1,
            )
        if len(line_temp) > 1:
            for i in range(len(line_temp[-1])):
                if x_norm_df["FTU"][i] == ftu_label_dict[0]:
                    fig.add_trace(go.Scatter3d(x=[line_temp[-1]["X"][i], line_temp[-2]["X"][i]],
                                               y=[line_temp[-1]["Y"][i], line_temp[-2]["Y"][i]],
                                               z=[line_temp[-1]["Z"][i], line_temp[-2]["Z"][i]],
                                               mode='lines',
                                               line=dict(
                                                   color=line_temp[-1]["color"][i],
                                                   width=2,
                                               ),
                                               opacity=0.1,
                                               showlegend=False,
                                               ),
                                  row=1,
                                  col=1,
                                  )

fig.update_xaxes(range=[-0.09, 1.09], showticklabels=False)
fig.update_yaxes(range=[-0.09, 1.09], showticklabels=False)
fig.update_layout(
    scene=dict(
        aspectmode='data',
    )
)
fig.update_layout(
    scene={
        "zaxis": {"ticktext": ["Encoder 0", "Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4"],
                  "tickvals": [(5 - i) * z_scale for i in range(5)]},
        "xaxis": {"visible": False, "showticklabels": False},
        "yaxis": {"visible": False, "showticklabels": False},
        # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    })
# fig.update_xaxes(visible=False, showticklabels=False)
# fig.update_yaxes(visible=False, showticklabels=False)
fig.write_html(fr".\result\{file_name_list[0].split('_')[0]}_{dr_type}_3d.html")
fig.show()
