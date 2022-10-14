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


def generate_one_line_df(df_list, filter):
    last_df = df_list[-1][df_list[-1]['FTU'] == filter]
    second_df = df_list[-2][df_list[-2]['FTU'] == filter]

    line_x = [None] * (len(last_df) * 2)
    line_y = [None] * (len(last_df) * 2)
    line_z = [None] * (len(last_df) * 2)
    line_x[::2] = last_df["X"]
    line_y[::2] = last_df["Y"]
    line_z[::2] = last_df["Z"]
    line_x[1::2] = second_df["X"]
    line_y[1::2] = second_df["Y"]
    line_z[1::2] = second_df["Z"]

    l_data = dict()
    l_data["x"] = line_x
    l_data["y"] = line_y
    l_data["z"] = line_z
    # v_color = ['red'] * len(vessel_x_list)
    # v_data["color"] = v_color
    l_df = pd.DataFrame(l_data)
    l_gap = (l_df.iloc[1::2]
             .assign(x=np.nan, y=np.nan)
             .rename(lambda x: x + .5))
    l_df_one = pd.concat([l_df, l_gap], sort=False).sort_index().reset_index(drop=True)
    l_df_one.loc[l_df_one.isnull().any(axis=1), :] = np.nan
    return l_df_one


dimension_reduction_types = ['tsne', 'umap', 'svd']
dr_index = 1
dr_type = dimension_reduction_types[dr_index]

ftu_label_dict = {
    1: "ship",
    0: "FTU edge",
    -1: "non-ship",
}

color_label_dict = {
    1: "blue",
    0: "gray",
    -1: "red",
}

root_path = r'X:\temp\neec/'
file_name_list = [file_name for file_name in os.listdir(root_path) if file_name.endswith(".mat")]

use_unet_structure_pos = True
if use_unet_structure_pos:
    cols = unet_pos_def.tom_1_unet_def["cols"]
    rows = unet_pos_def.tom_1_unet_def["rows"]
else:
    N = len(file_name_list)
    cols = 5
    rows = int(math.ceil(N / cols))

horizontal_spacing = 0.01
layer_names = [file_name.split('_')[1].split('.')[0] for file_name in file_name_list]
fig = make_subplots(
    rows=2, cols=3,
    column_widths=[1, 1, 1],
    row_heights=[0.75, 0.25],
    specs=[[{"type": "Scatter3d", }, {"type": "Scatter3d", }, {"type": "Scatter3d", }],
           [{"type": "Scatter3d", "colspan": 2}, None, None]
           ],
    print_grid=True,
    horizontal_spacing=horizontal_spacing, vertical_spacing=0.02, shared_xaxes=True,
    subplot_titles=["3D_test"]
)

# get FTU and non-FTU label
final_mat = io.loadmat(root_path + 'ship_final2.mat')
final_output = final_mat.get('feature_matrix')
non_zero_labels = tools.get_is_ftu_label(final_output)

x_scale = 0.25
z_scale = 0.25

layer_dict = {
    "colon_encoder": {
        "row": 1,
        "col": 1,
        "anchor": 5,
    },
    "colon_decoder": {
        "row": 1,
        "col": 2,
        "anchor": 5,
    },
    "colon_upsam": {
        "row": 1,
        "col": 3,
        "anchor": 5,
    },
}

scatter_legend_flag = True
line_legend_flag = True
pipeline_list = ["ship_encoder", "ship_decoder", "ship_upsam"]
for pipeline in pipeline_list:
    line_temp = []
    for t in range(len(file_name_list)):
        # for t in range(3):
        file_name = file_name_list[t]

        if file_name.startswith(pipeline):
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

            y = non_zero_labels
            X_norm = tools.get_norm(X_mapping)
            # plt.figure(figsize=(8, 8))

            x_norm_df = pd.DataFrame()
            anchor = layer_dict[pipeline]["anchor"]
            position = anchor - int(file_name[-5])
            x_norm_df["X"] = [i + position * x_scale for i in X_norm[:, 0]]
            x_norm_df["Y"] = X_norm[:, 1]
            x_norm_df["Z"] = [position * z_scale for i in X_norm[:, 1]]
            x_norm_df["FTU"] = [ftu_label_dict[label] for label in non_zero_labels]
            x_norm_df["color"] = [color_label_dict[label] for label in non_zero_labels]
            print(x_norm_df)
            line_temp.append(x_norm_df)

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
                                           showlegend=scatter_legend_flag,
                                           ),
                              row=layer_dict[pipeline]["row"],
                              col=layer_dict[pipeline]["col"],
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
                    row=layer_dict[pipeline]["row"],
                    col=layer_dict[pipeline]["col"],
                )

            if len(line_temp) > 1:
                # for i in range(len(line_temp[-1])):
                #     if x_norm_df["FTU"][i] == ftu_label_dict[0]:
                line_filter = 0
                for line_filter in ftu_label_dict:
                    legend = ftu_label_dict[line_filter]
                    line_one = generate_one_line_df(line_temp, ftu_label_dict[line_filter])
                    legend = ftu_label_dict[line_filter]

                    fig.add_trace(go.Scatter3d(x=line_one['x'],
                                               y=line_one['y'],
                                               z=line_one['z'],
                                               mode='lines',
                                               line=dict(
                                                   color=color_label_dict[line_filter],
                                                   width=2,
                                               ),
                                               opacity=0.1,
                                               name=legend + " line",
                                               legendgroup=legend + " line",
                                               showlegend=line_legend_flag,
                                               ),
                                  row=layer_dict[pipeline]["row"],
                                  col=layer_dict[pipeline]["col"],
                                  )
                line_legend_flag = False
            scatter_legend_flag = False

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
    },
    scene2={
        "zaxis": {"ticktext": ["Decoder 0", "Decoder 1", "Decoder 2", "Decoder 3", "Deoder 4"],
                  "tickvals": [(5 - i) * z_scale for i in range(5)]},
        "xaxis": {"visible": False, "showticklabels": False},
        "yaxis": {"visible": False, "showticklabels": False},
        # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    },
    scene3={
        "zaxis": {"ticktext": ["Upsample 1", "Upsample 2", "Upsample 3", "Upsample 4"],
                  "tickvals": [(4 - i) * z_scale for i in range(4)]},
        "xaxis": {"visible": False, "showticklabels": False},
        "yaxis": {"visible": False, "showticklabels": False},
        # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    },
)
fig['layout']['xaxis'].update(visible=False, showticklabels=False)
fig['layout']['yaxis'].update(visible=False, showticklabels=False)
# fig['layout']['zaxis2'].update(ticktext=["Decoder 0", "Decoder 1", "Decoder 2", "Decoder 3", "Decoder 4"])
# fig['layout']['zaxis2'].update(tickvals=[(5 - i) * z_scale for i in range(5)])
# fig.update_xaxes(visible=False, showticklabels=False)
# fig.update_yaxes(visible=False, showticklabels=False)
fig.write_html(fr".\result\{file_name_list[0].split('_')[0]}_{dr_type}_3d.html")
fig.show()
