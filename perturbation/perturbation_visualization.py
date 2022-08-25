import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import io
import pandas as pd
from utils import tools, mapping_functions
from os.path import exists

dimension_reduction_types = ['tsne', 'umap', ]  # 'svd']

mat_path = r'X:\temp\perturbation_1024\result.mat'
has_mapping_data = True

p_index = 8
horizontal_spacing = 0.025
vertital_spacing = 0.025

rep = len(dimension_reduction_types)

fig = make_subplots(
    rows=2, cols=rep,
    column_widths=[1 for _ in range(rep)],
    # row_heights=[0.75, 0.25],
    specs=[[{"type": "Scatter", } for _ in range(rep)],
           [{"type": "Scatter3D", } for _ in range(rep)]],
    # specs=[[{"type": "Scatter3d", } for _ in range(rep)], ],
    print_grid=True,
    horizontal_spacing=horizontal_spacing, vertical_spacing=0.02, shared_xaxes=True,
    subplot_titles=[f"{dr} {dimension}d" for dimension in [2, 3] for dr in dimension_reduction_types]
)

scatter_legend_flag = True
line_legend_flag = True
for dimension in [2, 3]:
    for i in range(rep):
        dr_type = dimension_reduction_types[i]
        mapping_mat_file_path = mat_path.replace('.mat', f'_{dr_type}_{dimension}d.mat')

        # try to read existing tsne data first
        if has_mapping_data and exists(mapping_mat_file_path):
            digits = io.loadmat(mapping_mat_file_path)
            X_mapping = digits.get('feature_matrix')
            print(f"Loaded {dr_type} mapping data from existing mat.", X_mapping.shape)

        # do tsne calculation and save the tsne mat
        else:
            digits = io.loadmat(mat_path)

            X = digits.get('feature_matrix')
            print(X.shape)
            mapping_function = getattr(mapping_functions, f"cal_{dr_type}")
            X = digits.get('feature_matrix')
            X_mapping = mapping_function(X, n=dimension)

            # save tsne raw result to mat
            digits['feature_matrix'] = X_mapping
            io.savemat(mapping_mat_file_path, mdict=digits)

        y = digits.get('image_id')
        y = [int(item) * p_index for item in y]
        X_norm = tools.get_norm(X_mapping)
        # plt.figure(figsize=(8, 8))

        x_norm_df = pd.DataFrame()
        x_norm_df["X"] = X_norm[:, 0]
        x_norm_df["Y"] = X_norm[:, 1]
        if dimension == 3:
            x_norm_df["Z"] = X_norm[:, 2]
        print(x_norm_df)

        if dimension == 3:
            fig.add_trace(go.Scatter3d(x=x_norm_df["X"], y=x_norm_df["Y"], z=x_norm_df["Z"],
                                       mode='markers',
                                       marker=dict(
                                           size=4,
                                           color=y,
                                           colorscale='blues',
                                           line_width=1),
                                       opacity=0.5,
                                       showlegend=False,
                                       hovertemplate='%{text}',
                                       text=y,
                                       ),

                          row=dimension - 1,
                          col=i + 1,
                          )
        else:
            fig.add_trace(go.Scatter(x=x_norm_df["X"], y=x_norm_df["Y"],
                                     mode='markers',
                                     marker=dict(
                                         size=10,
                                         color=y,
                                         colorscale='blues',
                                         line_width=1),
                                     opacity=0.5,
                                     showlegend=False,
                                     hovertemplate='%{text}',
                                     text=y,
                                     ),
                          row=dimension - 1,
                          col=i + 1,
                          )

fig.update_xaxes(range=[-0.02, 1.02], showticklabels=False)
fig.update_yaxes(range=[-0.02, 1.02], showticklabels=False)
fig.update_layout(
    scene=dict(
        aspectmode='data',
    )
)
# fig.update_layout(
#     scene={
#         "xaxis": {"visible": False, "showticklabels": False},
#         "yaxis": {"visible": False, "showticklabels": False},
#         # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
#         # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
#     },
#     scene2={
#         "xaxis": {"visible": False, "showticklabels": False},
#         "yaxis": {"visible": False, "showticklabels": False},
#         # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
#         # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
#     },
#     # scene3={
#     #     "xaxis": {"visible": False, "showticklabels": False},
#     #     "yaxis": {"visible": False, "showticklabels": False},
#     #     # 'camera_eye': {"x": 0, "y": -1, "z": 0.5},
#     #     # "aspectratio": {"x": 1, "y": 1, "z": 0.2}
#     # },
# )
# fig['layout']['xaxis'].update(visible=False, showticklabels=False)
# fig['layout']['yaxis'].update(visible=False, showticklabels=False)
fig.write_html(mat_path.replace('.mat', '.html'))
fig.show()
