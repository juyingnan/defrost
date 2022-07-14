# reference: https://github.com/DmitryUlyanov/Multicore-TSNE

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from scipy import io
import math
import os
from utils import tools
from os.path import exists
import pandas as pd
import unet_pos_def

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
    rows=rows, cols=cols,
    # column_widths=[1.0, 0],
    # row_heights=[0.7, 0.2, 0.1],
    specs=unet_pos_def.tom_1_unet_def["spec"] if use_unet_structure_pos else [
        [{"type": "Scatter"} for i in range(cols)] for j in range(rows)
    ],
    print_grid=True,
    horizontal_spacing=horizontal_spacing, vertical_spacing=0.02, shared_xaxes=True,
    subplot_titles=unet_pos_def.tom_1_unet_def["names"]
)

# get FTU and non-FTU label
final_mat = io.loadmat(root_path + 'colon_final2.mat')
final_output = final_mat.get('feature_matrix')
non_zero_labels = tools.get_is_ftu_label(final_output)

for t in range(len(file_name_list)):
    # for t in range(3):
    file_name = file_name_list[t]

    has_existing_tsne_data = True
    tsne_mat_file_path = root_path + rf'colon_tsne\{file_name.replace(".", "_tsne.")}'

    # try to read existing tsne data first
    if has_existing_tsne_data and exists(tsne_mat_file_path):
        digits = io.loadmat(tsne_mat_file_path)
        X_tsne = digits.get('feature_matrix')
        print("loaded tsne data from existing mat.", X_tsne.shape)

    # do tsne calculation and save the tsne mat
    else:
        mat_path = root_path + file_name
        digits = io.loadmat(mat_path)

        X = digits.get('feature_matrix')
        print(X.shape)
        X = X.reshape(X.shape[0], -1)
        n_samples, n_features = X.shape

        '''t-SNE'''
        tsne = TSNE(n_components=2, init='random', random_state=42)
        X_tsne = tsne.fit_transform(X)
        print(file_name, "\t",
              "After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter, X.shape[-1],
                                                                                              X_tsne.shape[-1]))

        # save tsne raw result to mat
        digits['feature_matrix'] = X_tsne
        io.savemat(tsne_mat_file_path, mdict=digits)

    # y = digits.get('image_id')[0]
    y = non_zero_labels
    X_norm = tools.get_norm(X_tsne)
    # plt.figure(figsize=(8, 8))

    x_norm_df = pd.DataFrame()
    x_norm_df["X"] = X_norm[:, 0]
    x_norm_df["Y"] = X_norm[:, 1]
    x_norm_df["FTU"] = ["FTU" if label == 1 else "non-FTU" for label in non_zero_labels]
    x_norm_df["color"] = ["blue" if label == 1 else "red" for label in non_zero_labels]
    print(x_norm_df)

    for legend in ["FTU", "non-FTU"]:
        select_df = x_norm_df[x_norm_df['FTU'] == legend]
        fig.add_trace(go.Scatter(x=select_df["X"], y=select_df["Y"],
                                 mode='markers',
                                 marker=dict(
                                     color=select_df["color"],
                                     line_width=1),
                                 opacity=0.8,
                                 name=legend,
                                 legendgroup=legend,
                                 showlegend=True if t == 0 else False,
                                 ),
                      row=unet_pos_def.tom_1_unet_def[layer_names[t]][0] if use_unet_structure_pos else (t // cols + 1),
                      col=unet_pos_def.tom_1_unet_def[layer_names[t]][1] if use_unet_structure_pos else (t % cols + 1),
                      )

fig.update_xaxes(range=[-0.09, 1.09], showticklabels=False)
fig.update_yaxes(range=[-0.09, 1.09], showticklabels=False)
fig.write_html(fr".\result\{file_name_list[0].split('_')[0]}_tsne.html")
fig.show()
