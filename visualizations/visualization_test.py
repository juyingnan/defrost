# reference: https://github.com/DmitryUlyanov/Multicore-TSNE

import matplotlib.pyplot as plt
from matplotlib import gridspec
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from scipy import io
import numpy as np
import math
import os

root_path = r'X:\temp/'
file_name_list = [file_name for file_name in os.listdir(root_path) if file_name.endswith(".mat")]

fig = plt.figure()
fig.tight_layout()

N = len(file_name_list)
cols = 5
rows = int(math.ceil(N / cols))
gs = gridspec.GridSpec(rows, cols)

# get FTU and non-FTU label
non_zero_labels = []
final_mat = io.loadmat(root_path + 'colon_final2.mat')
final_output = final_mat.get('feature_matrix')
for mask in final_output:
    non_zero_rate = len(np.nonzero(mask)[0]) / (mask.shape[1] * mask.shape[2])
    # print(non_zero_rate)
    if non_zero_rate > 0.05:
        non_zero_labels.append(1)
    else:
        non_zero_labels.append(0)


for t in range(len(file_name_list)):
    # for t in range(3):
    file_name = file_name_list[t]
    mat_path = root_path + file_name
    digits = io.loadmat(mat_path)

    X, y = digits.get('feature_matrix'), digits.get('image_id')[0]  # X: nxm: n=500//sample, m=12,10,71,400//feature
    y = non_zero_labels
    print(X.shape)
    # X = X[:, :, t:t + 1]
    X = X.reshape(X.shape[0], -1)
    n_samples, n_features = X.shape

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='random', random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(file_name, "\t",
          "After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter, X.shape[-1],
                                                                                          X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    ax = fig.add_subplot(gs[t])
    # plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        # plt.plot(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(int(y[i][0])))
        ax.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.tab20(int(y[i])),
                fontdict={'size': 8})
        ax.title.set_text(file_name.split('_')[1].split('.')[0])
        # ax.xticks([])
        # ax.yticks([])
# plt.title('colon')
plt.show()
