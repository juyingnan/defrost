# from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from umap import UMAP
from skimage.transform import rescale

def cal_tsne(feature_mat):
    print("tsne input shape:", feature_mat.shape)
    X = feature_mat.reshape(feature_mat.shape[0], -1)
    # n_samples, n_features = X.shape

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='random', random_state=42)
    return tsne.fit_transform(X)


def cal_umap(feature_mat):
    print("umap input shape:", feature_mat.shape)
    X = feature_mat.reshape(feature_mat.shape[0], -1)
    # n_samples, n_features = X.shape

    '''UMAP'''
    umap_2d = UMAP(n_components=2, init='random', random_state=42)
    return umap_2d.fit_transform(X)


def cal_svd(feature_mat, axis_threshold=2, return_correlation=True):
    print("svd input shape:", feature_mat.shape)

    # scale if input too large
    # X = np.average(feature_mat, axis=1)
    if feature_mat.shape[-1] * feature_mat.shape[-2] >= 10000:
        X = []
        for item in feature_mat:
            X.append([])
            for layer in item:
                scaled_layer = rescale(layer, 0.25, anti_aliasing=False)
                X[-1].append(scaled_layer)
        X = np.asarray(X)
        print("svd rescaled shape:", X.shape)
        X = X.reshape(X.shape[0], -1)
    else:
        X = feature_mat.reshape(feature_mat.shape[0], -1)

    # n_samples, n_features = X.shape

    '''SVD'''
    xx_sample_projection_list = list()
    xx_sample_correlation_list = list()
    # sample projection calculation
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    for axis_index in range(axis_threshold):
        ev1 = Vh[axis_index]
        xx_sample_projection_list.append(X.dot(ev1))

    s_x = preprocessing.normalize(X)
    normalized_vh = preprocessing.normalize(Vh)
    for axis_index in range(axis_threshold):
        s_ev1 = normalized_vh[axis_index]
        xx_sample_correlation_list.append(s_x.dot(s_ev1))
    result_sample_projection_list = [[xx_sample_projection_list[i][j] for i in range(len(xx_sample_projection_list))]
                                     for j in range(len(xx_sample_projection_list[0]))]
    result_sample_correlation_list = [[xx_sample_correlation_list[i][j] for i in range(len(xx_sample_correlation_list))]
                                      for j in range(len(xx_sample_correlation_list[0]))]
    return np.asarray(result_sample_correlation_list) \
        if return_correlation else np.asarray(result_sample_projection_list)
