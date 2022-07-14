import numpy as np


# get FTU and non-FTU label
def get_is_ftu_label(mask_mat, threshold=0.05, need_rates=False):
    ftu_labels = []
    non_zero_rates = []
    for mask in mask_mat:
        non_zero_rate = len(np.nonzero(mask)[0]) / (mask.shape[1] * mask.shape[2])
        non_zero_rates.append(non_zero_rate)
        # print(non_zero_rate)
        if non_zero_rate > threshold:
            ftu_labels.append(1)
        elif non_zero_rate > threshold / 100:
            ftu_labels.append(0)
        else:
            ftu_labels.append(-1)
    if need_rates:
        return ftu_labels, non_zero_rates
    else:
        return ftu_labels


def get_norm(mat):
    mat_min, mat_max = mat.min(0), mat.max(0)
    mat_norm = (mat - mat_min) / (mat_max - mat_min)
    return mat_norm
