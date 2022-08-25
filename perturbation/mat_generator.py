import numpy as np
import os
from scipy import io as sio
from skimage import io


def read_img_files(path, file_name_list, ):
    mat = []
    info = []
    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        print(f"reading {file_path}")
        img = io.imread(file_path, as_gray=True)

        img = np.where(img > 0.5, 1, 0)

        info_list = file_name.split('.')[0].split('_')
        label = info_list[-1]

        mat.append(img.astype('uint8'))
        info.append(label)

    return np.array(mat), info


if __name__ == '__main__':
    organ_item = r"result"
    root_path = r'X:\temp\perturbation_1024\\'
    raw_file_path = root_path + organ_item
    np.seterr(all='ignore')

    all_file_name_list = [file_name for file_name in os.listdir(raw_file_path)
                          if (os.path.isfile(os.path.join(raw_file_path, file_name)))]
    raw_mat, meta_info = read_img_files(raw_file_path, all_file_name_list)
    sio.savemat(root_path + f'{organ_item}.mat', mdict={'feature_matrix': raw_mat,
                                                        'image_id': meta_info,
                                                        })
