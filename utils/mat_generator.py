import numpy as np
import os
from scipy import io as sio

'''
mat format
    -- image_id
    -- batch_id
    -- patch_id
    
    -- id
    -- idx
    
    -- model_id
    -- layer_name
    
    -- layer_count
    -- width
    -- height
    
'''

def read_img_files(path, is_normalized=False):
    return

if __name__ == '__main__':
    organ_item = r"colon"
    root_path = r'X:\temp\\'
    raw_file_path = root_path + organ_item
    np.seterr(all='ignore')
    raw_mat, sample_rates, lengths, meta_info_labels = read_img_files(raw_file_path, is_normalized=True)

    sio.savemat(root_path + f'{organ_item}.mat', mdict={'feature_matrix': raw_mat,
                                                        'image_id': meta_info_labels[0],
                                                        'batch_id': meta_info_labels[1],
                                                        'patch_id': meta_info_labels[2],
                                                        'id': meta_info_labels[3],
                                                        'idx': meta_info_labels[4],
                                                        'model_id': meta_info_labels[5],
                                                        'layer_name': meta_info_labels[6],
                                                        'layer_count': meta_info_labels[7],
                                                        'width': meta_info_labels[8],
                                                        'height': meta_info_labels[9],
                                                        })