import numpy as np
import os
from scipy import io as sio
from skimage import io

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
    -- nrow
    -- width
    -- height
    
'''

layer_nrow_dict = {
    'encoder0': 8,
    'encoder1': 16,
    'encoder2': 32,
    'encoder3': 32,
    'encoder4': 64,
    'decoder0': 8,
    'decoder1': 8,
    'decoder2': 8,
    'decoder3': 8,
    'decoder4': 8,
    'upsample1': 8,
    'upsample2': 8,
    'upsample3': 8,
    'upsample4': 8,
    'deep1': 8,
    'deep2': 8,
    'deep3': 8,
    'deep4': 8,
    'center': 32,
    'final_conv': 1,
    'final0': 8,
    'final1': 8,
    'final2': 1,
    'raw': 4,
}


def read_img_files(path, file_name_list, batch_size):
    idx = 0
    mat = []
    info = []

    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        img = io.imread(file_path, as_gray=True)
        # io.imsave(file_path, img)

        # read img metafile
        info_list = file_name.split('.')[0].split('_')
        meta_info_labels = []

        # -- image_id
        image_id = int(info_list[0].split('image')[1])
        # print("-- image_id", image_id)
        meta_info_labels.append(image_id)

        # -- batch_id
        batch_id = int(info_list[1].split('batch')[1])
        # print("-- batch_id", batch_id)
        meta_info_labels.append(batch_id)

        # -- patch_id
        patch_id = int(info_list[4].split('feature')[1])
        # print("-- patch_id", patch_id)
        meta_info_labels.append(patch_id)

        _id = batch_id * batch_size + patch_id
        meta_info_labels.append(_id)
        # print("-- id", id)

        meta_info_labels.append(idx)
        # print("-- idx", idx)
        idx += 1

        # -- model_id
        model_id = int(info_list[2].split('model')[1])
        # print("-- model_id", model_id)
        meta_info_labels.append(model_id)

        # -- layer_name
        layer_name = info_list[3]
        # print("-- layer_name", layer_name)
        meta_info_labels.append(layer_name)

        n_row = layer_nrow_dict[layer_name]
        img_w = img.shape[1]
        img_h = img.shape[0]
        patch_w = patch_h = img_w // n_row
        n_col = img_h // patch_h

        layer_count = n_row * n_col

        # print("---- img_w", img_w)
        # print("---- img_h", img_h)
        # print("---- n_row", n_row)
        # print("---- n_col", n_col)
        # print("-- layer_count", layer_count)
        # print("-- width", patch_w)
        # print("-- height", patch_h)

        meta_info_labels.append(layer_count)
        meta_info_labels.append(patch_w)
        meta_info_labels.append(patch_h)

        # img -> mat
        patches = []
        for c in range(n_col):
            for r in range(n_row):
                patch = img[c * patch_h: c * patch_h + patch_h, r * patch_w: r * patch_w + patch_w] * 255
                patches.append(patch.astype(np.uint8))
        del img
        # idd = 0
        # for p in patches:
        #     io.imsave(f"{idd}.jpg", p)
        #     idd += 1

        mat.append(np.stack(patches, axis=0))
        info.append(meta_info_labels)

    return mat, info


def read_raw_img_files(path, file_name_list, batch_size):
    idx = 0
    mat = []
    info = []

    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        img = io.imread(file_path)
        # io.imsave(file_path, img)

        # read img metafile
        info_list = file_name.split('.')[0].split('_')

        image_id = int(info_list[0].split('image')[1])
        batch_id = int(info_list[1].split('batch')[1])
        layer_name = 'raw'

        n_row = layer_nrow_dict[layer_name]
        img_w = img.shape[1]
        img_h = img.shape[0]
        patch_w = patch_h = img_w // n_row
        n_col = img_h // patch_h

        layer_count = n_row * n_col

        # img -> mat
        patch_id = 0
        for c in range(n_col):
            for r in range(n_row):
                patch = img[c * patch_h: c * patch_h + patch_h, r * patch_w: r * patch_w + patch_w]

                _id = batch_id * batch_size + patch_id
                model_id = "none"

                meta_info_labels = [image_id, batch_id, patch_id,
                                    _id, idx,
                                    model_id, layer_name, layer_count, patch_w, patch_h]

                # save test
                # io.imsave(f"{image_id}_{batch_id}_{patch_id}_{_id}_{idx}.jpg", patch)

                mat.append(patch)
                info.append(meta_info_labels)
                patch_id += 1
                idx += 1
        del img
    return mat, info


if __name__ == '__main__':
    organ_item = r"colon"
    root_path = r'X:\temp\\'
    raw_file_path = root_path + organ_item
    np.seterr(all='ignore')

    all_file_name_list = [file_name for file_name in os.listdir(raw_file_path)
                          if (os.path.isfile(os.path.join(raw_file_path, file_name)))]
    file_name_list_dict = {}
    for file_name in all_file_name_list:
        file_name_info = file_name.split('_')
        if len(file_name_info) > 2:
            layer = file_name_info[3]
        else:
            layer = "raw"
        if layer not in file_name_list_dict:
            file_name_list_dict[layer] = []
        file_name_list_dict[layer].append(file_name)

    for l_name in file_name_list_dict:
        """ read images from folder """
        print('reading the images:%s\t%s' % (raw_file_path, l_name))
        if l_name != "raw":
            raw_mat, meta_info = read_img_files(raw_file_path, file_name_list_dict[l_name], batch_size=12)
        else:
            raw_mat, meta_info = read_raw_img_files(raw_file_path, file_name_list_dict[l_name], batch_size=12)
        sio.savemat(root_path + f'{organ_item}_{l_name}.mat', mdict={'feature_matrix': raw_mat,
                                                                     'image_id': [item[0] for item in meta_info],
                                                                     'batch_id': [item[1] for item in meta_info],
                                                                     'patch_id': [item[2] for item in meta_info],
                                                                     'id': [item[3] for item in meta_info],
                                                                     'idx': [item[4] for item in meta_info],
                                                                     'model_id': [item[5] for item in meta_info],
                                                                     'layer_name': [item[6] for item in meta_info],
                                                                     'layer_count': [item[7] for item in meta_info],
                                                                     'width': [item[8] for item in meta_info],
                                                                     'height': [item[9] for item in meta_info],
                                                                     })
