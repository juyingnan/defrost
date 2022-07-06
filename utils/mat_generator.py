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
}


def read_img_files(path, batch_size, filter):
    """ read images from folder """
    print('reading the images:%s' % path)

    # add a temp filter for saving space
    file_name_list = [file_name for file_name in os.listdir(path)
                      if (os.path.isfile(os.path.join(path, file_name)) and filter in file_name)]
    mats = {}
    infos = {}

    idx = 0

    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        img = io.imread(file_path, as_gray=True)
        # io.imsave(file_path, img)

        # read img metafile
        info_list = file_name.split('.')[0].split('_')
        meta_info_labels = []

        # skip the raw image for now
        if len(info_list) < 3:
            continue

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

        id = batch_id * batch_size + patch_id
        meta_info_labels.append(id)
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

        if layer_name not in mats:
            mats[layer_name] = []

        if layer_name not in infos:
            infos[layer_name] = []

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

        mats[layer_name].append(np.stack(patches, axis=0))
        infos[layer_name].append(meta_info_labels)

    return mats, infos


if __name__ == '__main__':
    organ_item = r"colon"
    root_path = r'X:\temp\\'
    raw_file_path = root_path + organ_item
    np.seterr(all='ignore')
    raw_mats, meta_info_labels = read_img_files(raw_file_path, batch_size=12, filter="upsample1")

    for l_name in raw_mats:
        sio.savemat(root_path + f'{organ_item}_{l_name}.mat', mdict={'feature_matrix': raw_mats[l_name],
                                                                     'image_id': [item[0] for item in meta_info_labels[l_name]],
                                                                     'batch_id': [item[1] for item in meta_info_labels[l_name]],
                                                                     'patch_id': [item[2] for item in meta_info_labels[l_name]],
                                                                     'id': [item[3] for item in meta_info_labels[l_name]],
                                                                     'idx': [item[4] for item in meta_info_labels[l_name]],
                                                                     'model_id': [item[5] for item in meta_info_labels[l_name]],
                                                                     'layer_name': [item[6] for item in meta_info_labels[l_name]],
                                                                     'layer_count': [item[7] for item in meta_info_labels[l_name]],
                                                                     'width': [item[8] for item in meta_info_labels[l_name]],
                                                                     'height': [item[9] for item in meta_info_labels[l_name]],
                                                                     })
