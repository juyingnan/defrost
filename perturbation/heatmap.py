import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io


def read_img_files(path, file_name_list, ):
    base = np.zeros((3000, 3000))
    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        print(f"reading {file_path}")
        img = io.imread(file_path, as_gray=True)

        img = np.where(img > 0.5, 1, 0)
        base = base + img

    return base


if __name__ == '__main__':
    organ_item = r"result"
    root_path = r'X:\temp\perturbation_1024\\'
    raw_file_path = root_path + organ_item
    np.seterr(all='ignore')

    all_file_name_list = [file_name for file_name in os.listdir(raw_file_path)
                          if (os.path.isfile(os.path.join(raw_file_path, file_name)))]
    overall = read_img_files(raw_file_path, all_file_name_list)

    # uncertainty map
    vmax = np.max(overall)
    uncertainty = (vmax - np.where(overall == 0, vmax, overall)) / vmax
    overall = overall / vmax

    fig = plt.figure(figsize=(6.75, 3))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.10,
                     )
    im1 = grid[0].imshow(overall, cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
    im2 = grid[1].imshow(uncertainty, cmap='plasma', interpolation='nearest', vmin=0, vmax=1)

    grid[1].cax.colorbar(im2)
    grid[1].cax.toggle_label(True)

    # fig.colorbar(im2, ax=axes, location='bottom', aspect=12)
    # plt.colorbar()
    plt.show()
