from empatches import EMPatches
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import imgviz

image_path = rf'X:\temp\colon_mask\35275_160325_A_8_3_largeintestine.tiff'

img = io.imread(image_path)  # [:, ::-1]
image_resized = resize(img, (1120, 1120), anti_aliasing=True).astype(np.uint8) * 255

emp = EMPatches()
img_patches, indices = emp.extract_patches(image_resized, patchsize=320, overlap=0.5)

for i in range(6):
    for j in range(6):
        index = j * 6 + i
        patch = img_patches[i * 6 + j]
        io.imsave(rf'X:\temp\colon_mask\{index}.jpg', patch[::-1, :])
