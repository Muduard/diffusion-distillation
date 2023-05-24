import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

batch_size = 16


def make_grid(path, files, rows, cols):
    images = []
    for f in files:
        images.append(Image.open(path + f))
    w = 32
    h = 32
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def get_sample_str(files, sample):
    res = []
    for s in sample:
        res.append(files[s])
    return res


path1 = "eps-1024-22-512/"
path2 = "eps-29-512/"

files1 = os.listdir(path1)
files2 = os.listdir(path2)

sample1 = np.random.random(batch_size) * len(files1)
sample2 = np.random.random(batch_size) * len(files2)

sample1 = sample1.astype(int)
sample2 = sample2.astype(int)

sample1_str = get_sample_str(files1, sample1)
sample2_str = get_sample_str(files2, sample2)

grid1 = make_grid(path1, sample1_str, 4, 4)
grid2 = make_grid(path2, sample2_str, 4, 4)


fig, axes = plt.subplots(1, 2)

axes[0].imshow(grid1)
axes[1].imshow(grid2)
plt.show()
