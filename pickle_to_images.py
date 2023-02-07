import os
import pickle
from tqdm import tqdm
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_folder = 'cifar-10-batches-py/'
image_folder = "cifar10/"
#image_folder = "c/"
files = os.listdir(batch_folder)
files = list(filter(lambda x: x[:4] == "data", files))
for f in files:
    x = unpickle(batch_folder + f)

    images = x[b'data']
    names = x[b'filenames']
    names = list(map(lambda n: n.decode("utf-8"), names))
    for i in tqdm(range(len(names))):
        im = images[i].reshape((3, 32, 32))
        im = im.transpose(1, 2, 0)
        im = Image.fromarray(im)
        im.save(f'{image_folder}{names[i]}')

