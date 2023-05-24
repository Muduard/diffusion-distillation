from pytorch_fid import fid_score
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt


path1 = "c10/cifar10/"
path2 = "eps-1024-22/"

files1 = os.listdir(path1)
files2 = os.listdir(path2)

n_datapoints = 10
n = len(files2)
n_step = int(n/n_datapoints)
fids = []
s1 = "sample_dir1/"
s2 = "sample_dir2/"
for i in range(n_step-1,n,n_step):
    sample1 = files1[:]
    sample2 = files2[:i]
    shutil.rmtree(s1, ignore_errors=True)
    shutil.rmtree(s2, ignore_errors=True)
    os.mkdir(s1)
    os.mkdir(s2)
    for f in sample1:
        shutil.copyfile(path1 + f, s1 + f)
    for f in sample2:
        shutil.copyfile(path2 + f, s2 + f)
    fids.append(fid_score.calculate_fid_given_paths((s1,s2),16,"cuda:0",2048,16))

with plt.style.context("seaborn"):
    plt.plot(range(n_datapoints), fids)
    print(fids)
    plt.show()
