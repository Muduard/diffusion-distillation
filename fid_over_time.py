from pytorch_fid import fid_score
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from natsort import natsorted

def compute_fids(path1, path2, n_datapoints, n):

    if os.path.isdir(path1):
        path1_stats = path1 + "_stats"
        compute_original_statistics(path1, stats_name=path1_stats)
        path1_stats = path1_stats + ".npz"
    else:
        path1_stats = path1
    test_dir2 = "sample_dir2/"
    fids = []
    filetest_dir2 = natsorted(os.listdir(path2))
    n_step = int(n / n_datapoints)
    for i in range(n_step-1, n, n_step):
        sample2 = filetest_dir2[:i]
        shutil.rmtree(test_dir2, ignore_errors=True)
        os.mkdir(test_dir2)
        for f in sample2:
            shutil.copyfile(path2 + f, test_dir2 + f)
        fids.append(fid_score.calculate_fid_given_paths((path1_stats,test_dir2),16,"cuda:0",2048,16))
        print(fids[-1])
    return fids


def fid_over_time_plot(path1, path2, n_datapoints, n_dataset):
    fids = compute_fids(path1, path2, n_datapoints, n_dataset)

    with plt.style.context("seaborn"):
        plt.plot(np.linspace(n_dataset/n_datapoints, n_dataset, n_datapoints, dtype=int), fids)
        plt.ylabel("FID")
        plt.xlabel("Sample Size")
        print(fids)
        plt.show()


def diff_fid_over_time_plot(path1, path2, path3, n_datapoints, n_dataset):
    fids1 = compute_fids(path1, path2, n_datapoints, n_dataset)
    fids2 = compute_fids(path1, path3, n_datapoints, n_dataset)
    fids1 = np.array(fids1)
    fids2 = np.array(fids2)
    fids = fids1 - fids2
    print(fids1)
    print(fids2)
    print(fids)
    with plt.style.context("seaborn"):
        plt.plot(np.linspace(n_dataset/n_datapoints, n_dataset, n_datapoints, dtype=int), fids)
        plt.ylabel("FID difference")
        plt.xlabel("Sample Size")

        plt.show()

def compute_original_statistics(path, stats_name):
    fid_score.save_fid_stats((path, stats_name), 16, "cuda:0",2048,16)

n_datapoints = 1
path1 = "27"
path1 = "cifar_stats.npz"
path2 = "./v-base83-32/"
path3 = "eps_good75/"

fid_over_time_plot(path1,path2,n_datapoints,2000)

