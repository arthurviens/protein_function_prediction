import pandas as pd 
import numpy as np
import scipy.stats


def remove_outliers(data):
    z_scores = scipy.stats.zscore(data)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    print(f"{(1 - filtered_entries).values.sum()} outliers deleted")
    return data[filtered_entries]


def describe_histograms(desc, rm_outliers=True):
    for line in desc.index()[1:]:
        values = desc.loc[line]
        #maxvalues = remove_outliers(maxvalues)
        values.hist(bins=50)