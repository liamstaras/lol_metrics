import numpy as np
import matplotlib.pyplot as plt
from support import *

def make_mean_histogram(image_set, bins, data_range):
    output = []
    for image in image_set:
        histogram, bin_edges = np.histogram(image, bins=bins, range=data_range)
        output.append(histogram)
    mean_bin_edges = (bin_edges[1:] + bin_edges[:-1])/2
    return Series(mean_bin_edges, np.mean(output, axis=0), np.std(output, axis=0, ddof=1)/np.sqrt(len(output)))
