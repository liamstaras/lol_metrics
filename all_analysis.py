### imports

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tabulate import tabulate

from histograms import *
from data import *
from support import *
from routines import *

### data loading

data, data_log = load_data(directory='output', in_is_log=True)
# alternative to select only certain image sets:
#data, data_log = load_data(directory='output', in_is_log=True, select_names=('GT','Cond','Out_e'))


### display the output data

## functions to form data series from image sets

def pixel_counts(image_set):
    return make_mean_histogram(
        image_set = image_set,
        bins = (np.logspace(0, 0.7)-2),
        data_range = (-1,3)
    )

def peak_counts(image_set):
    return make_mean_histogram(
        image_set = (get_peaks(image) for image in image_set),
        bins = 100,
        data_range = (-0.5,3.5)
    )

def power_spectrum(image_set):
    spec_out = [np.abs(calc_ps(image)['power']) for image in image_set]
    mean = np.mean(spec_out, axis=0)
    std = np.std(spec_out, axis=0, ddof=1)
    k_out = np.abs(calc_ps(image_set[0])['k'])
    return Series(k_out, mean, std, len(image_set[0]))

## create Metric objects for each function above
## this requires the ground truth image set to be specified, as this is used in the difference calculation
## the diff_offset parameter is to remove any NaNs that occur at the start of the diff calculation

metrics = [
    Metric('pixel counts', pixel_counts, data['GT'], diff_offset=10, x_axis_name='pixel value', y_axis_name='density'),
    Metric('peak counts', peak_counts, data['GT'], diff_offset=5, x_axis_name='peak value', y_axis_name='density'),
    Metric('power spectrum', power_spectrum, data['GT'], diff_offset=0, x_axis_name='k', y_axis_name='power spectrum', x_scale='log', y_scale='log')
]

# create an empty OrderedDict to store the output metric scores
metric_scores = OrderedDict()

for name in (name for name in list(data) if name != 'GT'): # must exclude GT we don't want to compare it to itself!
    metric_scores[name] = OrderedDict()
    for metric in metrics:
        # add_image_set does three things: calculates and plots i. the series histogram ii. the relative difference, and iii. returns the metric score alla Davide's 'test_single_epoch'
        metric_scores[name][metric.name] = metric.add_image_set(data[name], name)

## create a nice table to show metric scores
metric_names = [metric.name for metric in metrics]
header_row = ['name'] + metric_names + ['quadrature sum']
data_rows = [
    [name] # first column is name
    + [metric_scores[name][metric_name] for metric_name in metric_names] # this is a list of all metric values
    + [np.sqrt(sum([metric_scores[name][metric_name]**2 for metric_name in metric_names]))] # sum the above list to get the total score
    for name in list(data) if name != 'GT' # again, calculating the metric for the ground truth is useless
]

print(tabulate(data_rows, headers=header_row)) # use the tabulate package for easy table creation

plt.show() # show the figures