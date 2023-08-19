### imports

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tabulate import tabulate
import cmocean

import support
import image
import cosmo_metrics

### data loading

#data, data_log = load_data(directory='output', in_is_log=True)
# alternative to select only certain image sets:
data, data_log = support.load_data(directory='output', in_is_log=True, select_names=('GT','Cond','Out_b', 'Out_e'))


### display the output data

metrics = [
    cosmo_metrics.PeakCounts(data['GT']),
    cosmo_metrics.PixelCounts(data['GT']),
    cosmo_metrics.PowerSpectrum(data['GT'])
]

# create an empty OrderedDict to store the output metric scores
metric_scores = OrderedDict()

for name in (name for name in list(data) if name != 'GT'): # must exclude GT - we don't want to compare it to itself!
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

image_comparison = image.ImageComparison(data_log, cmap=cmocean.cm.deep_r)

print(tabulate(data_rows, headers=header_row)) # use the tabulate package for easy table creation

plt.show() # show the figures