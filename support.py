import matplotlib.pyplot as plt
import numpy as np

class Series:
    # x: x values, y_mean: y values, y_std_err: standard error in the mean for each y value
    def __init__(self, x, y_mean, y_std, sample_size):
        self.x = x
        self.y_mean = y_mean
        self.y_std = y_std
        self.sample_size = sample_size
    @property
    def y_std_err(self):
        return self.y_std/np.sqrt(self.sample_size)

class Metric:
    def __init__(self, name, process_function, gt_image_set, data_flags=(), diff_offset=0, x_axis_name='', y_axis_name='', x_scale='linear', y_scale='linear'):
        ## initalize basic attributes
        self.process_function = process_function
        self.diff_offset = diff_offset
        self.data_flags = data_flags
        self.name = name
        fig, self.axes = plt.subplots(2, sharex=True)
        fig.suptitle(name) # set title for figure
        
        ## configure subplots
        # axis names
        self.axes[0].set_ylabel(y_axis_name)
        self.axes[1].set_ylabel('relative difference')
        self.axes[1].set_xlabel(x_axis_name)

        # axis scales (note: x is shared)
        self.axes[0].set_xscale(x_scale)
        self.axes[0].set_yscale(y_scale)
        self.axes[1].set_xscale(x_scale)
        self.axes[1].set_yscale('linear') # relative difference always has linear scale

        # create empty placeholder for legend object
        self.legend = None

        # advance the colour iterator on the second set of axes, as no GT image is plotted there
        self.axes[1].plot([])
        self.axes[1].fill_between([],[])
        
        ## load and plot ground truth data
        self.gt_series = self.process_function(gt_image_set)
        self.plot_series(self.gt_series, axes_index=0, name='GT')

    def add_image_set(self, image_set, name, error_alpha=0.3):
        data_series = self.process_function(image_set)
        diff_series = self.diff_function(data_series)
        self.plot_series(data_series, name, axes_index=0, error_alpha=error_alpha)
        self.plot_series(diff_series, name, axes_index=1, error_alpha=error_alpha)
        return np.mean(np.abs(diff_series.y_mean)/diff_series.y_std_err)
    
    def plot_series(self, series, name, axes_index, error_alpha=0.3):
        self.axes[axes_index].plot(series.x, series.y_mean, label=name)
        self.axes[axes_index].fill_between(series.x, series.y_mean-series.y_std_err, series.y_mean+series.y_std_err, alpha=error_alpha)
        self.legend = self.axes[0].legend()
    
    def diff_function(self, data_series):
        diff_y_mean = data_series.y_mean/self.gt_series.y_mean - 1
        diff_y_std = np.sqrt(np.abs(
            (data_series.y_std_err**2 + (data_series.y_mean**2)/data_series.sample_size)/(self.gt_series.sample_size*self.gt_series.y_std_err**2 + self.gt_series.y_mean**2)
            - data_series.y_mean**2/(self.gt_series.sample_size*self.gt_series.y_mean**2)
        ))*np.sqrt(data_series.sample_size)
        return Series(
            self.gt_series.x[self.diff_offset:],
            diff_y_mean[self.diff_offset:],
            diff_y_std[self.diff_offset:],
            data_series.sample_size
        )