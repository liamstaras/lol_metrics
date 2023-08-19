import support
from histograms import make_mean_histogram
import numpy as np
import routines


class PeakCounts(support.Metric):
    def __init__(self, gt_image_set, name='peak counts', diff_offset=5, x_axis_name='peak value', y_axis_name='density', x_scale='linear', y_scale='log'):
        super().__init__(gt_image_set, name=name, diff_offset=diff_offset, x_axis_name=x_axis_name, y_axis_name=y_axis_name, x_scale=x_scale, y_scale=y_scale)
    def process_function(self, image_set):
        return make_mean_histogram(
            image_set = (routines.get_peaks(image) for image in image_set),
            bins = 100,
            data_range = (-0.5,3.5)
        )

class PowerSpectrum(support.Metric):
    def __init__(self, gt_image_set, name='power spectrum', diff_offset=0, x_axis_name='k', y_axis_name='power spectrum', x_scale='log', y_scale='log'):
        super().__init__(gt_image_set, name=name, diff_offset=diff_offset, x_axis_name=x_axis_name, y_axis_name=y_axis_name, x_scale=x_scale, y_scale=y_scale)
    def process_function(self, image_set):
        spec_out = [routines.calc_ps(image)['power'] for image in image_set]
        mean = np.mean(spec_out, axis=0)
        std = np.std(spec_out, axis=0, ddof=1)
        k_out = routines.calc_ps(image_set[0])['k']
        return support.Series(k_out, mean, std, len(image_set[0]))
    
class PixelCounts(support.Metric):
    def __init__(self, gt_image_set, name='pixel counts', diff_offset=10, x_axis_name='ln(1+pixel value)', y_axis_name='density', x_scale='linear', y_scale='log'):
        super().__init__(gt_image_set, name=name, diff_offset=diff_offset, x_axis_name=x_axis_name, y_axis_name=y_axis_name, x_scale=x_scale, y_scale=y_scale)
    def process_function(self, image_set):
        return make_mean_histogram(
            image_set = np.log(1+image_set),
            bins = 100,
            data_range = (-2,2)
        )