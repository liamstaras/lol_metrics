import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.image import AxesImage
from collections import OrderedDict
from data import *


class ImageComparison:
    def __init__(self, data, cmap=plt.get_cmap()):
        self.data = data
        self.cmap = cmap
        self.fig, axes = plt.subplots(1,len(self.data))
        self.axes = OrderedDict((name, axes[i]) for i, name in enumerate(self.data))
        for name in self.data:
            self.axes[name].set_title(name)
        self.slider_ax = self.fig.add_axes([0.25, 0.05, 0.63, 0.0225])
        self.bar_ax = self.fig.add_axes([0.05, 0.25, 0.0225, 0.63])
        self.image_slider = Slider(
            ax=self.slider_ax,
            label="Image",
            valmin=0,
            valmax=len(self.data['GT'])-1,
            valstep=1,
            valinit=0,
            orientation="horizontal"
        )
        self.slider_ax.add_artist(self.slider_ax.xaxis)
        sl_xticks = np.arange(len(self.data['GT']))
        self.slider_ax.set_xticks(sl_xticks)
        self.axes_images = OrderedDict(
            (name, self.axes[name].imshow(np.zeros((128,128)),cmap=self.cmap)) for name in self.data
        )
        self.colorbar = self.fig.colorbar(self.axes_images['GT'], self.bar_ax)
        self.image_slider.on_changed(self.show_images)
        self.show_images(0)
        
    def show_images(self, index):
        vmin = np.min([self.data[name][index] for name in self.data])
        vmax = np.max([self.data[name][index] for name in self.data])
        for name in self.data:
            self.axes_images[name].set_data(self.data[name][index])
            self.axes_images[name].set_clim(vmin=vmin, vmax=vmax)
        self.fig.canvas.draw_idle()
