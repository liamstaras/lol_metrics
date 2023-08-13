import numpy as np
import os
import parse
from collections import OrderedDict

def load_data(directory='output', in_is_log=True, select_names=None):
    # make blank dicts to store loaded data and the array shape
    data_dict = dict()
    shape = dict()
    # loading loop
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        parsed = parse.parse('{type}_{index:d}{rest}', filename)
        index = parsed['index']
        name = parsed['type']+parsed['rest'].split('.')[0]
        extn = parsed['rest'].split('.')[1]
        if extn == 'npy':
            array = np.load(f).squeeze()
            if select_names is None or (select_names is not None and name in select_names):
                if name not in data_dict:
                    data_dict[name] = dict()
                    shape[name] = array.shape
                data_dict[name][index] = (array)

    ## convert from dict of dicts to dict of numpy arrays, and giving log and non-log data

    data = OrderedDict()
    data_log = OrderedDict()
    for name in sorted(data_dict):
        data[name] = np.zeros((max(data_dict[name])+1, *shape[name]))
        data_log[name] = np.zeros((max(data_dict[name])+1, *shape[name]))
        for index in data_dict[name]:
            if in_is_log:
                data_log[name][index][:] = data_dict[name][index][:]
                data[name][index][:] = np.exp(data_dict[name][index][:])-1
            else:                
                data_log[name][index][:] = np.log(1+data_dict[name][index][:])
                data[name][index][:] = data_dict[name][index][:]
    
    return data, data_log