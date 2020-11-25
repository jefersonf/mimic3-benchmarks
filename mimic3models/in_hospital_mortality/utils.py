from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random

from scipy.stats import skew

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
               (3, 10), (3, 25), (3, 50)]


def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)


def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]['possible_values']) != 0:
            ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open(os.path.join(os.path.dirname(__file__), "resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data


def add_common_arguments(parser):
    """ Add all the parameters which are common across the tasks
    """
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--depth', type=int, default=1,
                        help='number of bi-LSTMs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--load_state', type=str, default="",
                        help='state file path')
    parser.add_argument('--mode', type=str, default="train",
                        help='mode: train or test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save state every x epoch')
    parser.add_argument('--prefix', type=str, default="",
                        help='optional prefix of network name')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rec_dropout', type=float, default=0.0,
                        help="dropout rate for recurrent connections")
    parser.add_argument('--batch_norm', type=bool, default=False,
                        help='batch normalization')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    parser.add_argument('--whole_data', dest='small_part', action='store_false')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--size_coef', type=float, default=4.0)
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    parser.set_defaults(small_part=False)


class DeepSupervisionDataLoader:
    r"""
    Data loader for decompensation and length of stay task.
    Reads all the data for one patient at once.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listfile : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """
    def __init__(self, dataset_dir, listfile=None, small_part=False):

        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header

        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), y) for (x, t, y) in self._data]
        self._data = sorted(self._data)

        mas = {"X": [],
               "ts": [],
               "ys": [],
               "name": []}
        i = 0
        while i < len(self._data):
            j = i
            cur_stay = self._data[i][0]
            cur_ts = []
            cur_labels = []
            while j < len(self._data) and self._data[j][0] == cur_stay:
                cur_ts.append(self._data[j][1])
                cur_labels.append(self._data[j][2])
                j += 1

            cur_X, header = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mas["ts"].append(cur_ts)
            mas["ys"].append(cur_labels)
            mas["name"].append(cur_stay)

            i = j
            if small_part and len(mas["name"]) == 256:
                break

        self._data = mas

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)



def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
