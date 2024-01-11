import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeries(Dataset):
    def __init__(self, pred_len, seq_len):
        self.pred_len = pred_len
        self.seq_len = seq_len

    def _normalize(self):
        avg = torch.mean(self.data, dim=0, keepdim=True)
        std = torch.std(self.data, dim=0, keepdim=True)
        self.data = (self.data - avg) / std

    def _split(self, mode):
        total_len = self.data.shape[0] - self.seq_len - self.pred_len * 3 + 3
        max_train_idx = int(total_len * 0.7) + self.seq_len + self.pred_len - 1
        max_valid_idx = int(total_len * 0.8) + self.seq_len + self.pred_len * 2 - 2
        if mode == 'train':
            self.data = self.data[:max_train_idx]
        elif mode == 'valid':
            self.data = self.data[max_train_idx + self.pred_len - 1: max_valid_idx]
        elif mode == 'test':
            self.data = self.data[max_train_idx + self.pred_len - 1:]

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, item):
        left, mid, right = item, item + self.seq_len, item + self.seq_len + self.pred_len
        x = self.data[left: mid]
        y = self.data[mid: right]
        return x, y


class Electricity(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/LD2011_2014.txt', delimiter=';')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class ETTh(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/ETTh.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class ETTm(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/ETTm.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class Exchange(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/exchange_rate.csv', delimiter=',', header=None)
        assert dim <= dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class QPS(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/QPS.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class Solar(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/solar_alabama.csv', delimiter=',')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class Traffic(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/PeMS.csv', delimiter=',')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class Weather(TimeSeries):
    def __init__(self, device, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/mpi_roof.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(mode)


data_dict = {
    'Electricity': Electricity,
    'ETTh': ETTh, 'ETTm': ETTm,
    'Exchange': Exchange,
    'QPS': QPS,
    'Solar': Solar,
    'Traffic': Traffic,
    'Weather': Weather
}