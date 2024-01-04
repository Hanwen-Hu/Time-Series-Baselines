from torch.utils.data import Dataset
from torch import mean, std, Tensor
from pandas import read_csv, to_datetime
import numpy as np

class TimeSeries(Dataset):
    def __init__(self, pred_len, seq_len, channel_dim):
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.channel_dim = channel_dim
        self.data = None
        self.timestamp = None

    def __getitem__(self, item):
        x_begin = item
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = x_end + self.pred_len
        x = self.data[x_begin:x_end]
        t = self.timestamp[x_begin:x_end]
        y = self.data[y_begin:y_end]
        return x, t, y

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def _generate_timestamp(self, date_time):
        #  (month, day, weekday, hour, minute)
        timestamps = np.zeros([self.data.shape[0], 5])
        timestamps[:, 0] = date_time.dt.month
        timestamps[:, 1] = date_time.dt.day
        timestamps[:, 2] = date_time.dt.weekday
        timestamps[:, 3] = date_time.dt.hour
        timestamps[:, 4] = date_time.dt.minute
        return timestamps

    def _normalize(self):
        avg = mean(self.data, dim=0, keepdim=True)
        dev = std(self.data, dim=0, keepdim=True)
        self.data = (self.data - avg) / dev

    def _split(self, flag):
        if flag == 'train':
            self.data = self.data[:int(self.data.shape[0] * 0.7), -self.channel_dim:]
        elif flag == 'valid':
            self.data = self.data[int(self.data.shape[0] * 0.7):int(self.data.shape[0] * 0.8), -self.channel_dim:]
        elif flag == 'test':
            self.data = self.data[int(self.data.shape[0] * 0.8):, -self.channel_dim:]


class ETT_Hour(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len, channel_dim)
        dataset = read_csv('data/ETTh1.csv')
        assert channel_dim < dataset.shape[1]
        self.data = Tensor(dataset.iloc[:, 1:].values).to(device)
        date_time = to_datetime(dataset['date'])
        self.timestamp = Tensor(self._generate_timestamp(date_time))
        self._normalize()
        self._split(flag)


class ETT_Minute(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('data/ETTm1.csv')
        assert d_in < dataset.shape[1]
        self.data = Tensor(dataset.iloc[:, 1:].values).to(device)
        date_time = to_datetime(dataset['date'])
        self.timestamp = Tensor(self._generate_timestamp(date_time))
        self._normalize()
        self._split(flag)