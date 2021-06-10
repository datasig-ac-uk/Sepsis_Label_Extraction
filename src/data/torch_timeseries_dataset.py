import torch
from torch.utils.data import Dataset

from data.functions import pytorch_rolling


class LSTM_Dataset(Dataset):
    """time series dataset class for generating rolling windows timeseries """

    def __init__(self, x, y, p, lengths, device):
        """

        :param x(list of tensors with variable time length): each sample tensor in the list x has shape (W_i,C) where W
        is the variable length of stay
        :param y(tensor array): output label array with shape (N,) N is total patients timestep s.t. N=sum(W_i)
        :param p (int) : rolling window length
        :param length (tensor array): array of lenght of each patient timeserie, W_i
        return: input and output pair for LSTM with fixed p rolling window. then for each output label at time t,
               we have input timeseries X_(t-p+1),...,X_(t) with shape(p,C), and the corresponding output is label_t
        """
        #
        self.lengths = lengths
        x = pytorch_rolling(x, 1, p, return_same_size=True)
        x_list = []
        for i, l in enumerate(self.lengths):
            x_list.append(x[i, 0:int(l), :])

        x = torch.cat(x_list).transpose(1, 2)

        x[torch.isnan(x)] = 0
        self.x = x

        self.y = y
        self.p = p
        self.device = device

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        return self.x[idx].to(self.device), self.y[idx].to(self.device)
