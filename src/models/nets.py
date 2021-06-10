import torch
from torch import nn


class MLP(nn.Module):
    """ Simple multi-layer perception module. """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    """ Simple RNN. """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, nonlinearity='tanh', bias=True,
                 dropout=0):
        super(RNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.total_hidden_size = num_layers * hidden_channels

        self.rnn = nn.RNN(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)

        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, x):
        hidden = self.rnn(x)[0]
        return self.linear(hidden)


class LSTM_variable_length(nn.Module):
    """ Simple LSTM with variable length input. """

    def __init__(self, in_channels, hidden_channels, hidden_1, out_channels, num_layers=1, bias=True,
                 dropout=0):
        super(LSTM_variable_length, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.hidden_1 = hidden_1
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.total_hidden_size = num_layers * hidden_channels

        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
                            bias=bias,
                            dropout=dropout,
                            batch_first=True)
        #self.bn = torch.nn.BatchNorm1d(self.in_channels)
        self.linear1 = nn.Linear(self.hidden_channels, hidden_1)
        self.linear2 = nn.Linear(self.hidden_1, self.out_channels)

    def forward(self, x, lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)

        x = self.lstm(x)[0]
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # x=self.bn(hidden)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LSTM(nn.Module):
    """ Simple LSTM. """

    def __init__(self, in_channels, hidden_channels, hidden_1, out_channels, num_layers=1, bias=True,
                 dropout=0):
        super(LSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.hidden_1 = hidden_1
        self.num_layers = num_layers
        self.bias = bias
        #self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.total_hidden_size = num_layers * hidden_channels

        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
                            bias=bias,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        #self.bn = torch.nn.BatchNorm1d(self.in_channels)
        self.linear1 = nn.Linear(self.hidden_channels*2, hidden_1)
        self.linear2 = nn.Linear(self.hidden_1, self.out_channels)

    def forward(self, x):

        lstm = self.lstm(x)[0][:, -1, :]

        # x=self.bn(hidden)
        x = self.dropout1(lstm)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class GRU(nn.Module):
    """ Standard GRU. """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, bias=True, dropout=0):
        super(GRU, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.total_hidden_size = num_layers * hidden_channels

        self.gru = nn.GRU(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, x):
        hidden = self.gru(x)[0]
        return self.linear(hidden)
