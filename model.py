import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_features=5, hidden_features=100, mid_features=20, drop_out=0.5, criterion=nn.MSELoss()):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features)
        self.f1 = nn.Linear(hidden_features, mid_features)
        self.f2 = nn.Linear(mid_features, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

        self.criterion = criterion

    def forward(self, input, gt=None):
        output, (hn, cn) = self.lstm(input)
        x = self.dropout(hn)
        x = x.squeeze()
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        if gt is None:
            return x
        loss = self.criterion(x, gt)
        return x, loss