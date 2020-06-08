import json
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from util import AverageMeter, build_timeseries, trim_dataset
from model import LSTM_Model


TIME_STEPS = 60
BATCH_SIZE = 20
EPOCHS = 300
LR = 0.0001


def visualize(df_ge):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(df_ge["Open"], linewidth=0.5)
    plt.plot(df_ge["High"], linewidth=0.5)
    plt.plot(df_ge["Low"], linewidth=0.5)
    plt.plot(df_ge["Close"], linewidth=0.5)
    plt.title('GE stock price history')
    plt.ylabel('Price (USD)')
    plt.xlabel('Days')
    plt.legend(['Open', 'High', 'Low', 'Close'], loc='upper left')
    plt.subplot(2, 1, 2)
    plt.plot(df_ge["Volume"], linewidth=0.5)
    plt.title('GE stock volume history')
    plt.ylabel('Volume')
    plt.xlabel('Days')

    plt.show()


def train():
    df_ge = pd.read_csv("ge.txt", engine='python')
    df_ge.tail()

    # visualize(df_ge)

    train_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
    print("Train and Test size", len(df_train), len(df_test), flush=True)
    # scale the feature MinMax, build array
    x = df_train.loc[:, train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    x_t, y_t = build_timeseries(x_train, 3)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    x_temp, y_temp = build_timeseries(x_test, 3)
    x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)

    model = LSTM_Model()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

    model.train()

    loss_epo_list = list()
    loss_iter_list = list()

    for i in range(EPOCHS):
        total_iter = int(x_t.shape[0] / BATCH_SIZE)
        loss_avg = AverageMeter()
        for j in range(total_iter):
            input = x_t[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :, :]
            gt = y_t[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            gt = torch.Tensor(gt)
            input = torch.Tensor(input)
            input = input.permute(1,0,2)
            output, loss = model(input, gt)
            loss_avg.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_iter_list.append(loss.item())
            if j%30 == 0:
                print("Epoch{}:{}/{} loss:{}".format(i, j, total_iter, loss_avg.avg), flush=True)

        loss_epo_list.append(loss_avg.avg)
        with open("loss_epo.json", 'w') as f:
            json.dump(loss_epo_list, f, indent=4)
        with open("loss_iter.json", 'w') as f:
            json.dump(loss_iter_list, f, indent=4)


    torch.save({
        'loss': loss,
        'state_dict': model.state_dict(),
    }, '{}.pth'.format('checkpoint'))


def evaluate():
    model = LSTM_Model()
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict['state_dict'], strict=True)

    df_ge = pd.read_csv("ge.txt", engine='python')
    df_ge.tail()

    # visualize(df_ge)

    train_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
    print("Train and Test size", len(df_train), len(df_test), flush=True)
    # scale the feature MinMax, build array
    x = df_train.loc[:, train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    x_t, y_t = build_timeseries(x_train, 3)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    # x_temp, y_temp = build_timeseries(x_test, 3)
    # x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
    # y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)

    model.eval()
    y_gt = list()
    y_pred = list()
    x = np.array(list(range(11246+60, 11246+2812)))

    for i in range(x_test.shape[0] - TIME_STEPS):
        input = x_test[i:i+TIME_STEPS, :]
        gt = x_test[i+TIME_STEPS, 3]
        y_gt.append(gt)
        input = torch.Tensor(input)
        input = input.unsqueeze(dim=1)
        with torch.no_grad():
            output = model(input)
        output = np.array(output)
        y_pred.append(output)

    y_pred = np.array(y_pred)
    y_gt = np.array(y_gt)
    mse = np.sum(np.power(y_pred - y_gt, 2)) / y_pred.shape[0]
    # mse:  0.00023299293330091727
    print("mse: ",mse)
    y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
    y_gt_org = (y_gt * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

    plt.figure()
    plt.ylabel('Price (USD)')
    plt.xlabel('Days')
    plt.title('Prediction vs Real Stock Price')
    plt.plot(x, y_gt_org, linewidth=0.8)
    plt.plot(x, y_pred_org, linewidth=0.8)
    plt.legend(['Real', 'Prediction'], loc='upper left')
    plt.show()


def visualize_loss():
    with open("loss_epo.json", 'r') as f:
        loss_epo = json.load(f)
    with open("loss_iter.json", 'r') as f:
        loss_iter = json.load(f)
    loss_epo = np.array(loss_epo)
    loss_iter = np.array(loss_iter) / 8
    x_epo = np.array(list(range(len(loss_epo))))
    x_iter = np.array(list(range(len(loss_iter))))
    x_iter = x_iter * max(x_epo) / max(x_iter)

    plt.figure()
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.plot(x_iter, loss_iter, linewidth=0.8)
    plt.plot(x_epo, loss_epo, linewidth=1.5)
    plt.legend(['Iteration (/8)', 'Epoch'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    train()