from train import trainStockNet
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sliding_window import sliding_window_view
import numpy as np
import random
import os
import tensorflow as tf


class trainStockNetZS(trainStockNet):
    def data_preprocessing(self, kind=0, windows=50):
        df = pd.read_csv('./data/zs' + str(self.ticker) + '.csv')
        df = df.fillna(0)
        key = ['preCloseIndex', 'openIndex', 'lowestIndex', 'highestIndex', 'closeIndex', 'turnoverVol',
               'turnoverValue', 'CHG', 'CHGPct']
        self.k_long = len(key)

        x = np.array(df[key])
        y = np.array(df['CHG'])

        if kind == 0:  # 涨跌额
            y_max = abs(max(y.min(), y.max(), key=abs))
            # # 最大值
            # print('max:')
            # print(y_max)

            # y 归一化
            y = y / y_max

            # 归一化
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x = x_scaler.fit_transform(x)

            # y 回归正确的归一化
            # x[:, 5] = y.reshape((-1))
            y_min = 0
            # print(x.max())

        else:  # 收盘价
            pass

        # 滑动窗口
        x = sliding_window_view(x, (windows, self.k_long)).reshape((-1, windows, len(key)))
        # print('sliding_window_view')
        # print(x.shape)
        # print(y.shape)
        x = x[0:-1]
        y = y[windows:]
        # print(x.shape)
        # print(y.shape)

        self.x = x
        self.y = y
        self.y_max = y_max
        self.y_min = y_min

        self.scaler = x_scaler

        return x, y


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 设置随机性
    seed = 1998
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    model_list = ['baselines', 'nbeats', 'lstm_1_net', 'lstm_2_net', 'lstm_3_net', 'gru_3_net', 'rnn_3_net', 'bp_5_net']
    loss_list = ['nloss', 'mse', 'mae']
    ticker_list = ['v000688']

    for k in ticker_list:
        for j in model_list:
            for i in loss_list:
                print(k, j, i)
                # ticker, loss_fun, net_model, epochs
                t = trainStockNetZS(k, i, j, 50)
                t.train_main()

