import os

import numpy as np
import pandas as pd

from universal import tools, algo

from universal.algo import Algo

from sklearn.cluster import Birch

from scipy.spatial.distance import euclidean

class PSCS(Algo):
    """ PSCS """
    PRICE_TYPE = 'absolute'
    REPLACE_MISSING = True

    def __init__(self, datasetname, win_size, asset_num_each_clu=2, cluster_num=3, theta=1.1, epsilon=100, tran_cost=0):
        super(PSCS, self).__init__(min_history=1)

        self.tran_cost = tran_cost
        self.datasetname = datasetname
        #
        self.filepath = 'D:\\SZU\\master\\code\\UPalgoTest\\universal\\data\\' + datasetName + '.pkl'
        self.originData = pd.read_pickle(self.filepath)
        #
        self.win_size = win_size
        self.asset_num_each_clu = asset_num_each_clu
        self.cluster_num = cluster_num
        self.theta = theta

        self.epsilon = epsilon


    def init_weights(self, m):
        return np.ones(m) / m

    def cluster_2dim(self, clu_num, coordinate):

        birch = Birch(threshold=0.00001).fit(coordinate)
        cluster = birch.labels_

        point_each_clu = {}
        for i in range(len(cluster)):
            clu = cluster[i]
            if point_each_clu.get(clu) == None:
                point_each_clu[clu] = []
            point_each_clu[clu].append(i)
        return point_each_clu

    def select_subset_drop_rise_cluster(self, history, ratio_history, win_size, clu_num=3, asset_num_each_clu=2):
        win_data = history.iloc[-win_size:]
        win_ratio = ratio_history.iloc[-win_size:]
        tmp_algo = Algo()
        cwratio = tmp_algo._convert_prices(history, 'cwratio')  #
        cwr = cwratio.iloc[-win_size:]  # t

        ratio = cwr.values  # 如果 win_ratio 是 Pandas DataFrame，将其转换为 NumPy 数组
        cumulative_returns = np.prod(1 + ratio, axis=0) - 1  # t
        distance_matrix = np.zeros((len(cumulative_returns), len(cumulative_returns)))
        for i in range(len(cumulative_returns)):
            for j in range(len(cumulative_returns)):
                distance_matrix[i, j] = euclidean(cumulative_returns[i].ravel(), cumulative_returns[j].ravel())

        # 聚类
        result = self.cluster_2dim(clu_num, distance_matrix)
        select_metric = win_ratio.max(axis=0) * win_ratio.min(axis=0) / (self.theta * np.log(win_data.max(axis=0)/win_data.iloc[-1]) + 1)
        select_metric.reset_index(drop=True, inplace=True)
        # 从每个簇中选择最小的几个
        select_coreset = []
        for i in range(clu_num):
            smallest_select_metric = select_metric.iloc[result[i]].nsmallest(asset_num_each_clu).keys().tolist()
            for j in smallest_select_metric:
                select_coreset.append(j)
            select_coreset = list(set(select_coreset))

        return select_coreset

    def adjust_ppt_last_b_ratioLast(self, history, last_b):
        mean_ratio = np.log(history.iloc[-5:].mean()/history.iloc[-1])  # 10
        adjust_last_b = tools.simplex_proj(last_b + mean_ratio)  # 9
        return adjust_last_b

    def adjust_ppt(self, history, last_b, select_asset_index, x_tplus1, epsilon=100):
        asset_num = history.shape[1]
        select_asset_num = len(select_asset_index)
        onesd = (np.ones([select_asset_num, 1])).reshape(select_asset_num, 1)

        adjust_last_b = self.adjust_ppt_last_b_ratioLast(history.iloc[-self.win_size:, select_asset_index], last_b[select_asset_index])  # 9

        daily_port = adjust_last_b
        var_deta = np.dot(adjust_last_b, x_tplus1.T) ** 2 - np.linalg.norm(x_tplus1.T)**2 * np.linalg.norm(adjust_last_b) **2 + np.linalg.norm(x_tplus1.T)**2 * epsilon**2
        var = np.dot(adjust_last_b, x_tplus1.T) + np.sqrt(var_deta)
        if  var_deta >= 0 and var >0 :
            daily_port = var / np.linalg.norm(x_tplus1.T)**2 * x_tplus1.T


        daily_port = np.array(daily_port)
        daily_port = tools.simplex_proj(daily_port)    # 13
        b = np.zeros(asset_num)
        b[select_asset_index] = daily_port
        return b

    def step(self, x, last_b, history):
        t = history.shape[0] - 1
        asset_num = history.shape[1]
        # 将history转格式
        tmp_algo = Algo()
        ratio_history = tmp_algo._convert_prices(history, 'ratio')  #价格变化率
        raw_history = tmp_algo._convert_prices(history, 'raw')   #原始价格

        select_asset_index = []
        if t < self.win_size:
            x_tplus1 = ratio_history.iloc[t, :]

            window_origin = history
            window_raw = raw_history
            window_ratio = ratio_history

            select_asset_index = [i for i in range(asset_num)]
        else:
            window_origin = history.iloc[(t - self.win_size + 1):]
            window_raw = raw_history.iloc[(t - self.win_size + 1):]
            window_ratio = ratio_history.iloc[(t - self.win_size + 1):]
            # 通过聚类构造核心集
            select_asset_index = self.select_subset_drop_rise_cluster(history, ratio_history, self.win_size,
                                                                      clu_num=self.cluster_num, asset_num_each_clu=self.asset_num_each_clu)
            select_asset_index.sort()

            from universal.algos.rmr import RMR
            rmr_pred = RMR().predict(window_raw.iloc[-1, select_asset_index], window_raw.iloc[:, select_asset_index])  #  (6)
            x_tplus1 = rmr_pred
        print('\r', "["+str(history.shape[0])+"/"+str(self.originData.shape[0])+"]",
              end='', flush=True)
        b = self.adjust_ppt(history, last_b, select_asset_index, x_tplus1, self.epsilon)
        # print(b)
        return b

def _parallel_weights(tuple_args):
    self, X, min_history, log_progress = tuple_args
    try:
        return self.weights(X, min_history=min_history, log_progress=log_progress)
    except TypeError:   # weights are missing log_progress parameter
        return self.weights(X, min_history=min_history)


if __name__ == '__main__':
    datasetList = [
        # 'djia',
        # 'csi300',
        # 'NYSE  ',
        'hs300',
        # 'russel1000'
    ]


    for datasetName in datasetList:

        data_path = 'D:\\SZU\\master\\code\\UPalgoTest\\universal\\data\\' + datasetName + '.pkl'
        df_original = pd.read_pickle(data_path)  # DataFrame

        t = tools.quickrun(PSCS(datasetName, 30), df_original)
