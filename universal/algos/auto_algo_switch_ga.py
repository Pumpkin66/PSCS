from universal.algo import Algo
from universal.algos.ga import GeneticAlgorithm
from universal.result import ListResult
import numpy as np
import pandas as pd
from universal import tools
import heapq
from universal.result import AlgoResult
import warnings

warnings.filterwarnings("ignore")


class AutoAlgoSwitch(Algo):
    PRICE_TYPE = 'absolute'
    REPLACE_MISSING = True

    def __init__(self, dataset, tau=6, epsilon=100, switchWinSize=30, risk_free_rate=0.01,
                 afa=0.5, mutation_rate=0.01, num_iterations=3, pop_size=500):

        window = 5
        super(AutoAlgoSwitch, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        self.window = window

        self.mutation_rate = mutation_rate
        self.num_generation = num_iterations
        self.pop_size = pop_size

        self.tau = tau
        self.histLen = 0

        self.filepath = "../data/" + dataset + ".pkl"
        self.data = pd.read_pickle(self.filepath)
        self.stockNum = self.data.shape[1]

        self.history = None
        self.epsilon = epsilon
        self.switchWinSize = switchWinSize
        self.risk_free_rate = risk_free_rate
        self.afa = afa

        self.flag = 0
        self.flags = [-1, -1, -1, -1, -1, -1]
        self.switch_time = []

        self.select_experts = []
        self.select_times = []

        self.price_type_names = {
            0: "max",
            1: "min",
            2: "ema",
            3: "ema2"
        }  # 价格预测

        self.direction_names = {
            0: "up",
            1: "down"
        }

        self.target_names = {
            0: "E",
            1: "F",
            2: "G"
        }

        self.prefixes = ['B', 'b']


        self.infixes = [
            f"{price}_{direction}"
            for price in self.price_type_names.values()
            for direction in self.direction_names.values()
        ]

        self.suffixes = list(self.target_names.values())

        self.expert_n = len(self.infixes) * len(self.suffixes)

        self.d = np.zeros(self.expert_n)  # 专家矩阵
        self.select_expert = 0

        for prefix in self.prefixes:
            for infix in self.infixes:
                for suffix in self.suffixes:
                    setattr(self, f'{prefix}_{infix}_{suffix}', [])

        self.optimizer = \
            GeneticAlgorithm(risk_free_rate=self.risk_free_rate,
                             mutation_rate=self.mutation_rate,
                             num_generation=self.num_generation,
                             pop_size=self.pop_size,
                             price_type_names=self.price_type_names,
                             direction_names=self.direction_names,
                             target_names=self.target_names)

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        self.history = history
        self.histLen = history.shape[0]
        last_window_history = history[-self.tau:]

        if self.histLen <= 7:
            self.select_expert = self.ga()
            self.flags.append(self.select_expert)

            self.select_times.append(self.histLen)
            self.select_experts.append(self.select_expert)
        else:
            if self.histLen % self.switchWinSize == 1:
                self.select_expert = self.ga()
                self.select_times.append(self.histLen)
                self.select_experts.append(self.select_expert)

            self.flags.append(self.select_expert)


        B = []

        strategies = []
        for target_key in self.target_names.keys():
            for price_key in self.price_type_names.keys():
                for direction_key in self.direction_names.keys():
                    strategy = (price_key, direction_key, target_key)
                    strategies.append(strategy)


        for price_type, direction, target_index in strategies:
            # cal p
            price_type_name = self.price_type_names[price_type]
            predicted_price = self.predict_price(last_window_history, price_type)
            # cal x
            direction_name = self.direction_names[direction]
            delta = self.relative_price(last_window_history, predicted_price, direction)
            # cal x_
            delta_t = self.calculate_delta_T(delta, self.stockNum)
            # cal last_b
            last_b = self.calculate_last_b(history)
            # cal b
            target_name = self.target_names[target_index]
            b = self.update_new_b(delta, delta_t, last_b, target_index)

            if isinstance(b, np.ndarray):
                pass
            else:
                b = b.to_numpy()

            attr_name = f'b_{price_type_name}_{direction_name}_{target_name}'
            setattr(self, attr_name, b)
            B.append(getattr(self, attr_name))

        if self.flags[-1] != self.flags[-2]:
            self.switch_time.append(self.histLen)
        print('\r', "[" + str(history.shape[0]) + "/" + str(self.data.shape[0]) + "]", end=" ", flush=True)
        return B[self.select_expert]

    def predict_price(self, last_window_history, price_index):
        """
        预测 ~ 绝对价格
        price_index:
           0 max
           1 min
           2 ema
           3 ema2
        """
        if price_index == 0:
            return last_window_history.max()
        elif price_index == 1:
            return last_window_history.min()
        elif price_index == 2:
            ema_values = []
            for i in range(self.tau):
                weight = self.afa * ((1 - self.afa) ** (self.tau - i))
                ema_values.append((weight * last_window_history.iloc[-(i + 1)]).to_list())
            ema = np.sum(ema_values, axis=0)
            return ema
        elif price_index == 3:
            ema_values = []
            for i in range(self.tau):
                weight = self.afa * ((1 - self.afa) ** i)
                ema_values.append((weight * last_window_history.iloc[-(i + 1)]).to_list())
            ema = np.sum(ema_values, axis=0)
            return ema
        else:
            print("Price Index: Only Support the number [0, 1, 2, 3]")
            return False

    def relative_price(self, last_window_history, price, relative_index):
        """
        预测 ~ 相对价格
        relative_index:
           0 预测价格在上  up
           1 预测价格在下  down
        """
        x = last_window_history.iloc[-1]
        if relative_index == 0:
            return price / x
        elif relative_index == 1:
            return x / price
        else:
            print(print("Relative Index: Only Support the number [0, 1]"))
            return False

    def calculate_return(self, S):
        """
        calculate return
        S: relative price
        """
        results = []
        names = []
        for suffix in self.suffixes:
            for infix in self.infixes:
                result = AlgoResult(S, getattr(self, f'B_{infix}_{suffix}')[:self.histLen])
                results.append(result)
                names.append(infix + '_' + suffix)
        res = ListResult(results, names)
        algos_return = res.to_dataframe()
        return algos_return

    def calculate_delta_T(self, delta, asset_amount):
        unit_vector = np.ones(delta.shape[0])
        delta_t = delta - (np.dot(unit_vector, delta) / asset_amount) * unit_vector
        return delta_t

    def calculate_last_b(self, history):
        last_b = history.iloc[-1] / history.iloc[-2]
        last_b = tools.simplex_proj(last_b)
        return last_b

    def update_new_b(self, delta, delta_t, last_b, update_index):
        """
        更新配比
        update_index:(control condition2)
            0
            1
            2
        """
        condition1 = np.dot(delta, delta_t)
        if condition1 == 0:
            return last_b

        last_b = np.array(last_b)
        delta = np.array(delta)

        condition2 = last_b + (self.epsilon * delta_t) / (condition1 ** 0.5)

        if condition2.min() < 0:
            if update_index == 0:
                last_b = delta
            elif update_index == 1:
                gamma_list = []
                for i in range(len(condition2)):
                    if condition2[i] < 0:
                        gamma_list.append(last_b[i] / (last_b[i] - condition2[i]))
                gamma = np.min(gamma_list)
                last_b = (1 - gamma) * last_b + condition2
            elif update_index == 2:
                delta_t_norm = np.linalg.norm(delta_t)
                delta_t = (delta_t * self.epsilon) / delta_t_norm
                last_b = last_b + delta_t
            else:
                print(print("Update Index: Only Support the number [0, 1, 2]"))
                return False

            # find the max one
            k = map(list(last_b).index, heapq.nlargest(1, list(last_b)))
            k = list(k)
            k = k[0]
            for i in range(delta.shape[0]):
                if i == k:
                    last_b[i] = 1
                else:
                    last_b[i] = 0

        else:
            last_b = condition2

        # simplex
        last_b = tools.simplex_proj(last_b)
        return last_b

    # call ga
    def ga(self):
        algos_return = self.calculate_return(self._convert_prices(self.data, 'ratio')[:self.histLen])
        algos_return = algos_return.iloc[-self.switchWinSize:]
        # get the best expert from ga
        expert_name = self.optimizer.optimize(algos_return)
        expert_index = algos_return.columns.get_loc(expert_name)
        return expert_index

    def weights(self, X, min_history=None):
        min_history = self.min_history if min_history is None else min_history
        # init
        B = X.copy() * 0.

        for infix in self.infixes:
            for suffix in self.suffixes:
                setattr(self, f'B_{infix}_{suffix}', X.copy() * 0.)

        last_b = self.init_weights(X.shape[1])
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        for infix in self.infixes:
            for suffix in self.suffixes:
                _last_b = self.init_weights(X.shape[1])
                if isinstance(_last_b, np.ndarray):
                    _last_b = pd.Series(_last_b, X.columns)
                setattr(self, f'last_b_{infix}_{suffix}', _last_b)

        # use history in step method?
        use_history = self._use_history_step()

        # run algo+
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.iloc[t] = last_b

            for infix in self.infixes:
                for suffix in self.suffixes:
                    B_attr = getattr(self, f'B_{infix}_{suffix}')
                    last_b_attr = getattr(self, f'last_b_{infix}_{suffix}')
                    B_attr.iloc[t] = last_b_attr

            # keep initial weights for min_history
            if t < min_history:
                continue

            # trade each `frequency` periods
            if (t + 1) % self.frequency != 0:
                continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t + 1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            for infix in self.infixes:
                for suffix in self.suffixes:
                    last_b_attr = getattr(self, f'b_{infix}_{suffix}')
                    setattr(self, f'last_b_{infix}_{suffix}', last_b_attr)

            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            for infix in self.infixes:
                for suffix in self.suffixes:
                    last_b_attr = getattr(self, f'last_b_{infix}_{suffix}')
                    if type(last_b_attr) == np.matrix:
                        last_b_attr = np.squeeze(np.array(last_b_attr))
                        setattr(self, f'last_b_{infix}_{suffix}', last_b_attr)

        return B

# if __name__ == '__main__':
#     datasetList = ['tse', 'hs300', 'msci', 'fof', 'crypto', 'csi300']
#     for dlist in datasetList:
#         path = '../data/' + dlist + '.pkl'
#         df_original = pd.read_pickle(path)
#         t = AutoAlgoSwitch(dlist)
#         df = t._convert_prices(df_original, 'raw')
#         B = t.weights(df)
#         # B.to_csv("../gamepsWeight/" + dlist + ".csv")
#         Return = AlgoResult(t._convert_prices(df_original, 'ratio'), B)
#         res = ListResult([Return], ['GA-MEPS']).to_dataframe()
#         last_return = res.iloc[-1].values[0]
#         print(dlist + ": last_return = ", last_return)
#         print(t.select_experts)
#         print(t.select_times)
#         print("====================" + dlist + " done=======================")
