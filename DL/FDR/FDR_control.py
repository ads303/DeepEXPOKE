import numpy as np
import pandas as pd
import scipy.stats as stats
import os

class FDR_control:
    
    def __init__(self):
        print("__init__")
    
    def kfilter(self, W, offset=1.0, q=0.05):
        """
        Adaptive significance threshold with the knockoff filter
        :param W: vector of knockoff statistics
        :param offset: equal to one for strict false discovery rate control
        :param q: nominal false discovery rate
        :return a threshold value for which the estimated FDP is less or equal q
        """
        t = np.insert(np.abs(W[W!=0]),0,0)
        t = np.sort(t)
        ratio = np.zeros(len(t))
        for i in range(len(t)):
            ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        index = np.where(ratio <= q)[0]
        if len(index) == 0:
            thresh = float('inf')
        else:
            thresh = t[index[0]]
       
        return thresh
    
    def controlFilter(self, X_data_path, W_result_path, offset=1, q=-100, bootstrap=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        X_data = pd.read_csv(X_data_path, sep="\t")
        feature_list = X_data.columns.tolist()
        number_of_features = X_data.shape[1]

        result_path = W_result_path + '/result_epoch20_featImport.csv'
        result_file = pd.read_csv(result_path, header=None)
        result_list = result_file.iloc[0].values.tolist()
        print(len(result_list))

        original_result_list = result_list[:number_of_features]
        knockoff_result_list = result_list[number_of_features:]

        stats = []
        for index in range(number_of_features):
            stat = abs(original_result_list[index]) - abs(knockoff_result_list[index])
            stats.append(stat)

        W = np.array(stats)
        threshold = self.kfilter(W, offset=offset, q=q)
        if threshold == np.inf:
            threshold = 0

        selected_features = []
        for index in range(number_of_features):
            stat = abs(original_result_list[index]) - abs(knockoff_result_list[index])
            if stat > -100:
                selected_features.append((feature_list[index], stat, threshold))

        return selected_features
    
    def controlFilter_return_all(self, X_data_path, W_result_path, offset=1, q=0.05, bootstrap=False, seed=None, randomize=False):
        if seed is not None:
            np.random.seed(seed)

        X_data = pd.read_csv(X_data_path, sep="\t")
        feature_list = X_data.columns.tolist()
        number_of_features = X_data.shape[1]

        result_path = W_result_path + '/result_epoch20_featImport.csv'
        result_file = pd.read_csv(result_path, header=None)
        result_list = result_file.iloc[0].values.tolist()
        print(len(result_list))

        original_result_list = result_list[:number_of_features]
        knockoff_result_list = result_list[number_of_features:]

        if bootstrap:
            # Bootstrap sampling
            indices = np.random.choice(range(number_of_features), size=number_of_features, replace=True)
            original_result_list = [original_result_list[i] for i in indices]
            knockoff_result_list = [knockoff_result_list[i] for i in indices]
            feature_list = [feature_list[i] for i in indices]

        stats = []
        for index in range(number_of_features):
            stat = abs(original_result_list[index]) - abs(knockoff_result_list[index])
            stats.append(stat)

        # Introduce additional randomness if required
        if randomize:
            np.random.shuffle(stats)

        W = np.array(stats)
        threshold = self.kfilter(W, offset=offset, q=q)
        if threshold == np.inf:
            threshold = 0

        selected_features = []
        for index in range(number_of_features):
            stat = abs(original_result_list[index]) - abs(knockoff_result_list[index])
            if stat > threshold:
                selected_features.append((feature_list[index], stat, threshold))

        return W, selected_features, feature_list, stats
    
    def calculate_multiple_W_for_selected(self, X_data_path, W_result_path, selected_features, num_iterations=10, offset=1, q=0.05, bootstrap=False):
        all_W_statistics = {feature[0]: [] for feature in selected_features}

        for i in range(num_iterations):
            seed = np.random.randint(0, 10000)
            randomize = (i > 0)  # Introduce randomness for iterations after the first
            print(f"Iteration {i}, Seed: {seed}, Randomize: {randomize}")
            _, _, feature_list, stats = self.controlFilter_return_all(X_data_path, W_result_path, offset, q, bootstrap, seed, randomize=randomize)

            selected_feature_indices = [feature_list.index(feature[0]) for feature in selected_features]
            for feature, index in zip(selected_features, selected_feature_indices):
                all_W_statistics[feature[0]].append(stats[index])

        return all_W_statistics
