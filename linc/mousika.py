import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from .SoftTree import SoftTreeClassifier
from .mousika_utils import accuracy
from sklearn.metrics import classification_report
import time
import os


def KFold_split(y, num_fold=2):
    # allocate (X, y) according to category and store in a dictionary
    ind_dict = {}
    for i in range(len(y)):
        if y[i] not in ind_dict:
            ind_dict[y[i]] = []
        ind_dict[y[i]].append(i)

    # print(np.unique(y, return_counts=True))
    # exit()

    Fold = [[] for i in range(num_fold)]
    for k, v in ind_dict.items():
        num_data = len(v)
        num_data_per_fold = num_data // num_fold
        n = 0
        # print("category %d, total: %d" % (k, num_data))

        for i in range(num_fold):
            if i == num_fold - 1:
                Fold[i] += v[n:]
                break
            Fold[i] += v[n : n + num_data_per_fold]
            n += num_data_per_fold

    for i in range(num_fold):
        train_ind = []
        test_ind = np.array(Fold[i])

        for j in range(num_fold):
            if j != i:
                train_ind += Fold[j]
        yield np.array(train_ind), test_ind


def produce_soft_labels(data, round_num, fold_num, k=1, model="rf"):

    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    end = time.time()

    for i in range(round_num):
        for train_index, test_index in KFold_split(data[:, -1], num_fold=fold_num):

            # kf = KFold(n_splits=fold_num)
            # for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X, train_Y = train_set[:, :-1], train_set[:, -1].astype(int)
            test_X = test_set[:, :-1]
            if model == "rf":
                clf = RandomForestClassifier(
                    3, min_samples_leaf=5, criterion="gini", random_state=2023
                )
            clf.fit(train_X, train_Y)

            pred_prob = clf.predict_proba(test_X)
            soft_label[test_index] += pred_prob
            # print("fold.time: %.2f" % (time.time()-end))

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = (soft_label * k + hard_label) / (k + 1)
    # soft_label = (soft_label+ 2*hard_label) / (1+2)
    print("time cost: %.2f" % (time.time() - end))
    return soft_label


class Mousika:
    def __init__(self):
        self.clf = None

    def fit(self, train_data):

        # from imblearn.over_sampling import RandomOverSampler
        # ros = RandomOverSampler(random_state=0)
        # X, y = ros.fit_resample(train_data[:, :-1], train_data[:, -1])
        # train_data = np.column_stack((X, y))

        begin = time.time()
        feature_attr = ["d" for _ in range(len(train_data[0]) - 1)]
        soft_label = produce_soft_labels(
            train_data, round_num=1, fold_num=2, k=1, model="rf"
        )

        begin = time.time()
        self.clf = SoftTreeClassifier(n_features="all", min_sample_leaf=3)
        self.clf.fit(train_data[:, :-1], soft_label, feature_attr)

    def predict(self, test_data):
        return self.clf.predict(test_data[:, :-1])

    def score(self, test_data):
        pred = self.predict(test_data)
        return accuracy(pred, test_data[:, -1])

    def export(self, path):
        # check the path file is exist or not
        if os.path.exists(path):
            # delete the file
            os.remove(path)
        self.clf.show_tree(path)

    def report(self, test_data):
        pred = self.predict(test_data)
        return classification_report(test_data[:, -1], pred, digits=4)
