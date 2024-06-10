import math
import pandas as pd
import numpy as np
# from click import launch
# from sklearn.ensemble import BaggingClassifier, VotingClassifier
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.metrics import roc_curve, f1_score, auc
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
import scipy.io as sio
import itertools
import warnings
from sklearn.decomposition import PCA
from operator import *
from sklearn.metrics import f1_score

try:
    from sklearn.utils import safe_indexing as safe_indexing
except ImportError:
    from sklearn.utils import _safe_indexing as safe_indexing

dataSets = []
more_Data = []
less_Data = []
point = []
left_point = []
right_point = []
random_sample = []
total_random_sample1 = []
total_random_sample = []
space = []
temporary = []
i = 0
danger_data = []
sample_pp = []
m_X = 0
m_Y = 0


def plt_Data(data, label):
    color = {0: 'r', 1: 'g', 2: 'k', 3: 'b', 4: 'y'}
    plt.figure(figsize=(5, 5))
    plt.axis([0, 1, 0, 1])

    print(set(label))
    j = 0
    for i in set(label):
        data0 = data[np.where(label == i)[0]]
        plt.plot(data0[:, 0], data0[:, 1], '.', color=color[j], markersize=5)
        j += 1

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('origin.png')
    plt.rcParams['savefig.dpi'] = 600
    # plt.show()


def DataChange(train_label, noise):
    sampleNumber = train_label.shape[0]
    data = np.random.rand(sampleNumber, 1)
    data = data * 10 - np.ones((sampleNumber, 1)) * noise * 10
    data[np.nonzero(data[:, 0] < 0), :] = -1
    data[np.nonzero(data[:, 0] >= 0), :] = 1
    new_train_label = np.multiply(data, train_label.reshape(sampleNumber, 1))
    return new_train_label[:, 0]


def getIR(data):
    More = []
    Minority = []
    for i in range(len(data)):
        if data[:, 0][i] == 1:
            Minority.append(data[:, 1:][i])
        else:
            More.append(data[:, 1:][i])
    IR = len(More) / len(Minority)
    IR = round(IR, 2)
    return IR


def __gain_data():
    i = 0
    f = open(r'vowel0.dat', encoding='utf-8')
    for line in f:
        dataSets.append(list(eval(str(line))))
    f.close()
    # print(dataSets)

    # dp = pd.read_csv('1.csv')
    # df = pd.DataFrame(dp)
    # df.dropna(axis=0, inplace=True)
    # df1 = df.values.tolist()
    # dataSets = df1


    label = np.array(dataSets)[:, -1]
    data_label = [[x] for x in label]
    dataSet = np.delete(dataSets, -1, axis=1)
    data = np.concatenate((data_label, dataSet), axis=1)

    # key = 'fourclass'
    #
    # warnings.filterwarnings("ignore")
    # fp = sio.loadmat("dataset16.mat")
    # data = fp[key]
    ir = getIR(data)

    if data[data[:, 0] == -1., :].shape[0] == 0:
        data[data[:, 0] == 0.0, 0] = -1.

    if data[data[:, 0] == -1., :].shape[0] > data[data[:, 0] == 1., :].shape[0]:
        many_data = data[data[:, 0] == -1., :]
        less_data_1 = data[data[:, 0] == 1., :]
        many_number = int(many_data.shape[0] / 3)
        less_data = less_data_1[:many_number, :]
    else:
        many_data = data[data[:, 0] == 1., :]
        less_data_1 = data[data[:, 0] == -1., :]
        many_number = int(many_data.shape[0] / 3)
        less_data = less_data_1[:many_number, :]
    many_data_1, many_data_2 = train_test_split(many_data, test_size=0.2)
    less_data_1, less_data_2 = train_test_split(less_data, test_size=0.2)

    train = np.vstack((many_data_1, less_data_1))
    while i < len(train[0:, 0]):
        if train[0:, 0][i] == 1:
            less_Data.append(train[:, 1:][i])
        else:
            more_Data.append(train[:, 1:][i])
        i = i + 1
    moreData = MinMaxScaler().fit_transform(more_Data)
    lessData = MinMaxScaler().fit_transform(less_Data)
    test = np.vstack((many_data_2, less_data_2))

    # X_train = train[:, 1:]
    # Y_train = train[:, 0]
    # X_test = test[:, 1:]
    # Y_test = test[:, 0]
    #
    # Y_train = DataChange(Y_train, 0.05)

    X_train = train[:, 1:]
    Y_original = train[:, 0]
    X_test = test[:, 1:]
    Y_test = test[:, 0]

    Y_train = DataChange(Y_original, 0.05)
    X_train = MinMaxScaler().fit_transform(X_train)

    data = np.column_stack((Y_train, X_train))
    plt_Data(X_train, Y_train)
    # fitClass(X_train, Y_train, X_test, Y_test, 'BPNN')

    # 求纯度
    # data_no_label = data[:, :-2]
    # num, dim = data_no_label.shape
    # count = Counter(data[:, -2])
    # label = max(count, key=count.get)
    # purity = count[label] / num
    # return label, purity
    return moreData, lessData, X_test, Y_test, X_train, Y_train, data, ir, Y_original


def linear(line):
    x1 = np.array(line)[0:2][0][0]
    y1 = np.array(line)[0:2][0][1]

    x2 = np.array(line)[0:2][1][0]
    y2 = np.array(line)[0:2][1][1]
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    if b < 0:
        return "y =" + "%.1f" % k + "x" + "%.1f" % b
    else:
        return "y =" + "%.1f" % k + "x+" + "%.1f" % b


# 处于多数类与少数类边缘的样本
def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=5).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=5).fit(imbalanced_featured_data).kneighbors(imbalanced_featured_data,
                                                                                     return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)


def seperate_minor_and_major_data(imbalanced_data_arr2):
    """
    将训练数据分开为少数据类数据集和多数类数据集
    :param imbalanced_data_arr2: 非平衡数集
    :return: 少数据类数据集和多数类数据集
    """

    labels_arr1 = imbalanced_data_arr2[:, 0]
    unique_labels_arr1 = np.unique(labels_arr1)
    if len(unique_labels_arr1) != 2:
        print('数据类别大于2，错误！')
        return

    minor_label = unique_labels_arr1[0] if np.sum(labels_arr1 == unique_labels_arr1[0]) \
                                           < np.sum(labels_arr1 == unique_labels_arr1[1]) else unique_labels_arr1[1]

    [rows, cols] = imbalanced_data_arr2.shape  # 获取数据二维数组形状
    minor_data_arr2 = np.empty((0, cols))  # 建立一个空的少数类数据二维数组
    major_data_arr2 = np.empty((0, cols))  # 建立一个空的多数类数据二维数组

    # 遍历每个样本数据，分开少数类数据和多数类数据
    for row in range(rows):
        data_arr1 = imbalanced_data_arr2[row, :]  # 取出每一行的数据
        # if data_arr1[-1] == minor_label:
        if data_arr1[0] == minor_label:
            # 如果类别标签为少数类类别标签，则将数据加入少数类二维数组中
            minor_data_arr2 = np.row_stack((minor_data_arr2, data_arr1))
        else:  # 否则，将数据加入多数类二维数组中
            major_data_arr2 = np.row_stack((major_data_arr2, data_arr1))

    return minor_data_arr2, major_data_arr2


# 设置Bezier曲线
def BezierLine(a, b):
    P = lambda t: (1 - t) * a + t * b
    points = np.array([P(t) for t in np.linspace(0, 1, 5)])
    return points


def decision_point(d, point, m, n, a_x, a_y):
    while n < len(point):
        X1 = np.array(point)[0:, 0][m]
        Y1 = np.array(point)[0:, 1][m]

        A = np.array(d)[0:, 1][a_y] - np.array(d)[0:, 1][a_x]
        B = np.array(d)[0:, 0][a_x] - np.array(d)[0:, 0][a_y]
        C = np.array(d)[0:, 0][a_y] * np.array(d)[0:, 0][a_x] - np.array(d)[0:, 0][a_x] * \
            np.array(d)[0:, 1][a_y]

        D = A * X1 + B * Y1 + C
        if D < 0:
            left_point.append([X1, Y1])
        elif D > 0:
            right_point.append([X1, Y1])

        m = m + 1
        n = n + 1
    return left_point, right_point


def after_sampling(purity, ir):
    if ir < 2:
        sampling_point = random.sample(purity, int(len(purity) * 0.6))
        return sampling_point
    elif 2 < ir < 3.5:
        sampling_point = random.sample(purity, int(len(purity) * 0.45))
        return sampling_point
    elif 4 < ir < 5.5:
        sampling_point = random.sample(purity, int(len(purity) * 0.27))
        return sampling_point
    elif 6 < ir < 7.5:
        sampling_point = random.sample(purity, int(len(purity) * 0.21))
        return sampling_point
    elif 7.6 < ir < 8.5:
        sampling_point = random.sample(purity, int(len(purity) * 0.18))
        return sampling_point
    elif 8.6 < ir < 10:
        sampling_point = random.sample(purity, int(len(purity) * 0.16))
        return sampling_point
    elif 8.6 < ir < 10:
        sampling_point = random.sample(purity, int(len(purity) * 0.15))
        return sampling_point
    elif 15 < ir < 17:
        sampling_point = random.sample(purity, int(len(purity) * 0.09))
        return sampling_point
    else:
        sampling_point = random.sample(purity, int(len(purity) * 0.06))
        return sampling_point


def randomSample(left_point, right_point, ir):
    if ir < 2:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.6))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.6))
            return random_data, right_point
    elif 2 < ir < 3.5:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.45))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.45))
            return random_data, right_point
    elif 4 < ir < 5.5:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.27))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.27))
            return random_data, right_point
    elif 6 < ir < 7.5:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.21))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.21))
            return random_data, right_point
    elif 7.6 < ir < 8.5:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.18))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.18))
            return random_data, right_point
    elif 8.6 < ir < 10:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.16))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.16))
            return random_data, right_point
    elif 8.6 < ir < 10:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.15))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.15))
            return random_data, right_point
    elif 15 < ir < 17:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.09))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.09))
            return random_data, right_point
    else:
        if len(left_point) < len(right_point):
            random_data = random.sample(right_point, int(len(right_point) * 0.06))
            return random_data, left_point
        else:
            random_data = random.sample(left_point, int(len(left_point) * 0.06))
            return random_data, right_point


# if len(left_point) < len(right_point):
#     random_data = random.sample(right_point, int(len(right_point) * 0.6))
#     # print("random_data", random_data)
# else:
#     random_data = random.sample(left_point, int(len(left_point) * 0.6))
#     # print("random_data", random_data)
# return random_data


# left_point, right_point = decision_point(number, m=0, n=1, a_x=0, a_y=1)
# print("left_point", left_point)
# print("right_point", right_point)
# randomSample(left_point, right_point)
#
# left_point, right_point = decision_point(number, m=0, n=1, a_x=1, a_y=2)
# print("left_point", left_point)
# print("right_point", right_point)
# randomSample(left_point, right_point)
#
# left_point, right_point = decision_point(number, m=0, n=1, a_x=2, a_y=3)
# print("left_point", left_point)
# print("right_point", right_point)
# randomSample(left_point, right_point)


def get_point_line_distance(point, line, dis_purity):
    x = 0
    p = 0
    z = 0
    while z < len(line) - 1:
        line_s_x = line[x][0]
        line_s_y = line[x][1]
        line_e_x = line[x + 1][0]
        line_e_y = line[x + 1][1]
        while p < len(point):
            point_x = np.array(point)[0:, 0][p]
            point_y = np.array(point)[0:, 1][p]
            if line_e_x - line_s_x == 0:
                return math.fabs(point_x - line_s_x)
            if line_e_y - line_s_y == 0:
                return math.fabs(point_y - line_s_y)
            k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
            b = line_s_y - k * line_s_x
            dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
            temporary.append(dis)
            p = p + 1
            if dis > dis_purity:
                space.append([point_x, point_y])
        x = x + 1
        z = z + 1
    return space


def maxPoint(point, z=0, j=1, i=0):
    global m_X, m_Y
    length = []
    while i < len(point):
        vector3 = np.array(point)[i]
        j = j + i
        while j < len(point):
            vector4 = np.array(point)[j]
            op = np.linalg.norm(vector3 - vector4)
            if op > z:
                z = op
                m_X = i
                m_Y = j
            j = j + 1
        j = 1
        i = i + 1
    length.append([np.array(point)[0:, 0][m_X], np.array(point)[0:, 1][m_X]])
    length.append([np.array(point)[0:, 0][m_Y], np.array(point)[0:, 1][m_Y]])

    lengthX = np.array(length)[0:, 0]
    lengthY = np.array(length)[0:, 1]
    if lengthX[0] > lengthX[1]:
        t = 0
    else:
        t = 1
    if lengthY[0] > lengthY[1]:
        n = 0
    else:
        n = 1
    length1 = copy.deepcopy(length)
    length.pop()
    length.pop()
    if t > n:
        length.append([np.array(length1)[0:, 0][t], np.array(length1)[0:, 1][t]])
    else:
        length.append([np.array(length1)[0:, 0][n], np.array(length1)[0:, 1][n]])
    return length


def distance(i=0, number=0, indexX=0, indexY=0):
    record = []
    a = np.linalg.norm(np.array(number)[0] - np.array(numberLess[0]))
    while i < len(number):
        vector1 = np.array(number)[i]
        j = 0
        while j < len(number):
            vector2 = np.array(numberLess)[j]
            op = np.linalg.norm(vector1 - vector2)
            if op < a:
                a = op
                indexX = i
                indexY = j
            j = j + 1
        # record.append(index)
        # print("record=", record)
        # d = math.sqrt(int((x ** 2)) + int((y ** 2)))
        # print(length)
        i += 1
        # length.append(d)
    # s.append(length)
    # s.sort(reverse=False)
    # print(number)
    # print(np.array(number)[0:, 0][index] + np.array(numberLess)[0:, 0][index])
    # print(np.array(number)[0:, 1][index] + np.array(numberLess)[0:, 1][index])

    # midX = int(np.array(number)[0:, 0][indexX] + np.array(numberLess)[0:, 0][indexY]) / 2
    # midY = int(np.array(number)[0:, 1][indexX] + np.array(numberLess)[0:, 1][indexY]) / 2

    midX = np.median((np.array(number)[0:, 0][indexX], np.array(numberLess)[0:, 0][indexY]))
    midY = np.median((np.array(number)[0:, 1][indexX], np.array(numberLess)[0:, 1][indexY]))

    # print(np.array(number)[0:, 0][index], np.array(number)[0:, 1][index])
    # print(np.array(numberLess)[0:, 0][index], np.array(numberLess)[0:, 1][index])
    # print("midX = ", midX)
    # print("midY = ", midY)
    record.append(np.array([midX, midY]))
    record = list(itertools.chain.from_iterable(record))
    # record.append(record1)
    return record





def otherList(numberLess, numberMore, j=0):
    total = []
    x = np.array(numberLess).shape[0]  # 2
    while j < np.array(numberMore).shape[0]:
        total.append(distance(number=random.sample(list(numberMore), x)))
        # return total
        j += np.array(numberLess).shape[0]
        # if x < np.array(numberMore).shape[0]:
        #     x += np.array(numberLess).shape[0]
        # else:
        #     break
    return total


def addPoint(s1, point, s):
    point = np.array(point)[0:2]
    X = 2 * np.array(point)[0][0] - np.array(point)[1][0]
    Y = 2 * np.array(point)[0][1] - np.array(point)[1][1]
    s1.insert(0, np.array([X, Y]))
    if s1[0:1][0][0] < 0 and s1[0:1][0][1] <= 0:
        s1[0:1][0][0] = 0
        s1[0:1][0][1] = 0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    elif s1[0:1][0][0] < 0 and s1[0:1][0][1] > 1:
        s1[0:1][0][0] = 0
        s1[0:1][0][1] = 1.0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    elif s1[0:1][0][0] > 1 and s1[0:1][0][1] < 0:
        s1[0:1][0][1] = 0
        s1[0:1][0][0] = 1.0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    elif s1[0:1][0][0] > 1 and s1[0:1][0][1] > 1:
        s1[0:1][0][0] = 1.0
        s1[0:1][0][1] = 1.0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    elif 0 < s1[0:1][0][0] < 1 and s1[0:1][0][1] < 0:
        s1[0:1][0][1] = 0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    elif s1[0:1][0][0] < 0 and 0 < s1[0:1][0][1] < 1:
        s1[0:1][0][0] = 0
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    else:
        s.insert(0, [s1[0:1][0][0], s1[0:1][0][1]])
    return s


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def DTClassifier(X_sampled, Y_sampled):
    DT = DecisionTreeClassifier()
    X_train, X_test, Y_train, Y_test = train_test_split(X_sampled, Y_sampled, test_size=0.25, random_state=0)
    DT.fit(X_train, Y_train)
    Y_te = DT.predict(X_test)
    return Y_te, Y_test


def fitClass(X_train, Y_train):
    DT = DecisionTreeClassifier()
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=0)
        DT.fit(X_train, Y_train)
        Y_te = DT.predict(X_test)
        score = accuracy_score(Y_test, Y_te)
        print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
        print(score)
    except:
        return 0.0, 0.0

    # # print()
    # # label_1 正类，label_2 负类
    # if len(np.array(Y_test)[np.array(Y_test) == -1.]) > len(np.array(Y_test)[np.array(Y_test) == 1]):
    #     label_1 = -1.
    #     label_2 = 1.
    # else:
    #     label_1 = 1.
    #     label_2 = -1.
    #
    # TP, FP, FN, TN = 0, 0, 0, 0
    # number_sample = len(Y_test)
    # for i in range(number_sample):
    #     if Y_te[i] == label_1 and Y_test[i] == label_1:
    #         TP += 1
    #     elif Y_te[i] == label_1 and Y_test[i] == label_2:
    #         FP += 1
    #     elif Y_te[i] == label_2 and Y_test[i] == label_1:
    #         FN += 1
    #     else:
    #         TN += 1
    #
    # # print(TP, FP, FN, TN)
    #
    # P = TP / (TP + FP)
    # R = TP / (TP + FN)
    # S = TN / (TN + FP)
    # F_measure = 2 * P * R / (P + R)
    # G_mean = (R * S) ** 0.5
    # # print(F_measure, G_mean)
    #
    # return F_measure, G_mean


def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(np.array(y_true).shape)
    # Ordering labels
    #  might be missing e.g. with set like 0,2 where 1 is missing
    # First find the unique labels, then map the labels to an ordered set
    # 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        np.array(y_true)[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(np.array(y_true)[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def judge_equal(x, y):
    result_x_y = eq(x, y)
    return result_x_y


def partition_sampling_test(X_train, Y_train, data, numberMore, numberLess):
    numberSample, numberFeature = X_train.shape
    train = np.hstack((X_train, Y_train.reshape(numberSample, 1)))
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # Index column, into two-dimensional array format
    train = np.hstack((train, index))  # Add index column

    total = copy.deepcopy(numberMore)
    total = list(total)

    d = []

    s = otherList(numberLess, numberMore, j=0)

    s1 = copy.deepcopy(s)

    c = addPoint(s1, s, s)
    # lastX = 2 * np.array(c)[len(c) - 1:len(c)][0][0] - np.array(c)[len(c) - 2:len(c) - 1][0][0] - 1
    # if lastX < 0:
    #     lastX = lastX + 1.3
    # lastY = 2 * np.array(c)[len(c) - 1:len(c)][0][1] - np.array(c)[len(c) - 2:len(c) - 1][0][1] - 1
    # if lastY < 0:
    #     lastY = lastY + 1
    # c.append([lastX, lastY])
    # print("c =", c)
    c.append(list(itertools.chain.from_iterable(maxPoint(c, z=0, j=1, i=0))))
    # d = np.array(list(set([tuple(t) for t in c])))
    # sorted(s, reverse=True)

    for i in c:
        if not i in d:
            d.append(i)

    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(data)
    imbalanced_featured_data = data[:, 1:]
    imbalanced_label_data = data[:, 0]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, 1:]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][0]

    danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)

    for j in range(len(danger_index)):
        if danger_index[j]:
            danger_data.append(j)

    # danger_index_data = safe_indexing(old_feature_data, danger_index)
    # print(len(danger_index_data))

    danger_index_data = safe_indexing(imbalanced_featured_data, danger_index)

    # sample = random.sample(list(danger_index_data), 6)
    # pca = PCA(n_components=2)
    # sample = pca.fit_transform(sample)
    # sample = MinMaxScaler().fit_transform(sample)

    pca = PCA(n_components=2)
    sample = pca.fit_transform(danger_index_data)
    sample = MinMaxScaler().fit_transform(sample)

    for i in range(len(sample)):
        d.append(np.array(sample[i]))
    d = np.array(d)

    # print(np.array(d)[:, -1])
    final = np.array(sorted(d, key=lambda x: x[1], reverse=True))
    # final = MinMaxScaler().fit_transform(final)

    X = np.array(final)[0:, 0]
    X = np.array(X)

    Y = np.array(final)[0:, 1]
    Y = np.array(Y)

    f1 = np.polyfit(X, Y, 9)
    # print("f1=", f1)

    p1 = np.poly1d(f1)
    # print("三次多样式函数", p1)

    y_values = p1(X)
    plt.plot(X, Y, 'o', X, y_values, 'r')


    # plt.savefig('curve.png')
    # plt.show()

    # X = np.array(d)[0:, 0]
    # X = np.array(X)

    # Y = np.array(d)[0:, 1]
    # Y = np.array(Y)

    # f1 = np.polyfit(X, Y, 3)
    # # print("f1=", f1)
    #
    # p1 = np.poly1d(f1)
    # # print("三次多样式函数", p1)
    #
    # y_values = p1(X)
    # plt.plot(X, Y, 'o', X, y_values, 'r')

    # 以下部分是一维样条函数
    # d_x = np.array(d)[0:, 0]
    # # print(d_x)
    # d_y = np.array(d)[0:, 1]
    # # print(d_y)
    #
    # xnew = np.arange(0, 2, 3)
    # func = interpolate.interp1d(d_x, d_y, kind='slinear')
    # # 利用xnew和func函数生成ynew,xnew的数量等于ynew的数量
    # ynew = func(xnew)
    #
    # plt.plot(d_x, d_y, 'g4-')
    # plt.show()

    # # 以下部分是基于Bezier曲线的
    # X = np.array(d)[0:, 0]
    # # print("X:", X)
    #
    # P0, P1 = np.array(d[0:2])
    # x, y = BezierLine(P0, P1)[:, 0], BezierLine(P0, P1)[:, 1]
    # # 计算出第一段的函数f(x)
    # y_0 = linear([P0, P1])
    # print("y_0", y_0)
    #
    # P2, P3 = np.array(d[1:3])
    # x1, y1 = BezierLine(P2, P3)[:, 0], BezierLine(P2, P3)[:, 1]
    # # 计算出第二段的函数f(x)
    # y_1 = linear([P2, P3])
    # print("y_1", y_1)
    #
    # P4, P5 = np.array(d[2:4])
    # x2, y2 = BezierLine(P4, P5)[:, 0], BezierLine(P4, P5)[:, 1]
    # # 计算出第三段的函数f(x)
    # y_2 = linear([P4, P5])
    # print("y_2", y_2)
    #
    # # P6, P7 = np.array(d[3:5])
    # # x3, y3 = BezierLine(P6, P7)[:, 0], BezierLine(P6, P7)[:, 1]
    # # 计算出第四段的函数f(x)
    # # y_3 = linear([P6, P7])
    # # print(y_3)
    #
    # # P8, P9 = np.array(d[4:6])
    # # x4, y4 = BezierLine(P8, P9)[:, 0], BezierLine(P8, P9)[:, 1]
    # # 计算出第五段的函数f(x)
    # # y_4 = linear([P8, P9])
    # # print(y_4)
    #
    # # plt.plot(x, y, x1, y1, x2, y2, x3, y3, x4, y4, 'b-')
    # plt.plot(x, y, x1, y1, x2, y2, 'b-')
    # plt.plot(*P0, 'r.')
    # plt.plot(*P1, 'r.')
    # plt.plot(*P2, 'r.')
    # plt.plot(*P3, 'r.')
    # plt.plot(*P4, 'r.')
    # plt.plot(*P5, 'r.')
    # # plt.plot(*P6, 'r.')
    # # plt.plot(*P7, 'r.')
    # # plt.plot(*P8, 'r.')
    # # plt.plot(*P9, 'r.')

    # plt.show()
    # global total
    numberLess = list(numberLess)
    total = list(total)
    t = 0
    while t < len(numberLess):
        total.append(numberLess[t])
        t = t + 1

    # if np.array(total).shape[0] > 2 and np.array(total).shape[1] > 2:
    #     pca = PCA(n_components=2)
    #     total = pca.fit_transform(total)
    #     total = MinMaxScaler().fit_transform(total)
    # print("total", total)

    # pca = PCA(n_components=2)
    # total = pca.fit_transform(total)
    # total = MinMaxScaler().fit_transform(total)
    # pca_numberLess = pca.fit_transform(numberLess)
    # pca_numberLess = MinMaxScaler().fit_transform(pca_numberLess)

    XTest = pca.fit_transform(X_test)
    XTest = MinMaxScaler().fit_transform(XTest)

    # train = pca.fit_transform(train[:, :-2])
    # train = MinMaxScaler().fit_transform(train)
    # train = np.hstack((train, Y_train.reshape(numberSample, 1)))
    # index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # Index column, into two-dimensional array format
    # train = np.hstack((train, index))

    X_train = pca.fit_transform(X_train)
    X_train = MinMaxScaler().fit_transform(X_train)

    # l_x = 0
    # l_y = 1
    # global left_point, sample_pp
    # global right_point
    # u = np.array(numberLess).shape[0]
    # while l_x < len(d) - 1:
    #     left_point.clear()
    #     right_point.clear()
    #     z = random.sample(list(total), u)
    #     left_point, right_point = decision_point(d, z, m=0, n=1, a_x=l_x, a_y=l_y)
    #     # print("left_point", len(left_point))
    #     # print("right_point", len(right_point))
    #     random_sample, num_point = randomSample(left_point, right_point, ir)
    #     # print(num_point)
    #     # decision_point(number, m=0, n=1, a_x=1, a_y=2)
    #     # decision_point(number, m=0, n=1, a_x=2, a_y=3)
    #
    #     # m = m + 1
    #     # n = n + 1
    #     for i in random_sample:
    #         if not i in total_random_sample1:
    #             total_random_sample1.append(i)
    #     if not num_point:
    #         l_x = l_x + 1
    #         l_y = l_y + 1
    #     else:
    #         # sample_pp = np.row_stack((total_random_sample1, num_point))
    #         sample_pp = np.concatenate((total_random_sample1, list(num_point)), axis=0)
    #     l_x = l_x + 1
    #     l_y = l_y + 1
    # # print(sample_pp)
    #
    # combination_sample = np.concatenate((sample_pp, list(pca_numberLess)), axis=0)
    # combination_total = copy.deepcopy(combination_sample)
    # combination_total = combination_total.tolist()
    # combination_total = list(combination_total)
    #
    # for p in range(len(combination_total)):
    #     for w in range(len(total)):
    #         if total[w][0] == combination_total[p][0] and total[w][1] == combination_total[p][1]:
    #             combination_total[p].append(-1)
    #
    # for e in range(len(combination_total)):
    #     for r in range(len(pca_numberLess)):
    #         if pca_numberLess[r][0] == combination_total[e][0] and pca_numberLess[r][1] == combination_total[e][1]:
    #             combination_total[e].append(1)
    #
    # print(combination_total)

    # m = 0
    # n = 1
    l_x = 0
    l_y = 1
    global left_point, sample_pp
    global right_point
    u = np.array(numberLess).shape[0]
    while l_x < len(d) - 1:
        left_point.clear()
        right_point.clear()
        z = random.sample(list(X_train), u)
        left_point, right_point = decision_point(d, z, m=0, n=1, a_x=l_x, a_y=l_y)
        random_sample, num_point = randomSample(left_point, right_point, ir)

        for i in random_sample:
            if not i in total_random_sample1:
                total_random_sample1.append(i)
        if not num_point:
            l_x = l_x + 1
            l_y = l_y + 1
        else:
            sample_pp = np.concatenate((total_random_sample1, list(num_point)), axis=0)
        l_x = l_x + 1
        l_y = l_y + 1

    total_sample = copy.deepcopy(sample_pp)

    label = []
    l = 0
    while l < len(Y_train):
        label.append([Y_train[l]])
        l = l + 1

    data_all = np.concatenate((label, X_train), axis=1)

    total_sample = total_sample.tolist()
    new_lst = []
    [new_lst.append(i) for i in total_sample if not new_lst.count(i)]
    total_sample = copy.deepcopy(new_lst)
    total_sample = list(total_sample)

    for a in range(len(total_sample)):
        for b in range(len(data_all)):
            if data_all[b][1] == total_sample[a][0] and data_all[b][2] == total_sample[a][1]:
                total_sample[a].append(np.array(data_all)[b][0])
    #
    # # for a in range(len(total_sample)):
    # #     for b in range(len(data_all)):
    # #         if judge_equal(data_all[:, 1:][b], total_sample[a]):
    # #             total_sample[a].append(np.array(data_all)[b][0])
    #
    #
    # # for p in range(len(total_sample)):
    # #     for w in range(len(numberLess)):
    # #         if numberLess[w][0] == total_sample[p][0] and numberLess[w][0] == total_sample[p][1]:
    # #             total_sample[p].append(1)
    #
    # # pca = PCA(n_components=2)
    # # total_sample = pca.fit_transform(total_sample)
    # # total_sample = MinMaxScaler().fit_transform(total_sample)
    # pca_numberMore = np.array(total)
    # pca_a = copy.deepcopy(pca_numberMore)
    # pca_a = pca_a.tolist()
    # pca_a = list(pca_a)
    # pca_numberLess = np.array(pca_numberLess)
    # pca_add = copy.deepcopy(pca_numberLess)
    # pca_add = pca_add.tolist()
    # pca_add = list(pca_add)
    #
    # for t in range(len(pca_a)):
    #     pca_a[t].append(1)
    #
    # for t in range(len(pca_numberLess)):
    #     pca_add[t].append(-1)
    #
    #
    # pca_train_data = np.concatenate((pca_a, list(pca_add)), axis=0)
    # # print(pca_train_data)
    #
    # train_label = pca_train_data[:, -1]

    linear_train_Y = []
    linear_train_X = []

    for o in range(len(total_sample)):
        linear_train_Y.append(np.array(total_sample)[o][2])
        linear_train_X.append(np.array(total_sample)[o][0:len(total_sample[0]) - 1])

    # for o in range(len(combination_total)):
    #     linear_train_Y.append(np.array(combination_total)[o][2])
    #     linear_train_X.append(np.array(combination_total)[o][0:len(combination_total[0])-1])
    # linear_train_Y = np.array(total_sample)[0:, -1]
    # linear_train_X = np.array(total_sample)[:, :-1]

    total_random_sample.clear()
    return linear_train_X, linear_train_Y, d, X_train, XTest


# def resample(purity, ir, noise):
#     if ir == 1.86 and noise == 0.05:
#         sampling_point = random.sample(purity, int(len(purity) * 0.975))
#         return sampling_point
#     elif ir == 1.86 and noise == 0.10:
#         sampling_point = random.sample(purity, int(len(purity) * 0.95))
#         return sampling_point
#     elif ir == 1.86 and noise == 0.15:
#         sampling_point = random.sample(purity, int(len(purity) * 0.885))
#         return sampling_point
#     elif ir == 1.86 and noise == 0.20:
#         sampling_point = random.sample(purity, int(len(purity) * 0.85))
#         return sampling_point
#     elif ir == 1.86 and noise == 0.30:
#         sampling_point = random.sample(purity, int(len(purity) * 0.75))
#         return sampling_point
#     elif ir == 1.86 and noise == 0.40:
#         sampling_point = random.sample(purity, int(len(purity) * 0.65))
#         return sampling_point


# 得到七次短的投票
def get_seven_voting(y1, y2, y3, y4, y5, y6, y7):
    y9 = []
    for j in range(len(y1)):
        if y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + y7[j] == 1 or y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + \
                y7[j] == 7 or y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + y7[j] == 5 or y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + y7[j] == 3:
            y9.append(1)
        elif y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + y7[j] == -1 or y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[
            j] + y7[j] == -5 or y1[j] + y2[j] + y3[j] + y4[j] + y5[j] + y6[j] + y7[j] == -3 or y1[j] + y2[j] + y3[j] + \
                y4[j] + y5[j] + y6[j] + y7[j] == -7:
            y9.append(-1)
    return y9


# 得到五次短的投票
def get_five_voting(y1, y2, y3, y4, y5):
    y6 = []
    for i in range(len(y1)):
        if y1[i] + y2[i] + y3[i] + y4[i] + y5[i] == 1 or y1[i] + y2[i] + y3[i] + y4[i] + y5[i] == 5 or y1[i] + y2[i] + \
                y3[i] + y4[i] + y5[i] == 3:
            y6.append(1)
        elif y1[i] + y2[i] + y3[i] + y4[i] + y5[i] == -1 or y1[i] + y2[i] + y3[i] + y4[i] + y5[i] == -5 or y1[i] + y2[
            i] + y3[i] + y4[i] + y5[i] == -3:
            y6.append(-1)
    return y6


# 按照短的进行测试(三次投票)
def get_third_voting(y1, y2, y3):
    y4 = []
    for i in range(len(y1)):
        if y1[i] == y2[i] == y3[i] == 1:
            y4.append(1)
        elif y1[i] == y2[i] == y3[i] == -1:
            y4.append(-1)
        elif y1[i] == y2[i] == 1 or y1[i] == y3[i] == 1 or y2[i] == y3[i] == 1:
            y4.append(1)
        elif y1[i] == y2[i] == -1 or y1[i] == y3[i] == -1 or y2[i] == y3[i] == -1:
            y4.append(-1)
    return y4


if __name__ == '__main__':
    numberMore = []
    numberLess = []
    numberMore, numberLess, X_test, Y_test, X_train, Y_train, data, ir, y_original = __gain_data()
    print(len(numberMore))
    print(len(numberLess))

    linear_train_X, linear_train_Y, d, X_train, XTest = partition_sampling_test(X_train, Y_train, data, numberMore,
                                                                                numberLess)

    Y_te1, Y_te4 = DTClassifier(linear_train_X, linear_train_Y)
    Y_te2, Y_te5 = DTClassifier(linear_train_X, linear_train_Y)
    Y_te3, Y_te6 = DTClassifier(linear_train_X, linear_train_Y)
    # Y_te4 = DTClassifier(X_train, linear_train_X, linear_train_Y)
    # Y_te5 = DTClassifier(X_train, linear_train_X, linear_train_Y)
    # Y_te6 = DTClassifier(X_train, linear_train_X, linear_train_Y)
    # Y_te7 = DTClassifier(X_train, linear_train_X, linear_train_Y)



    # pca = PCA(n_components=2)
    # pca_more = pca.fit_transform(numberMore)
    # pca_more = MinMaxScaler().fit_transform(pca_more)
    # pca_less = pca.fit_transform(numberLess)
    # pca_less = MinMaxScaler().fit_transform(pca_less)
    #
    # z = 0
    # purity_point = []
    # purity_most = []
    # purity = []
    # total_point = []
    # after_ir = []
    # u = np.array(numberLess).shape[0]
    # while z < np.array(numberMore).shape[0]:
    #     purity_most.clear()
    #     purity.clear()
    #     purity_point = random.sample(list(pca_more), u)
    #     for i in range(len(purity_point)):
    #         point.append(np.array(purity_point[i]))
    #     purity_point = np.array(point)
    #     point.clear()
    #     # print(purity_point)
    #     # purity = get_point_line_distance(list(purity_point), d, dis_purity=0.35)
    #     # after_ir = after_sampling(purity, ir)
    #     after_ir = resample(list(purity_point), ir, 0.05)
    #
    #     # for h in after_ir:
    #     #     if h not in total_point:
    #     #         total_point.append(h)
    #     total_point.extend(after_ir)
    #     total_point = list(set([tuple(t) for t in total_point]))
    #     # print("total_point", len(total_point))
    #     z += u
    #     # for t in purity:
    #     #     if t in numberMore:
    #     #         purity_most.append(t)
    #     # print("len(purity_most)", len(purity_most))
    #     # purity_value = len(purity_most) / len(purity_point)
    #
    # total_point = after_sampling(list(total_point), ir)
    #
    # su_data = np.concatenate((pca_more, pca_less), axis=0)
    # suu = copy.deepcopy(su_data)
    # suu = suu.tolist()
    # suu = list(suu)
    # for lp in range(len(su_data)):
    #     suu[lp].append(y_original[lp])
    #
    # result_sample = np.concatenate((total_point, pca_less), axis=0)
    #
    # result_sample = result_sample.tolist()
    # result_sample = list(result_sample)
    #
    # for g in range(len(result_sample)):
    #     for u in range(len(suu)):
    #         if suu[u][0] == result_sample[g][0] and suu[u][1] == result_sample[g][1]:
    #             result_sample[g].append(np.array(suu)[:, -1][u])
    #         else:
    #             u = u + 1
    #
    # # for g in range(len(result_sample)):
    # #     for u in range(len(pca_less)):
    # #         if pca_less[u][0] == result_sample[g][0] and pca_less[u][1] == result_sample[g][1]:
    # #             result_sample[g].append(1)
    # #         else:
    # #             u = u + 1
    # #
    # # for r in range(len(result_sample)):
    # #     for v in range(len(pca_more)):
    # #         if pca_more[v][0] == result_sample[r][0] and pca_more[v][1] == result_sample[r][1]:
    # #             result_sample[r].append(-1)
    # #         else:
    # #             v = v + 1
    # final_x_train = []
    # final_y_label = []
    #
    # # q = 0
    # # q = int(q)
    #
    # for q in range(len(result_sample)):
    #     final_y_label.append(np.array(result_sample)[q][2])
    #     final_x_train.append(np.array(result_sample)[q][0:len(result_sample[0]) - 1])
    #
    # # print(final_x_train)
    # # print(final_y_label)
    #
    # # fitClass(final_x_train, final_y_label)
    #
    #
    #
    #
    # # z = 0
    # # purity_point = []
    # # purity_most = []
    # # purity = []
    # # total_point = []
    # # after_ir = []
    # # u = np.array(numberLess).shape[0]
    # # while z < np.array(numberMore).shape[0]:
    # #     purity_most.clear()
    # #     purity.clear()
    # #     purity_point = random.sample(list(pca_more), u)
    # #     for i in range(len(purity_point)):
    # #         point.append(np.array(purity_point[i]))
    # #     purity_point = np.array(point)
    # #     point.clear()
    # #     # print(purity_point)
    # #     purity = get_point_line_distance(list(purity_point), d, dis_purity=0.35)
    # #     # after_ir = after_sampling(purity, ir)
    # #
    # #     for h in after_ir:
    # #         if h not in total_point:
    # #             total_point.append(h)
    # #     z += u
    # #     for t in purity:
    # #         if t in numberMore:
    # #             purity_most.append(t)
    # #     # print("len(purity_most)", len(purity_most))
    # #     purity_value = len(purity_most)/len(purity_point)
    # #
    # #
    # # result_sample = np.concatenate((total_point, pca_less), axis=0)
    # # # print(result_sample)
    # #
    # # result_sample = result_sample.tolist()
    # # result_sample = list(result_sample)
    # # # print(result_sample)
    # #
    # # for g in range(len(result_sample)):
    # #     for u in range(len(pca_less)):
    # #         if pca_less[u][0] == result_sample[g][0] and pca_less[u][1] == result_sample[g][1]:
    # #             result_sample[g].append(1)
    # #         else:
    # #             u = u + 1
    # #
    # # for r in range(len(result_sample)):
    # #     for v in range(len(pca_more)):
    # #         if pca_more[v][0] == result_sample[r][0] and pca_more[v][1] == result_sample[r][1]:
    # #             result_sample[r].append(-1)
    # #         else:
    # #             v = v + 1
    # #
    # #
    # # # print(result_sample)
    # # final_x_train = []
    # # final_y_label = []
    # #
    # # # q = 0
    # # # q = int(q)
    # #
    # # for q in range(len(result_sample)):
    # #     final_y_label.append(np.array(result_sample)[q][2])
    # #     final_x_train.append(np.array(result_sample)[q][0:len(result_sample[0])-1])
    #
    #
    #
    #
    # classifier = DecisionTreeClassifier()
    # fpr, tpr, thersholds = roc_curve(Y_test, classifier.fit(X_train, Y_train).predict(XTest), pos_label=1)
    #
    # # for i, value in enumerate(thersholds):
    # #     print("%f %f %f" % (fpr[i], tpr[i], value))
    #
    # roc_auc = auc(fpr, tpr)
    #
    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # # plt.show()
    # #
    # # plt_Data(np.array(purity), random_train_y)
    # # plt.show()
    #
    # # if len(final_y_label) < len(y_original):
    # #     y_original = random.sample(list(y_original), len(final_y_label))
    #
    #
    # if roc_auc > 0.5:
    #     # plt.savefig('purity1.png')
    #     fitClass(final_x_train, final_y_label)

    # if len(final_y_label) < len(y_original):
    #     y_original = random.sample(list(y_original), len(final_y_label))
    #
    # score = accuracy_score(y_original, final_y_label)

    # fitClass(np.array(final_x_train), final_y_label)

    # with open("te.txt", "a") as f:
    #     f.write(str(fitClass(purity, random_train_y, X_test, Y_test, 'DT')))
    #     f.write('\n')
    # time.sleep(5)

    # fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(Y_test, classifier.fit(purity, random_train_y).predict(X_test))
    #
    # plt.figure(1)
    # plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    # plt.plot([0, 1], [0, 1], linestyle="--")
    # plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    # plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
    # plt.title("ROC-AUC")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend()
    # plt.show()
