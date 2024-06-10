import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from regression import linear_scores, ridge_scores, lasso_scores, elasticNet_scores

"""
    构造降序排序矩阵
"""


def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(-matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n + 1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i, j]] == matrix[i, sorts[i, j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i, j]] != matrix[i, sorts[i, j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i, sorts[i, j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i, j]] = j + 1
                continue
    return matrix


"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""


def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * (k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result / (n * (k - 1) - result)
    return result


"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""


def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))


# 用法
def excel_one_line_to_list():
    df = pd.read_excel("D:\\final.xlsx", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
    df_li = df.values.tolist()
    df_li = np.array(df_li)
    return df_li


if __name__ == '__main__':
    data = excel_one_line_to_list()
    # print(data)
    # n:数据集 k：算法个数,data是csv格式，n行k列
    rank = rank_matrix(data)
    print(rank)
    Friedman = friedman(15, 8, rank)
    CD = nemenyi(15, 8, 3.031)
    h_CD = CD / 2
    rank_mean = rank.mean(axis=0)
    print(rank_mean)
    _alg_ = [rank_mean[0], rank_mean[1], rank_mean[2], rank_mean[3], rank_mean[4], rank_mean[5], rank_mean[6], rank_mean[7]]
    y = [1, 2, 3, 4, 5, 6, 7, 8]

    plt.figure(figsize=(15, 8))
    plt.scatter(_alg_, y, s=100, c='black')
    for i in range(len(y)):
        yy = [y[i], y[i]]
        xx = [_alg_[i] - h_CD, _alg_[i] + h_CD]
        plt.plot(xx, yy, linewidth=3.0)

    plt.yticks(range(0, 10, 1), labels=['', 'RSDS', 'ABSmote', 'S+G+A', 'SDUS1', 'RSMOTE', 'GBS', 'SPE', 'PCFS', ''], size=23)
    plt.xticks(range(0, 10, 1), labels=['', '1', '2', '3', '4', '5', '6', '7', '8', ''], size=30)

    # plt.xlabel("Algorithm", size=20)

    # plt.title("Friedman", size=40)

    plt.savefig("title" + '.png', format='PNG', dpi=600, bbox_inches='tight', pad_inches=+0.1)

    plt.show()




