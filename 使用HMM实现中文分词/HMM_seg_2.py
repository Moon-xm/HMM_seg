#! /usr/bin/env python
# -*-coding:utf-8-*-
# Author: mx
# create date: 2019-11-23 09:28:16
# description: 使用隐马尔科夫模型HMM实现中文分词

import numpy as np
import math
from tqdm import tqdm


class HMM_seg:
    def __init__(self):
        self.n_label = 4  # 定义标签个数 0：B,1：M,2：E,3：S
        self.n_char = 65535  # unicode编码的范围,65535表示所有字符
        self.negative_infinite = 1e-10  # 负无穷,偏置, 用来防止log(0)或乘0的情况
        self.pi = np.zeros(self.n_label)  # 创建初始隐概率矩阵(shape:4)
        self.A = np.zeros((self.n_label, self.n_label))  # 创建转移概率矩阵(shape:4*4)
        self.B = np.zeros((self.n_label, self.n_char))  # 创建混淆矩阵(发射概率矩阵)(shape:4*65535)

    def log_norm(self, arr):
        """
        函数说明： 用于将数组进行对数归一化

        Parameter：
        ----------
            arr - 要归一化的数组
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-23 16:48:09
        """
        sum_log = math.log(sum(arr))
        for num in range(len(arr)):
            if arr[num] == 0:
                arr[num] = self.negative_infinite
            else:
                arr[num] = math.log(arr[num]) - sum_log

    def save_lamda(self):
        """
        函数说明： 保存训练得到的参数A,B,pi

        Parameter：
        ----------
            None
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-25 20:30:20
        """
        np.savetxt('pi.out', self.pi)
        np.savetxt('A.out', self.A)
        np.savetxt('B.out', self.B)

    def fit(self, corpus_path):
        """
        函数说明： 将以空格分开的语料库训练得到模型参数,包括pi,A,B

        Parameter：
        ----------
            corpus_path - 语料库位置
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-23 16:44:28
        """
        with open(corpus_path, mode='r', encoding='utf-8') as f:
            data = f.read()
            corpus = data.split(' ')
        # print(corpus)
        print('开始训练模型：')
        for i in tqdm(range(len(corpus))):
            if len(corpus[i]) == 0:
                continue
            if len(corpus[i]) == 1:  # 3:S
                self.pi[3] += 1
                if len(corpus[i - 1]) == 1:
                    self.A[3][3] += 1  # 3:S -> 3:S
                else:
                    self.A[2][3] += 1  # 2:E -> 3:S
                self.B[3][ord(corpus[i])] += 1  # 3:S -> 字的Unicode编码
                continue
            self.pi[0] += 1  # 0:B
            if len(corpus[i - 1]) == 1:
                self.A[3][0] += 1  # 3:S -> 0:B
            else:
                self.A[2][0] += 1  # 2:E -> 0:B
            self.B[0][ord(corpus[i][0])] += 1  # 0:B -> 字的Unicode编码
            if len(corpus[i]) == 2:
                self.A[0][2] += 1  # 0:B -> 2:E
                self.B[2][ord(corpus[i][1])] += 1  # 2:E -> 字的Unicode编码
            else:
                self.A[0][1] += 1  # 0:B -> 1:M
                self.A[1][1] += (len(corpus[i]) - 3)  # 1:M -> 1:M
                self.A[1][2] += 1  # 1:M-> 2:E
                for m in range(1, len(corpus[i]) - 1):
                    self.B[1][ord(corpus[i][m])] += 1  # 1:M -> 字的Unicode编码
                self.B[2][ord(corpus[i][len(corpus[i]) - 1])] += 1  # 2:E -> 字的Unicode编码
        self.pi[self.pi == 0] = self.negative_infinite
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))
        self.A[self.A == 0] = self.negative_infinite
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.negative_infinite
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        self.save_lamda()
        print('训练完成！')
        # return self.A, self.B, self.pi

    def viterbi(self, text):
        """
        函数说明： 维特比算法实现中文编码

        Parameter：
        ----------
            text - 要用于分词的文本
        Return:
        -------
            list(reversed(decode)) - 返回文本编码结果
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-25 15:06:05
        """
        T = len(text)
        delta = np.zeros((T, 4))  # shape:要预测的文本长度*4
        path = np.zeros((T, 4))  # shape:要预测的文本长度*4
        for i in range(self.n_label):  # 初始化
            delta[0][i] = self.pi[i] + self.B[i][ord(text[0])]
        for t in range(1, T):  # 递推
            for j in range(self.n_label):
                delta[t][j] = delta[t - 1][0] + self.A[0][j]
                for i in range(1, self.n_label):  # 找出最大的delta[t][i]
                    max_delta = delta[t - 1][i] + self.A[i][j]
                    if max_delta > delta[t][j]:
                        delta[t][j] = max_delta
                        path[t][j] = i
                delta[t][j] = delta[t][j] + self.B[j][ord(text[t])]
        decode = np.full(T, -1)  # 回溯
        decode[T - 1] = np.argmax(delta[T - 1])
        for i in range(T - 2, -1, -1):
            decode[i] = path[i + 1][decode[i + 1]]
        return decode

    def predict(self, text_path):
        """
        函数说明： 使用维特比算法对文本分词

        Parameter：
        ----------
            text_path - 文本路径
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-25 15:05:56
        """

        with open(text_path, mode='r', encoding='utf-8') as fr:
            text = fr.read()
        decode = self.viterbi(text)
        fw = open('tokenize.txt', 'w')
        print('开始解码')
        for i in tqdm(range(len(text))):  # 解码
            if decode[i] == 0 or decode[i] == 1:  # 0：B, 1:M
                # print(text[i], end='')
                fw.write(text[i])
            else:  # 2:E, 3:S
                # print(text[i], ' ', end='')
                fw.write(text[i] + ' ')
        fw.close()
        # print('\n')
        print('解码完成！分词结果保存在tokenize.txt')


def main():
    model = HMM_seg()
    model.fit(corpus_path='corpus/peopleDailyCorpus.txt')
    model.predict('santi.txt')


if __name__ == '__main__':
    main()
