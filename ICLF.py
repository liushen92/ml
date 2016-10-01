#!/usr/bin/env python
# encoding: utf-8

import sys
import json
import time
import math
import random
import pickle
import threading
import numpy as np

random.seed(1024)

class SparseMatrix:

    kv_dict = None
    rows = None
    cols = None
    _sum = 0.0
    mean = 0.0

    def __init__(self, rows = None, cols = None):
        self.kv_dict = dict()

        if rows == None:
            self.rows = dict()
        else:
            self.rows = dict(rows)

        if cols == None:
            self.cols = dict()
            self.add_col = True
        else:
            self.cols = dict(cols)
            self.add_col = False

        self._sum = 0.0
        self.mean = 0.0

    def __repr__(self):
        ret_str = ''
        ret_str += '[INFO]: kv_dict size: %d mb\n' % (sys.getsizeof(self.kv_dict) / (1024 * 1024))
        ret_str += '[INFO]: rows size: %d\n' % len(self.rows)
        ret_str += '[INFO]: cols size: %d\n' % len(self.cols)
        ret_str += '[INFO]: #sparse kv_dict: %d\n' % len(self.kv_dict)
        ret_str += '[INFO]: mean: %f\n' % self.mean
        ret_str += '[INFO]: sum: %f\n' % self._sum
        return ret_str

    def __len__(self):
        return len(self.kv_dict)

    def __getitem__(self, index):
        r_index, c_index = index

        if not r_index in self.rows or not c_index in self.cols:
            return 0.0

        if not (self.rows[r_index], self.cols[c_index]) in self.kv_dict:
            return 0.0

        return 1.0 * self.kv_dict[self.rows[r_index], self.cols[c_index]]

    def __setitem__(self, index, value):
        r_index, c_index = index

        if not c_index in self.cols:
            if not self.add_col:
                return
            self.cols[c_index] = len(self.cols)
        col = self.cols[c_index]

        if not r_index in self.rows:
            self.rows[r_index] = len(self.rows)
        row = self.rows[r_index]

        if not (row, col) in self.kv_dict:
            self._sum += value
        else:
            self._sum += value - self.kv_dict[row, col]

        self.kv_dict[row, col] = value

        self.mean = 1.0 * self._sum / len(self.kv_dict) if len(self.kv_dict) != 0 else 0.0

    def toFullMatrix(self):
        m = np.zeros([len(self.rows), len(self.cols)])
        for u in range(len(self.rows)):
            for i in range(len(self.cols)):
                if not (u, i) in self.kv_dict:
                    m[u, i] = 0
                else:
                    m[u, i] = self.kv_dict[u, i]
        return m

class ParallelSGD(threading.Thread):

    def __init__(self, name, model):
        threading.Thread.__init__(self)
        self.name = name
        self.model = model

    def run(self):

        while True:
            self.model.thread_lock.acquire()

            if self.model.current_sample >= len(self.model.L_train):
                self.model.thread_lock.release()
                break
            (u, i), r = self.model.L_train[self.model.current_sample]
            self.model.current_sample += 1

            I_i = self.model.item_category_matrix[i]
            Pu_hat = self.model.P_hat[u]
            Qi_hat = self.model.Q_hat[i]
            bu = self.model.bu[u]
            bi = self.model.bi[i]

            self.model.thread_lock.release()

            r_pred = self.model.mu + bu + bi + (Pu_hat * I_i).T.dot(Qi_hat * I_i)
            # note to replace it with sigmoid transformation
            err = r_pred - r

            Pu_hat -= self.model._learning_rate * (err * Qi_hat * I_i + self.model._lambda_c * Pu_hat)
            Qi_hat -= self.model._learning_rate * (err * Pu_hat * I_i + self.model._lambda_c * Qi_hat)

            bu -= self.model._learning_rate * (err + self.model._lambda_bias * bu)
            bi -= self.model._learning_rate * (err + self.model._lambda_bias * bi)

            self.model.thread_lock.acquire()


            self.model.P_hat[u] = Pu_hat
            self.model.Q_hat[i] = Qi_hat
            self.model.bu[u] = bu
            self.model.bi[i] = bi

            self.model.sqr_err += err ** 2

            self.model.thread_lock.release()
class ICLF:

    def __init__(self, num_factor = 50, _lambda_bias = 0.01, _lambda_c = 0.01, _lambda = 0.01, _beta = 0.01, _learning_rate = 0.01, max_iter1 = 20, max_iter2 = 10, validate = 1, num_thread = 10):
        self.P = None
        self.Q = None
        self.C = None
        self.P_hat = None
        self.Q_hat = None
        self.bu = None
        self.bi = None
        self.mu = 0.0
        self.num_category = 0
        self.item_category_matrix = None
        self.user_category_matrix = None

        self.rows = None
        self.cols = None

        self.num_factor = num_factor
        self._lambda_bias = _lambda_bias
        self._lambda_c = _lambda_c
        self._lambda = _lambda
        self._beta = _beta
        self._learning_rate = _learning_rate
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.validate = validate
        self.num_thread = num_thread
        self.sqr_err = 0.0

        self.L_train = None
        self.L_validate = None
        self.current_sample = 0
        self.thread_lock = threading.Lock()

    def train(self, ratings, item_category_matrix, model_path):
        self.mu = ratings.mean
        self.item_category_matrix = item_category_matrix
        tmp, self.num_category = np.shape(item_category_matrix)

        self.P = 0.001 * np.random.randn(len(ratings.rows), self.num_factor)
        self.Q = 0.001 * np.random.randn(len(ratings.cols), self.num_factor)
        self.C = 0.001 * np.random.randn(self.num_category, self.num_factor)
        self.P_hat = 0.001 * np.random.randn(len(ratings.rows), self.num_category)
        self.Q_hat = 0.001 * np.random.randn(len(ratings.cols), self.num_category)
        self.bu = 0.001 * np.random.randn(len(ratings.rows))
        self.bi = 0.001 * np.random.randn(len(ratings.cols))

        self.rows = dict(ratings.rows)
        self.cols = dict(ratings.cols)

        if self.validate > 0:
            T = ratings.kv_dict.items()
            random.shuffle(T)
            k = int(len(T) * 0.3)
            self.L_validate = T[0:k]
            self.L_train = T[k:]
        else:
            self.L_train = ratings.kv_dict.items()

        self.user_category_matrix = np.zeros([len(self.rows), self.num_category])
        for (u, i), r in self.L_train:
            category = self.item_category_matrix[i]
            self.user_category_matrix[u] = np.logical_or(category, self.user_category_matrix[u])

        a = [sum(self.user_category_matrix[:,g]) for g in range(self.num_category)]

        rmse_train = [0.0] * self.max_iter1
        rmse_validate = [0.0] * self.max_iter1
        
        #learning phase1
        sys.stderr.write('phase1:' )
        for s in range(self.max_iter1):
            random.shuffle(self.L_train)

            self.current_sample = 0
            self.sqr_err = 0.0
            self.threads = [ParallelSGD('Thread_%d' % n, self) for n in range(self.num_thread)]

            start = time.time()
            for t in self.threads:
                t.start()
                t.join()
            terminal = time.time()
            duration = terminal - start

            rmse_train[s] = math.sqrt(self.sqr_err / len(ratings.kv_dict))

            if self.validate > 0:
                m = SparseMatrix()
                m.kv_dict = {k : v for (k, v) in self.L_validate}
                rmse_validate[s] = float(self.test_phase1(m))

            sys.stderr.write('Iter: %4.4i' % (s + 1))
            sys.stderr.write('\t[Train RMSE] = %f' % rmse_train[s])
            if self.validate > 0:
                sys.stderr.write('\t[Validate RMSE] = %f' % rmse_validate[s])
            sys.stderr.write('\t[Duration] = %f' % duration)
            sys.stderr.write('\t[Samples] = %d\n' % len(self.L_train))
        #calculate S (the paper says using user-category matrix to calculate S,but I can't figure out how to do that. Here I'm using item-category to calculate S)
        #S = {(g, k): sum(np.logical_and(self.item_category_matrix[:,g], self.item_category_matrix[:,k])) / float(sum(self.item_category_matrix[:,g])) for g in range(self.num_category) for k in range(self.num_category) if g != k}
        S = {(g, k): sum(np.logical_and(self.user_category_matrix[:,g], self.user_category_matrix[:,k])) / float(sum(self.user_category_matrix[:,g])) for g in range(self.num_category) for k in range(self.num_category) if g != k}
        #learning phase2
        sys.stderr.write('phase2:' )
        trainM = SparseMatrix()
        trainM.kv_dict = {k : v for (k, v) in self.L_train}

        for s in range(self.max_iter2):
            start = time.time()
            for u in range(len(self.rows)):
                self.P[u] = np.linalg.solve(self.C.T.dot(self.C) + self._lambda * np.eye(self.num_factor), self.C.T.dot(self.P_hat[u].T))
            for i in range(len(self.cols)):
                self.Q[i] = np.linalg.solve(self.C.T.dot(self.C) + self._lambda * np.eye(self.num_factor), self.C.T.dot(self.Q_hat[i].T))
            for k in range(self.num_category):
                self.C[k] = np.linalg.solve(self.P.T.dot(self.P) + self.Q.T.dot(self.Q) + (self._beta * (sum([S[g, k] for g in range(self.num_category) if g != k]) + sum([S[k, h] for h in range(self.num_category) if h != k]))+ self._lambda) * np.eye(self.num_factor), self.P.T.dot(self.P_hat[:, k]) + self.Q.T.dot(self.Q_hat[:, k]) + self._beta * (sum([S[g, k] * self.C[g] for g in range(self.num_category) if g != k]) + sum([S[k, h] * self.C[h] for h in range(self.num_category) if h != k])))

            terminal = time.time()
            duration = terminal - start

            rmse_train = [0.0] * self.max_iter2
            rmse_validate = [0.0] * self.max_iter2

            rmse_train[s] = float(self.test_phase2(trainM))

            if self.validate > 0:
                m = SparseMatrix()
                m.kv_dict = {k : v for (k, v) in self.L_validate}
                rmse_validate[s] = float(self.test_phase2(m))

            sys.stderr.write('Iter: %4.4i' % (s + 1))
            sys.stderr.write('\t[Train RMSE] = %f' % rmse_train[s])
            if self.validate > 0:
                sys.stderr.write('\t[Validate RMSE] = %f' % rmse_validate[s])
            sys.stderr.write('\t[Duration] = %f' % duration)
            sys.stderr.write('\t[Samples] = %d\n' % len(self.L_train))

    def test_phase1(self, ratings):
        r_pred = np.zeros(len(ratings))
        r = np.zeros(len(ratings))
        tmp = 0
        for s, (u, i) in enumerate(ratings.kv_dict):
            if u >= len(self.rows) or i >= len(self.cols):
                tmp += 1
                continue
            r[s - tmp] = ratings.kv_dict[u, i]
            r_pred[s - tmp] = (self.P_hat[u] * self.item_category_matrix[i]).T.dot(self.Q_hat[i] * self.item_category_matrix[i]) + self.bi[i] + self.bu[u] + self.mu

        err = r - r_pred
        rmse = math.sqrt(sum(err ** 2) / (len(ratings) - tmp))

        return rmse

    def test_phase2(self, ratings):
        r_pred = np.zeros(len(ratings))
        r = np.zeros(len(ratings))
        tmp = 0
        PC = self.P.dot(self.C.T)
        QC = self.Q.dot(self.C.T)
        for s, (u, i) in enumerate(ratings.kv_dict):
            if u >= len(self.rows) or i >= len(self.cols):
                tmp += 1
                continue
            r[s - tmp] = ratings.kv_dict[u, i]
            r_pred[s - tmp] = (PC[u] * self.item_category_matrix[i]).T.dot(QC[i] * self.item_category_matrix[i]) + self.bi[i] + self.bu[u] + self.mu

        err = r - r_pred
        rmse = math.sqrt(sum(err ** 2) / (len(ratings) - tmp))

        return rmse

def read_sparse_matrix(fp_data, rows = None, cols = None):
    m = SparseMatrix(rows, cols)
    for line in fp_data:
        user_id, item_id, rating, timestamp = line.strip().split('::')
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)
        m[user_id, item_id] = rating
    return m

def read_category(fp_data, rows = None, cols = None):
    m = SparseMatrix(rows, cols)
    category_dict = dict()
    for line in fp_data:
        item_id, item_name, category_name_list = line.strip().split('::')
        item_id = int(item_id)
        for category_name in category_name_list.strip().split('|'):
            if not category_name in category_dict.keys():
                category_dict[category_name] = len(category_dict)
            m[item_id, category_dict[category_name]] = 1
    return m

def read_category2(fp_data, rows = None, cols = None):
    m = SparseMatrix(rows, cols)
    category_dict = dict()
    for line in fp_data:
        tmp = line.strip().split('|')
        item_id = int(tmp[0])
        for category in range(1, 20):
            if int(tmp[-category]) == 1:
                m[item_id, category - 1] = 1
    return m



if __name__ == '__main__':

    #train_data = 'movielens.1m.train'
    #category_data = 'movielens.1m.index'

    #item_category = read_category(open(category_data))

    train_data = 'movielens.100k.train'
    category_data = 'movielens.100k.index'

    item_category = read_category2(open(category_data))

    train_ratings = read_sparse_matrix(open(train_data), cols = item_category.rows)
    item_category_matrix = item_category.toFullMatrix()

    print >> sys.stderr, train_ratings

    iclf = ICLF()
    iclf.train(train_ratings, item_category_matrix,'mf_model')

    #rmse_train = iclf.test_phase2(train_ratings)
    #rmse_test = mclf.test(test_ratings)

    #print >> sys.stderr, 'Training RMSE for MovieLens 1m dataset: %lf' % rmse_train
    #print >> sys.stderr, 'Test RMSE for MovieLens 1m dataset: %lf' % rmse_test
