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
import numpy.ma as ma
import pdb

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

            k_set = self.model.item_category_matrix[i] == 1
            j_set = self.model.user_blanket[u]

            Puk = self.model.P[u, k_set]
            Qik = self.model.Q[i, k_set]
            bu = self.model.bu[u]
            bi = self.model.bi[i]

            self.model.thread_lock.release()
            r_pred = self.model.mu + bu + bi + np.diag(Puk.dot(Qik.T)).sum()
            # according to the paper, note to replace it with sigmoid transformation
            err = r_pred - r

            Puk -= self.model._learning_rate * (err * Qik + self.model._lambda * Puk)
            Qik -= self.model._learning_rate * (err * Puk + self.model._lambda * Qik + self.model._eta)
            
            #This part may take a lot of time
            '''for j in j_set:
                if j == i:
                    continue
                category = np.logical_and(k_set, self.model.item_category_matrix[j] == 1)
                tmp = np.zeros(np.shape(self.model.Q[j]))
                tmp[category] = self.model.Q[j, category]
                Qik -= self.model._learning_rate * self.model._alpha * (Qik - tmp[k_set])
            '''
            bu -= self.model._learning_rate * (err + self.model._lambda_bias * bu)
            bi -= self.model._learning_rate * (err + self.model._lambda_bias * bi)

            self.model.thread_lock.acquire()


            self.model.P[u, k_set] = Puk
            self.model.Q[i, k_set] = Qik
            self.model.bu[u] = bu
            self.model.bi[i] = bi

            self.model.sqr_err += err ** 2

            self.model.thread_lock.release()
class MCLF:

    def __init__(self, num_factor = 50, _lambda_bias = 0.001, _lambda = 0.001, _eta = 0.001, _alpha = 0.01, _learning_rate = 0.01, max_iter = 20, validate = 1, num_thread = 10):
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = 0.0
        self.num_category = 0
        self.item_category_matrix = None

        self.user_blanket = dict()
        self.rows = None
        self.cols = None
        self._alpha = _alpha

        self.num_factor = num_factor
        self._lambda_bias = _lambda_bias
        self._lambda = _lambda
        self._eta = _eta
        self._learning_rate = _learning_rate
        self.max_iter = max_iter
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
        self.P = 0.001 * np.random.randn(len(ratings.rows), self.num_category, self.num_factor)

        self.bu = 0.001 * np.random.randn(len(ratings.rows))
        self.Q = 0.001 * np.random.randn(len(ratings.cols), self.num_category, self.num_factor)
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

        for u in range(len(self.rows)):
            self.user_blanket[u] = list()

        for (u, i), r in self.L_train:
            self.user_blanket[u].append(i)

        rmse_train = [0.0] * self.max_iter
        rmse_validate = [0.0] * self.max_iter

        for s in range(self.max_iter):
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
                rmse_validate[s] = float(self.test(m))

            sys.stderr.write('Iter: %4.4i' % (s + 1))
            sys.stderr.write('\t[Train RMSE] = %f' % rmse_train[s])
            if self.validate > 0:
                sys.stderr.write('\t[Validate RMSE] = %f' % rmse_validate[s])
            sys.stderr.write('\t[Duration] = %f' % duration)
            sys.stderr.write('\t[Samples] = %d\n' % len(self.L_train))


    def test(self, ratings):
        r_pred = np.zeros(len(ratings))
        r = np.zeros(len(ratings))
        tmp = 0
        for s, (u, i) in enumerate(ratings.kv_dict):
            if u >= len(self.rows) or i >= len(self.cols):
                tmp += 1
                continue
            r[s - tmp] = ratings.kv_dict[u, i]
            k_set = self.item_category_matrix[i] == 1
            r_pred[s - tmp] = np.diag(self.P[u, k_set].dot(self.Q[i, k_set].T)).sum() + self.bi[i] + self.bu[u] + self.mu

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

    train_data = 'movielens.1m.train'
    category_data = 'movielens.1m.index'
    
    item_category = read_category(open(category_data))

    #train_data = 'movielens.100k.train'
    #category_data = 'movielens.100k.index'

    #item_category = read_category2(open(category_data))
    train_ratings = read_sparse_matrix(open(train_data), cols = item_category.rows)
    item_category_matrix = item_category.toFullMatrix()

    print >> sys.stderr, train_ratings

    mclf = MCLF()
    mclf.train(train_ratings, item_category_matrix,'mf_model')

    #rmse_train = mclf.test(train_ratings)
    #rmse_test = mclf.test(test_ratings)

    #print >> sys.stderr, 'Training RMSE for MovieLens 1m dataset: %lf' % rmse_train
    #print >> sys.stderr, 'Test RMSE for MovieLens 1m dataset: %lf' % rmse_test
