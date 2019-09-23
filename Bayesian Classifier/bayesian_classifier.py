import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import ListedColormap

class BayesClassifier_Case1(object):
    
    def __init__(self, data_prep, train_size=0.70):
        self.data_preprocess = data_prep
        self.train_c1, self.test_c1 = data_prep.train_test_split(1, train_size=train_size)
        self.train_c2, self.test_c2 = data_prep.train_test_split(2, train_size=train_size)
        self.train_c3, self.test_c3 = data_prep.train_test_split(3, train_size=train_size)
        self.train = {1: self.train_c1, 2: self.train_c2, 3: self.train_c3}
        self.test = {1: self.test_c1, 2: self.test_c2, 3: self.test_c3}
        
        self.mu = []
        self.mu.append(np.array([self.train_c1.iloc[:, 0].mean(), self.train_c1.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c2.iloc[:, 0].mean(), self.train_c2.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c3.iloc[:, 0].mean(), self.train_c3.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))

        self.sigma_c1 = np.array(self.train_c1.cov(), dtype='float32')
        self.sigma_c2 = np.array(self.train_c2.cov(), dtype='float32')    
        self.sigma_c3 = np.array(self.train_c3.cov(), dtype='float32')

        self.sigma = (self.sigma_c1 + self.sigma_c2 + self.sigma_c3) / 3
        self.variance = (self.sigma[0, 0] + self.sigma[1, 1]) / 2

    def discriminative_func(self, x, i):
        prior = self.data_preprocess.prior_calculation()
        w = self.mu[i-1] / self.variance
        w0 = (-1 / (2 * self.variance)) * np.matmul(np.transpose(self.mu[i-1]), self.mu[i-1]) + np.log(prior[i-1])
        g = np.matmul(np.transpose(w), x) + np.asscalar(w0)
        return np.asscalar(g)

    def plot_contour(self, x_min, x_max, y_min, y_max, steps, pair_wise=None, train_or_test=0, contour=None, smoothness=0.5):
        colors_bg = ['#8ACCC3', '#F39C91', '#F4E874']
        colors_pt = ['b', 'r', 'g']
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        x = np.linspace(x_min, x_max, steps, dtype='float32')
        y = np.linspace(y_min, y_max, steps, dtype='float32')
        g = np.meshgrid(x, y)
        X1, X2 = np.meshgrid(x, y)
        z = np.array([])
        count = 1

        for mesh_point in zip(*(x.flat for x in g)):
            print(count)
            point = np.array(mesh_point).reshape((2, 1))
            if not pair_wise:
                predicted_class = self.predict_class(point)
            if pair_wise:
                predicted_class = self.predict_class(point, pair_wise)
            z = np.append(z, predicted_class)
            count += 1
    
        Z = z.reshape(X1.shape)

        if not pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap(('#8ACCC3', '#F39C91', '#F4E874')))
            if contour:
                ax = self.data_preprocess.class_contour_lines(1, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(2, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(3, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[1]["X_c" + str(1)], self.train[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.train[2]["X_c" + str(2)], self.train[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.train[3]["X_c" + str(3)], self.train[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective training data.", fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[1]["X_c" + str(1)], self.test[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.test[2]["X_c" + str(2)], self.test[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.test[3]["X_c" + str(3)], self.test[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective test data.", fontsize=20)

        if pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap((colors_bg[pair_wise[0]-1], colors_bg[pair_wise[1]-1])))
            if contour:
                ax = self.data_preprocess.class_contour_lines(pair_wise[0], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(pair_wise[1], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[pair_wise[0]]["X_c" + str(pair_wise[0])], self.train[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.train[pair_wise[1]]["X_c" + str(pair_wise[1])], self.train[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective training data points.".format(pair_wise[0], pair_wise[1]), fontsize=15)
            if train_or_test == 1:
                ax.scatter(self.test[pair_wise[0]]["X_c" + str(pair_wise[0])], self.test[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.test[pair_wise[1]]["X_c" + str(pair_wise[1])], self.test[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective test data points.".format(pair_wise[0], pair_wise[1]), fontsize=15)

        plt.legend()
        ax.set_xlabel("Feature_1", fontsize=10)
        ax.set_ylabel("Feature_2", fontsize=10)
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()

    def predict_class(self, p, pair_wise=None):
        temp = []
        point = p
        if not pair_wise:
            g_i = self.discriminative_func(point, 1)
            g_j = self.discriminative_func(point, 2)
            g_k = self.discriminative_func(point, 3)
            temp.append(g_i)
            temp.append(g_j)
            temp.append(g_k)
            index = temp.index(max(temp))
            return index+1
        if pair_wise:
            g_i = self.discriminative_func(point, pair_wise[0])
            g_j = self.discriminative_func(point, pair_wise[1])
            temp.append(g_i)
            temp.append(g_j)
            index = temp.index(max(temp))
            return index+1

"""##########################################################################################################"""

class BayesClassifier_Case2(object):
    
    def __init__(self, data_prep, train_size=0.70):
        self.data_preprocess = data_prep
        self.train_c1, self.test_c1 = data_prep.train_test_split(1, train_size=train_size)
        self.train_c2, self.test_c2 = data_prep.train_test_split(2, train_size=train_size)
        self.train_c3, self.test_c3 = data_prep.train_test_split(3, train_size=train_size)
        self.train = {1: self.train_c1, 2: self.train_c2, 3: self.train_c3}
        self.test = {1: self.test_c1, 2: self.test_c2, 3: self.test_c3}
        
        self.mu = []
        self.mu.append(np.array([self.train_c1.iloc[:, 0].mean(), self.train_c1.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c2.iloc[:, 0].mean(), self.train_c2.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c3.iloc[:, 0].mean(), self.train_c3.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))

        self.sigma_c1 = np.array(self.train_c1.cov(), dtype='float32')
        self.sigma_c2 = np.array(self.train_c2.cov(), dtype='float32')    
        self.sigma_c3 = np.array(self.train_c3.cov(), dtype='float32')

        self.sigma = (self.sigma_c1 + self.sigma_c2 + self.sigma_c3) / 3

    def discriminative_func(self, x, i):
        prior = self.data_preprocess.prior_calculation()
        w =  np.matmul(np.linalg.inv(self.sigma), self.mu[i-1])
        w0 = -0.5 * np.linalg.multi_dot([np.transpose(self.mu[i-1]), np.linalg.inv(self.sigma), self.mu[i-1]]) + np.log(prior[i-1])
        g = np.matmul(np.transpose(w), x) + np.asscalar(w0)
        return np.asscalar(g)

    def plot_contour(self, x_min, x_max, y_min, y_max, steps, pair_wise=None, train_or_test=0, contour=None, smoothness=0.5):
        colors_bg = ['#8ACCC3', '#F39C91', '#F4E874']
        colors_pt = ['b', 'r', 'g']
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        x = np.linspace(x_min, x_max, steps, dtype='float32')
        y = np.linspace(y_min, y_max, steps, dtype='float32')
        g = np.meshgrid(x, y)
        X1, X2 = np.meshgrid(x, y)
        z = np.array([])
        count = 1

        for mesh_point in zip(*(x.flat for x in g)):
            print(count)
            point = np.array(mesh_point).reshape((2, 1))
            if not pair_wise:
                predicted_class = self.predict_class(point)
            if pair_wise:
                predicted_class = self.predict_class(point, pair_wise)
            z = np.append(z, predicted_class)
            count += 1
    
        Z = z.reshape(X1.shape)

        if not pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap(('#8ACCC3', '#F39C91', '#F4E874')))
            if contour:
                ax = self.data_preprocess.class_contour_lines(1, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(2, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(3, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[1]["X_c" + str(1)], self.train[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.train[2]["X_c" + str(2)], self.train[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.train[3]["X_c" + str(3)], self.train[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective training data.", fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[1]["X_c" + str(1)], self.test[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.test[2]["X_c" + str(2)], self.test[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.test[3]["X_c" + str(3)], self.test[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective test data.", fontsize=20)

        if pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap((colors_bg[pair_wise[0]-1], colors_bg[pair_wise[1]-1])))
            if contour:
                ax = self.data_preprocess.class_contour_lines(pair_wise[0], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(pair_wise[1], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[pair_wise[0]]["X_c" + str(pair_wise[0])], self.train[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.train[pair_wise[1]]["X_c" + str(pair_wise[1])], self.train[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective training data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[pair_wise[0]]["X_c" + str(pair_wise[0])], self.test[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.test[pair_wise[1]]["X_c" + str(pair_wise[1])], self.test[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective test data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)

        plt.legend()
        ax.set_xlabel("Feature_1", fontsize=10)
        ax.set_ylabel("Feature_2", fontsize=10)
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    def predict_class(self, p, pair_wise=None):
        temp = []
        point = p
        if not pair_wise:
            g_i = self.discriminative_func(point, 1)
            g_j = self.discriminative_func(point, 2)
            g_k = self.discriminative_func(point, 3)
            temp.append(g_i)
            temp.append(g_j)
            temp.append(g_k)
            index = temp.index(max(temp))
            return index+1
        if pair_wise:
            g_i = self.discriminative_func(point, pair_wise[0])
            g_j = self.discriminative_func(point, pair_wise[1])
            temp.append(g_i)
            temp.append(g_j)
            index = temp.index(max(temp))
            return index+1

"""##########################################################################################################"""

class BayesClassifier_Case3(object):

    def __init__(self, data_prep, train_size=0.70):
        self.data_preprocess = data_prep
        self.train_c1, self.test_c1 = data_prep.train_test_split(1, train_size=train_size)
        self.train_c2, self.test_c2 = data_prep.train_test_split(2, train_size=train_size)
        self.train_c3, self.test_c3 = data_prep.train_test_split(3, train_size=train_size)
        self.train = {1: self.train_c1, 2: self.train_c2, 3: self.train_c3}
        self.test = {1: self.test_c1, 2: self.test_c2, 3: self.test_c3}
        
        self.mu = []
        self.mu.append(np.array([self.train_c1.iloc[:, 0].mean(), self.train_c1.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c2.iloc[:, 0].mean(), self.train_c2.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c3.iloc[:, 0].mean(), self.train_c3.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))

        self.sigma_c1 = np.array(self.train_c1.cov(), dtype='float32')
        self.sigma_c2 = np.array(self.train_c2.cov(), dtype='float32')    
        self.sigma_c3 = np.array(self.train_c3.cov(), dtype='float32')

        # Diagonaloze the sigmas
        self.sigma_c1 = np.diag(np.diagonal(self.sigma_c1))
        self.sigma_c2 = np.diag(np.diagonal(self.sigma_c2))
        self.sigma_c3 = np.diag(np.diagonal(self.sigma_c3))

        self.sigma = {1: self.sigma_c1, 2: self.sigma_c2, 3: self.sigma_c3}

    def discriminative_func(self, x, i):
        prior = self.data_preprocess.prior_calculation()
        W_i = -0.5 * np.linalg.inv(self.sigma[i])
        w_i = np.matmul(np.linalg.inv(self.sigma[i]), self.mu[i-1])
        w0_i = -0.5 * np.linalg.multi_dot([np.transpose(self.mu[i-1]), np.linalg.inv(self.sigma[i]), self.mu[i-1]]) - 0.5 * (np.log(np.linalg.det(self.sigma[i]))) + np.log(prior[i-1])
        g_i = np.linalg.multi_dot([np.transpose(x), W_i, x]) + np.matmul(np.transpose(w_i), x) + w0_i
        
        return np.asscalar(g_i)

    def plot_contour(self, x_min, x_max, y_min, y_max, steps, pair_wise=None, train_or_test=0, contour=None, smoothness=0.5):
        colors_bg = ['#8ACCC3', '#F39C91', '#F4E874']
        colors_pt = ['b', 'r', 'g']
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        x = np.linspace(x_min, x_max, steps, dtype='float32')
        y = np.linspace(y_min, y_max, steps, dtype='float32')
        g = np.meshgrid(x, y)
        X1, X2 = np.meshgrid(x, y)
        z = np.array([])
        count = 1

        for mesh_point in zip(*(x.flat for x in g)):
            print(count)
            point = np.array(mesh_point).reshape((2, 1))
            if not pair_wise:
                predicted_class = self.predict_class(point)
            if pair_wise:
                predicted_class = self.predict_class(point, pair_wise)
            z = np.append(z, predicted_class)
            count += 1
    
        Z = z.reshape(X1.shape)

        if not pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap(('#8ACCC3', '#F39C91', '#F4E874')))
            if contour:
                ax = self.data_preprocess.class_contour_lines(1, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(2, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(3, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[1]["X_c" + str(1)], self.train[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.train[2]["X_c" + str(2)], self.train[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.train[3]["X_c" + str(3)], self.train[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective training data.", fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[1]["X_c" + str(1)], self.test[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.test[2]["X_c" + str(2)], self.test[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.test[3]["X_c" + str(3)], self.test[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective test data.", fontsize=20)

        if pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap((colors_bg[pair_wise[0]-1], colors_bg[pair_wise[1]-1])))
            if contour:
                ax = self.data_preprocess.class_contour_lines(pair_wise[0], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(pair_wise[1], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[pair_wise[0]]["X_c" + str(pair_wise[0])], self.train[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.train[pair_wise[1]]["X_c" + str(pair_wise[1])], self.train[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective training data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[pair_wise[0]]["X_c" + str(pair_wise[0])], self.test[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.test[pair_wise[1]]["X_c" + str(pair_wise[1])], self.test[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective test data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)

        plt.legend()
        ax.set_xlabel("Feature_1", fontsize=10)
        ax.set_ylabel("Feature_2", fontsize=10)
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    def predict_class(self, p, pair_wise=None):
        temp = []
        point = p
        if not pair_wise:
            g_i = self.discriminative_func(point, 1)
            g_j = self.discriminative_func(point, 2)
            g_k = self.discriminative_func(point, 3)
            temp.append(g_i)
            temp.append(g_j)
            temp.append(g_k)
            index = temp.index(max(temp))
            return index+1
        if pair_wise:
            g_i = self.discriminative_func(point, pair_wise[0])
            g_j = self.discriminative_func(point, pair_wise[1])
            temp.append(g_i)
            temp.append(g_j)
            index = temp.index(max(temp))
            return index+1

"""##########################################################################################################"""

class BayesClassifier_Case4(object):

    def __init__(self, data_prep, train_size=0.70):
        self.data_preprocess = data_prep
        self.train_c1, self.test_c1 = data_prep.train_test_split(1, train_size=train_size)
        self.train_c2, self.test_c2 = data_prep.train_test_split(2, train_size=train_size)
        self.train_c3, self.test_c3 = data_prep.train_test_split(3, train_size=train_size)
        self.train = {1: self.train_c1, 2: self.train_c2, 3: self.train_c3}
        self.test = {1: self.test_c1, 2: self.test_c2, 3: self.test_c3}
        
        self.mu = []
        self.mu.append(np.array([self.train_c1.iloc[:, 0].mean(), self.train_c1.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c2.iloc[:, 0].mean(), self.train_c2.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))
        self.mu.append(np.array([self.train_c3.iloc[:, 0].mean(), self.train_c3.iloc[:, 1].mean()], dtype='float32').reshape((2, 1)))

        self.sigma_c1 = np.array(self.train_c1.cov(), dtype='float32')
        self.sigma_c2 = np.array(self.train_c2.cov(), dtype='float32')    
        self.sigma_c3 = np.array(self.train_c3.cov(), dtype='float32')

        self.sigma = {1: self.sigma_c1, 2: self.sigma_c2, 3: self.sigma_c3}

    def discriminative_func(self, x, i):
        prior = self.data_preprocess.prior_calculation()
        W_i = -0.5 * np.linalg.inv(self.sigma[i])
        w_i = np.matmul(np.linalg.inv(self.sigma[i]), self.mu[i-1])
        w0_i = -0.5 * np.linalg.multi_dot([np.transpose(self.mu[i-1]), np.linalg.inv(self.sigma[i]), self.mu[i-1]]) - 0.5 * (np.log(np.linalg.det(self.sigma[i]))) + np.log(prior[i-1])
        g_i = np.linalg.multi_dot([np.transpose(x), W_i, x]) + np.matmul(np.transpose(w_i), x) + w0_i
        
        return np.asscalar(g_i)

    def plot_contour(self, x_min, x_max, y_min, y_max, steps, pair_wise=None, train_or_test=0, contour=None, smoothness=0.5):
        colors_bg = ['#8ACCC3', '#F39C91', '#F4E874']
        colors_pt = ['b', 'r', 'g']
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        x = np.linspace(x_min, x_max, steps, dtype='float32')
        y = np.linspace(y_min, y_max, steps, dtype='float32')
        g = np.meshgrid(x, y)
        X1, X2 = np.meshgrid(x, y)
        z = np.array([])
        count = 1

        for mesh_point in zip(*(x.flat for x in g)):
            print(count)
            point = np.array(mesh_point).reshape((2, 1))
            if not pair_wise:
                predicted_class = self.predict_class(point)
            if pair_wise:
                predicted_class = self.predict_class(point, pair_wise)
            z = np.append(z, predicted_class)
            count += 1
    
        Z = z.reshape(X1.shape)

        if not pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap(('#8ACCC3', '#F39C91', '#F4E874')))
            if contour:
                ax = self.data_preprocess.class_contour_lines(1, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(2, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(3, smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[1]["X_c" + str(1)], self.train[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.train[2]["X_c" + str(2)], self.train[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.train[3]["X_c" + str(3)], self.train[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective training data.", fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[1]["X_c" + str(1)], self.test[1]["Y_c" + str(1)], c=colors_pt[0], label='Class ' + str(1), alpha=0.70)
                ax.scatter(self.test[2]["X_c" + str(2)], self.test[2]["Y_c" + str(2)], c=colors_pt[1], label='Class ' + str(2), alpha=0.70)
                ax.scatter(self.test[3]["X_c" + str(3)], self.test[3]["Y_c" + str(3)], c=colors_pt[2], label='Class ' + str(3), alpha=0.70)
                ax.set_title("Decision Boundaries of each class with respective test data.", fontsize=20)

        if pair_wise:
            plt.contourf(X1, X2, Z, alpha = 0.50, cmap = ListedColormap((colors_bg[pair_wise[0]-1], colors_bg[pair_wise[1]-1])))
            if contour:
                ax = self.data_preprocess.class_contour_lines(pair_wise[0], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
                ax = self.data_preprocess.class_contour_lines(pair_wise[1], smoothness=smoothness, filled=None, all=False, plot_data_point=False, ax_pass=ax, fig_pass=fig, contour_label=False)
            if train_or_test == 0:
                ax.scatter(self.train[pair_wise[0]]["X_c" + str(pair_wise[0])], self.train[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.train[pair_wise[1]]["X_c" + str(pair_wise[1])], self.train[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective training data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)
            if train_or_test == 1:
                ax.scatter(self.test[pair_wise[0]]["X_c" + str(pair_wise[0])], self.test[pair_wise[0]]["Y_c" + str(pair_wise[0])], c=colors_pt[pair_wise[0]-1], label='Class ' + str(pair_wise[0]), alpha=0.70)
                ax.scatter(self.test[pair_wise[1]]["X_c" + str(pair_wise[1])], self.test[pair_wise[1]]["Y_c" + str(pair_wise[1])], c=colors_pt[pair_wise[1]-1], label='Class ' + str(pair_wise[1]), alpha=0.70)
                ax.set_title("Decision Boundaries between class {} and class {} with respective test data points.".format(pair_wise[0], pair_wise[1]), fontsize=20)

        plt.legend()
        ax.set_xlabel("Feature_1", fontsize=10)
        ax.set_ylabel("Feature_2", fontsize=10)
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    def predict_class(self, p, pair_wise=None):
        temp = []
        point = p
        if not pair_wise:
            g_i = self.discriminative_func(point, 1)
            g_j = self.discriminative_func(point, 2)
            g_k = self.discriminative_func(point, 3)
            temp.append(g_i)
            temp.append(g_j)
            temp.append(g_k)
            index = temp.index(max(temp))
            return index+1
        if pair_wise:
            g_i = self.discriminative_func(point, pair_wise[0])
            g_j = self.discriminative_func(point, pair_wise[1])
            temp.append(g_i)
            temp.append(g_j)
            index = temp.index(max(temp))
            return index+1