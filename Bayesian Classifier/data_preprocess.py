import numpy as np 
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D

class DataPreprocessing(object):

    def __init__(self, path_data_1, path_data_2, path_data_3, train_size=0.70):
        self.data_1, self.data_2, self.data_3 = self.read_data(path_data_1, path_data_2, path_data_3)

        self.data_min_x = min(self.data_1["X_c1"].min(), self.data_2["X_c2"].min(), self.data_3["X_c3"].min())
        self.data_min_y = min(self.data_1["Y_c1"].min(), self.data_2["Y_c2"].min(), self.data_3["Y_c3"].min())

        self.data_max_x = max(self.data_1["X_c1"].max(), self.data_2["X_c2"].max(), self.data_3["X_c3"].max())
        self.data_max_y = max(self.data_1["Y_c1"].max(), self.data_2["Y_c2"].max(), self.data_3["Y_c3"].max())

        self.data = {1: self.data_1, 2: self.data_2, 3: self.data_3}
        self.train_c1, self.test_c1 = self.train_test_split(1, train_size=train_size)
        self.train_c2, self.test_c2 = self.train_test_split(2, train_size=train_size)
        self.train_c3, self.test_c3 = self.train_test_split(3, train_size=train_size)

        self.train = {1: self.train_c1, 2: self.train_c2, 3: self.train_c3}

        self.X_c1_min, self.X_c1_max = self.train[1]["X_c1"].min(), self.train[1]["X_c1"].max()
        self.X_c2_min, self.X_c2_max = self.train[2]["X_c2"].min(), self.train[2]["X_c2"].max()
        self.X_c3_min, self.X_c3_max = self.train[3]["X_c3"].min(), self.train[3]["X_c3"].max()

        self.Y_c1_min, self.Y_c1_max = self.train[1]["Y_c1"].min(), self.train[1]["Y_c1"].max()
        self.Y_c2_min, self.Y_c2_max = self.train[2]["Y_c2"].min(), self.train[2]["Y_c2"].max()
        self.Y_c3_min, self.Y_c3_max = self.train[3]["Y_c3"].min(), self.train[3]["Y_c3"].max()

    def read_data(self, path_data_1, path_data_2, path_data_3):
        data1 = pd.read_csv(path_data_1, sep=' ', header=None, dtype='float32')
        data2 = pd.read_csv(path_data_2, sep=' ', header=None, dtype='float32')
        data3 = pd.read_csv(path_data_3, sep=' ', header=None, dtype='float32')

        if data1.columns.shape[0] > 2:
            data1.drop(data1.columns[-1], axis=1, inplace=True)
            data2.drop(data2.columns[-1], axis=1, inplace=True)
            data3.drop(data3.columns[-1], axis=1, inplace=True)

        data3.columns = ["X_c3", "Y_c3"]
        data2.columns = ["X_c2", "Y_c2"]
        data1.columns = ["X_c1", "Y_c1"]

        return data1, data2, data3

    def visualize_dataset(self, train_or_test_all=0, flag=None):
        if train_or_test_all == 0:
            data_1, data_2, data_3 = self.train_c1, self.train_c2, self.train_c3
        if train_or_test_all == 1:
            data_1, data_2, data_3 = self.test_c1, self.test_c2, self.test_c3
        if train_or_test_all == 2:
            data_1, data_2, data_3 = self.data_1, self.data_2, self.data_3

        fig, ax = plt.subplots(figsize=(20, 25))
        colors = ['#2300A8', '#00A658', '#F86441']

        ax.scatter(data_1["X_c1"], data_1["Y_c1"], alpha=0.70, color=colors[0], label='Class_1')
        ax.scatter(data_2["X_c2"], data_2["Y_c2"], alpha=0.70, color=colors[1], label='Class_2')
        ax.scatter(data_3["X_c3"], data_3["Y_c3"], alpha=0.70, color=colors[2], label='Class_3')

        ax.set_title('Dataset', fontsize=25)
        ax.set_xlabel('Feature_1', fontsize=16)
        ax.set_ylabel('Feature_2', fontsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.legend(loc='lower right')
        if flag == None:
            plt.show()
        if flag == 1:
            return ax, fig

    def visualize_corr(self, class_id):
        plt.figure(figsize=(9, 5))
        sns.heatmap(self.data[class_id].corr(), cmap=sns.cubehelix_palette(8))
        plt.title("Correlation Matrix of class " + str(class_id), fontsize=15)
        plt.show()

    def visualize_pairplot(self, class_id):
        g = sns.pairplot(self.data[class_id])
        g.fig.suptitle("Pairwise Plot " + str(class_id), fontsize=15)
        plt.show()

    def visualize_displot(self, class_id):
        for i in range(self.data[class_id].shape[1]):
            sns.distplot(self.data[class_id].iloc[:, i])
            if i == 0:
                plt.title("Displot of " + str(i+1) + "st feature of class " + str(class_id), fontsize=15)
            else:
                plt.title("Displot of " + str(i+1) + "nd feature of class " + str(class_id), fontsize=15)
            plt.show()
            plt.clf()

    def prior_calculation(self):
        prior = []
        prior.append(self.data[1].shape[0] / (self.data[1].shape[0] + self.data[2].shape[0] + self.data[3].shape[0]))
        prior.append(self.data[2].shape[0] / (self.data[1].shape[0] + self.data[2].shape[0] + self.data[3].shape[0]))
        prior.append(self.data[3].shape[0] / (self.data[1].shape[0] + self.data[2].shape[0] + self.data[3].shape[0]))
        return prior

    def gaussian_func(self, x, mu, sigma):
        delta_sigma = np.linalg.det(sigma)
        inverse_sigma = np.linalg.inv(sigma)
        term_1 = 1 / (2 * np.pi * np.sqrt(delta_sigma))
        term_2 = np.matmul(np.transpose(x - mu), inverse_sigma)
        term_3 = np.matmul(term_2, (x - mu))
        term_4 = -1 * (1/2) * term_3
        z = term_1 * np.exp(term_4)
        return z

    def bivariate_normal(self, class_id, smoothness=0.5):
        mu = np.array([self.data[class_id].iloc[:, 0].mean(), self.data[class_id].iloc[:, 1].mean()], dtype='float32').reshape((2, 1))
        sigma = np.array(self.data[class_id].corr(), dtype='float32')    

        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca(projection='3d')

        if class_id == 1:
            x = np.arange(self.X_c1_min-5, self.X_c1_max+5, smoothness)
            y = np.arange(self.Y_c1_min-5, self.Y_c1_max+5, smoothness)
        
        if class_id == 2:
            x = np.arange(self.X_c2_min-5, self.X_c2_max+5, smoothness)
            y = np.arange(self.Y_c2_min-5, self.Y_c2_max+5, smoothness)
        
        if class_id == 3:
            x = np.arange(self.X_c3_min-5, self.X_c3_max+5, smoothness)
            y = np.arange(self.Y_c3_min-5, self.Y_c3_max+5, smoothness)

        X, Y = np.meshgrid(x, y)
        zs = np.column_stack([X.flat, Y.flat])
        print(X.shape)
        z = np.array([self.gaussian_func(i.reshape((2, 1)), mu, sigma) for i in zs])
        
        Z = z.reshape(X.shape)
        
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap("GnBu"),
                            linewidth=0, antialiased=False)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel("Feature_1", fontsize=20, labelpad=15)
        ax.set_ylabel("Feature_2", fontsize=20, labelpad=15)
        ax.set_zlabel("Likelihood", fontsize=20, labelpad=15)
        ax.set_title("Probability Distribution Curve for Class " + str(class_id), fontsize=20)
        plt.show()

    def class_contour_lines(self, class_label, smoothness=0.5, filled=None, all=None, plot_data_point=False, ax_pass=None, fig_pass=None, contour_label=True):
        colors = ['#2300A8', '#00A658', '#F86441']
        if all:
            num_classes = 3
            class_label = 1
            filled = False
        if not all:
            num_classes = 1
            class_label = class_label
        
        if ax_pass:
            ax = ax_pass
            fig = fig_pass
        if not ax_pass:
            fig = plt.figure(figsize=(20, 25))
            ax = fig.gca()
        for class_id in range(1, num_classes+1):
            class_id = class_id + class_label - 1
            mu = np.array([self.train[class_id].iloc[:, 0].mean(), self.train[class_id].iloc[:, 1].mean()], dtype='float32').reshape((2, 1))
            sigma = np.array(self.train[class_id].corr(), dtype='float32')    

            x_y_class_id = {1: [np.arange(self.X_c1_min, self.X_c1_max, smoothness), np.arange(self.Y_c1_min, self.Y_c1_max, smoothness)],
                            2: [np.arange(self.X_c2_min, self.X_c2_max, smoothness), np.arange(self.Y_c2_min, self.Y_c2_max, smoothness)],
                            3: [np.arange(self.X_c3_min, self.X_c3_max, smoothness), np.arange(self.Y_c3_min, self.Y_c3_max, smoothness)]
            # }

            # x_y_class_id = {1: [np.arange(self.data_min_x - 2, self.data_max_x + 2, smoothness), np.arange(self.data_min_y - 2, self.data_max_y + 2, smoothness)],
            #                 2: [np.arange(self.data_min_x - 2, self.data_max_x + 2, smoothness), np.arange(self.data_min_y - 2, self.data_max_y + 2, smoothness)],
            #                 3: [np.arange(self.data_min_x - 2, self.data_max_x + 2, smoothness), np.arange(self.data_min_y - 2, self.data_max_y + 2, smoothness)]
            }

            X, Y = np.meshgrid(x_y_class_id[class_id][0], x_y_class_id[class_id][1])
            zs = np.column_stack([X.flat, Y.flat])
            z = np.array([self.gaussian_func(i.reshape((2, 1)), mu, sigma) for i in zs])
            
            Z = z.reshape(X.shape)

            if not filled:
                contour = ax.contour(X, Y, Z, levels=10)
                if contour_label:
                    ax.clabel(contour, inline=1)
                
            if filled:
                contourf = plt.contourf(X, Y, Z, levels=10)
                fig.colorbar(contourf, shrink=0.5, aspect=10)
        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel("Feature_1", fontsize=15)
            ax.set_ylabel("Feature_2", fontsize=15)
            ax.set_title("Contour Plots of Class " + str(class_id), fontsize=25)
            if plot_data_point:
                tr_c, _= self.train_test_split(class_id)
                ax.scatter(tr_c["X_c"+str(class_id)], tr_c["Y_c"+str(class_id)], color=colors[class_id-1], alpha=0.3)
                ax.set_title("Contour Plots of Class " + str(class_id) + " (with data)", fontsize=25)

        if all:
            ax.set_title("Contour Plots of all classes", fontsize=25)
            if plot_data_point:
                ax.set_title("Contour Plots of all classes (with respective data)", fontsize=25)
        if not ax_pass:
            plt.show()
        if ax_pass:
            return ax

    def contour_plot_pair_wise(self, i, j, smoothness=0.5, filled=None, plot_data_point=None):
        fig = plt.figure(figsize=(20, 25))
        ax = fig.gca()
        ax = self.class_contour_lines(i, smoothness=smoothness, filled=filled, plot_data_point=plot_data_point, ax_pass=ax, fig_pass=fig)
        ax = self.class_contour_lines(j, smoothness=smoothness, filled=filled, plot_data_point=plot_data_point, ax_pass=ax, fig_pass=fig)
        ax.set_title("Contour Plots for classes " + str(i) + " and " + str(j), fontsize=25)
        plt.show()

    def train_test_split(self, class_id, train_size=0.75):
        train = self.data[class_id].sample(frac=train_size, random_state=42)
        test = self.data[class_id].drop(train.index)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        return train, test

    def prepare_test_data(self, test_c1, test_c2, test_c3):
        test_c1.reset_index(drop=True, inplace=True)
        test_c1.columns = ['X', 'Y']
        test_c1['True_Label'] = pd.Series(np.ones_like(test_c1['X'], dtype='int64') * int(1), index=test_c1.index)

        test_c2.reset_index(drop=True, inplace=True)
        test_c2.columns = ['X', 'Y']
        test_c2['True_Label'] = pd.Series(np.ones_like(test_c2['X'], dtype='int64') * int(2), index=test_c2.index)

        test_c3.reset_index(drop=True, inplace=True)
        test_c3.columns = ['X', 'Y']
        test_c3['True_Label'] = pd.Series(np.ones_like(test_c3['X'], dtype='int64') * int(3), index=test_c3.index)
        
        frames = [test_c1, test_c2, test_c3]
        test_data = pd.concat(frames, axis=0)
        test_data.reset_index(drop=True, inplace=True)
        return test_data

    def shuffle_data(self, data):
        df = data.sample(frac=1).reset_index(drop=True)
        return df