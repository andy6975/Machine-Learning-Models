import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluation(object):

    def __init__(self, bayes_case, data_prep, test_size=0.30):
        self.data_preprocess = data_prep
        self.bayes_classifier = bayes_case

        _, test_c1 = data_prep.train_test_split(1, train_size=1-test_size)
        _, test_c2 = data_prep.train_test_split(2, train_size=1-test_size)
        _, test_c3 = data_prep.train_test_split(3, train_size=1-test_size)

        self.test_data = self.data_preprocess.prepare_test_data(test_c1, test_c2, test_c3)

    def confusion_matrix(self):
        df = self.data_preprocess.shuffle_data(self.test_data).reset_index(drop=True)

        pred_class = np.array([], dtype='int')
        for i in range(df.shape[0]):
            point = df.iloc[i, 0:2].values.reshape((2, 1))
            g_classes = []
            for class_id in range(1, 4):
                g_classes.append(self.bayes_classifier.discriminative_func(point, class_id))
            pred_class = np.append(pred_class, (g_classes.index(max(g_classes)) + 1))
        true_class = np.array(df.iloc[:, -1])
        num_classes = np.unique(true_class).shape[0]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype='int')
        for i in range(true_class.shape[0]):
            a, b = true_class[i] - 1, pred_class[i] - 1
            confusion_matrix[a][b] += 1

        return confusion_matrix

    def accuracy(self):
        cm = self.confusion_matrix()
        correct_predictions = np.trace(cm)
        total_samples = self.test_data.shape[0]
        acc = correct_predictions / total_samples

        return acc

    def precision(self, class_id):
        cm = self.confusion_matrix()
        a = class_id - 1
        numerator = cm[a][a]
        denominator = np.sum(cm, axis=1)[a]
        precision = numerator / denominator

        return precision

    def recall(self, class_id):
        cm = self.confusion_matrix()
        a = class_id - 1
        numerator = cm[a][a]
        denominator = np.sum(cm, axis=0)[a]
        recall = numerator / denominator

        return recall

    def f_score(self, class_id):
        prec, rec = self.precision(class_id), self.recall(class_id)
        f_score = (2 * prec * rec) / (prec + rec)

        return f_score

    def mean_precision(self):
        a, b, c = self.precision(1), self.precision(2), self.precision(3)
        mean_prec = (a + b + c) / 3

        return mean_prec

    def mean_recall(self):
        a, b, c = self.recall(1), self.recall(2), self.recall(3)
        mean_rec = (a + b + c) / 3

        return mean_rec

    def mean_f_score(self):
        a, b, c = self.f_score(1) + self.f_score(2) + self.f_score(3)
        mean_f_score = (a + b + c) / 3

        return mean_f_score

    def plot_confusion_matrix(self):
        cm = self.confusion_matrix()
        labels = ["C_1", "C_2", "C_3"]
        df_cm = pd.DataFrame(cm, index = [i for i in labels],
                  columns = [i for i in labels])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True, fmt='g', cmap="YlGnBu")
        plt.show()