"""
The :mod:`front_end.cl.run_al_cl` implements the methods needed to
run the command-line interface.
"""
import os, sys
path = os.path.join(os.path.dirname("__file__"), '../..')
sys.path.insert(0, path)

import argparse

from collections import defaultdict
from time import time

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split

from al.learning_curve import LearningCurve
from utils.utils import *

def load_data(dataset1, dataset2=None):
    """Loads the dataset(s) given in the the svmlight / libsvm format
    and assumes a train/test split

    **Parameters**

    * dataset1 (*str*) - Path to the file of the first dataset.
    * dataset2 (*str or None*) - If not None, path to the file of second dataset

    **Returns**

    * (X_pool, X_test, y_pool, y_test) - Pool and test files

    """
    if dataset2:
        X_pool, y_pool = load_svmlight_file(dataset1)
        _, num_feat = X_pool.shape

        # Splitting 2/3 of data as training data and 1/3 as testing
        # Data selected randomly
        X_test, y_test = load_svmlight_file(dataset2, n_features=num_feat)

    else:
        X, y = load_svmlight_file(dataset1)
        X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=(1./3.), random_state=42)

    return (X_pool, X_test, y_pool, y_test)

class cmd_parse(object):
    """Class - command line parser"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def retrieve_args(self):
        """Adds arguments to the parser for each respective setting of the command line interface"""
        # Classifier
        self.parser.add_argument("-c","--classifier", choices=['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'BernoulliNB',
                        'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNB', 'MultinomialNB'],
                        default='MultinomialNB', help="Represents the classifier that will be used (default: MultinomialNB) .")

        # Classifier's arguments
        self.parser.add_argument("-a","--arguments", default='',
                        help="Represents the arguments that will be passed to the classifier (default: '').")

        # Data: Testing and training already split
        self.parser.add_argument("-d", '--data', nargs=2, metavar=('pool', 'test'),
                        default=["data/imdb-binary-pool-mindf5-ng11", "data/imdb-binary-test-mindf5-ng11"],
                        help='Files that contain the data, pool and test, and number of features (default: data/imdb-binary-pool-mindf5-ng11 data/imdb-binary-test-mindf5-ng11 27272).')

        # Data: Single File
        self.parser.add_argument("-sd", '--sdata', type=str, default='',
                        help='Single file that contains the data, it will be splitted (default: None).')

        # File: Name of file that will be written the results
        self.parser.add_argument("-f", '--file', type=str, default='',
                        help='This feature represents the name that will be written with the result. If it is left blank, the file will not be written (default: '' ).')

        # Number of Trials
        self.parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

        # Strategies
        self.parser.add_argument("-st", "--strategies", choices=['erreduct', 'loggain', 'qbc', 'rand','unc'], nargs='*',default=['rand'],
                        help="Represent a list of strategies for choosing next samples (default: rand).")

        # Boot Strap
        self.parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')

        # Budget
        self.parser.add_argument("-b", '--budget', default=500, type=int,
                        help='Sets the budget (default: 500).')

        # Step size
        self.parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')

        # Sub pool size
        self.parser.add_argument("-sp", '--subpool', default=250, type=int,
                        help='Sets the sub pool size (default: 250).')


    def assign_args(self):
        """Assigns values to each of the specified command line arguments for use by :mod:`al.learning_curve`"""

        t0 = time()

        self.args = self.parser.parse_args()
        self.classifier = eval((self.args.classifier))
        model_arguments = self.args.arguments.split(',')
        self.alpha = {}

        for argument in model_arguments:
            if argument.find('=') >= 0:
                index, value = argument.split('=')
                self.alpha[index] = eval(value)

        self.num_trials = self.args.num_trials
        self.strategies = self.args.strategies
        self.boot_strap_size = self.args.bootstrap
        self.budget = self.args.budget
        self.step_size = self.args.stepsize
        self.sub_pool = self.args.subpool

        self.filename = self.args.file
        self.duration = defaultdict(lambda: 0.0)
        self.accuracies = defaultdict(lambda: [])
        self.aucs = defaultdict(lambda: [])

        if self.args.sdata:
            self.X_pool, self.X_test, self.y_pool, self.y_test = load_data(self.args.sdata)
        else:
            self.X_pool, self.X_test, self.y_pool, self.y_test = load_data(self.args.data[0], self.args.data[1])

        duration = time() - t0

        print
        print "Loading took %0.2fs." % duration
        print

        self.num_test = self.X_test.shape[0]

    def run_al(self):
        """Calls :mod:`al.learning_curve.LearningCurve` and draws plots using :mod:`utils.utils`"""
        learning_api = LearningCurve()
        # if self.filename:
        #     f = open(self.filename, 'a')
        # else:
        #     f = open('avg_results.txt', 'a')
        for strategy in self.strategies:
            values, avg_accu, avg_auc = learning_api.run_trials(self.X_pool, self.y_pool, self.X_test, self.y_test, strategy, self.classifier, self.alpha, self.boot_strap_size, self.step_size, self.budget, self.num_trials)
            accu_x, accu_y, auc_x, auc_y = assign_plot_params(avg_accu, avg_auc)

            # Write data to file
            data_to_file(self.filename, strategy, accu_y, auc_y, values)

            # Draw Plots
            draw_plots(strategy, accu_x, accu_y, auc_x, auc_y)

        show_plt()

    def main(self):
        """Calls :mod:`retrieve_args`, :mod:`assign_args`, :mod:`run_al`"""
        self.retrieve_args()
        self.assign_args()
        self.run_al()

if __name__ == '__main__':
    cli_app = cmd_parse()
    cli_app.main()
