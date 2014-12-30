"""
The :mod:`al.learning_curve` implements the methods needed to
run a given active learning strategy.
"""
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from time import time

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.datasets import load_svmlight_file

from sklearn.cross_validation import train_test_split

from instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, RotateStrategy, BootstrapFromEach, QBCStrategy, ErrorReductionStrategy


def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, num_trials):
    """Runs a given active learning strategy multiple trials and returns
    the average performance.

    Parameters
    ----------
    TBC.

    Returns
    -------
    (avg_accu, avg_auc)
      - respective average performance

    """

    self.accuracies = defaultdict(lambda: [])
    self.aucs = defaultdict(lambda: [])

    for t in range(num_trials):
        print "trial", t
        self.run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t)

    avg_accu = {}
    avg_auc = {}

    values = sorted(self.accuracies.keys())
    for val in values:
        avg_accu[val] = np.mean(self.accuracies[val])
        avg_auc[val] = np.mean(self.aucs[val])

    return avg_accu, avg_auc

def run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t):
    """Helper method for running multiple trials."""

    # Gaussian Naive Bayes requires denses matrizes
    if (classifier_name) == type(GaussianNB()):
        X_pool_csr = X_pool.toarray()
    else:
        X_pool_csr = X_pool.tocsr()

    pool = set(range(len(y_pool)))

    trainIndices = []

    bootstrapped = False

    # Choosing strategy
    if al_strategy == 'erreduct':
        active_s = ErrorReductionStrategy(classifier=classifier_name, seed=t, classifier_args=classifier_arguments)
    elif al_strategy == 'loggain':
        active_s = LogGainStrategy(classifier=classifier_name, seed=t, classifier_args=classifier_arguments)
    elif al_strategy == 'qbc':
        active_s = QBCStrategy(classifier=classifier_name, classifier_args=classifier_arguments)
    elif al_strategy == 'rand':
        active_s = RandomStrategy(seed=t)
    elif al_strategy == 'unc':
        active_s = UncStrategy(seed=t)

    model = None

    #Loop for prediction
    while len(trainIndices) < budget and len(pool) > step_size:

        if not bootstrapped:
            boot_s = BootstrapFromEach(t)
            newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
            bootstrapped = True
        else:
            newIndices = active_s.chooseNext(pool, X_pool_csr, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

        pool.difference_update(newIndices)

        trainIndices.extend(newIndices)

        model = classifier_name(**classifier_arguments)

        model.fit(X_pool_csr[trainIndices], y_pool[trainIndices])

        # Prediction

        # Gaussian Naive Bayes requires dense matrices
        if (classifier_name) == type(GaussianNB()):
            y_probas = model.predict_proba(X_test.toarray())
        else:
            y_probas = model.predict_proba(X_test)

        # Metrics
        auc = metrics.roc_auc_score(y_test, y_probas[:,1])

        pred_y = model.classes_[np.argmax(y_probas, axis=1)]

        accu = metrics.accuracy_score(y_test, pred_y)

        self.accuracies[len(trainIndices)].append(accu)
        self.aucs[len(trainIndices)].append(auc)
