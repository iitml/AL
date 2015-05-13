"""
The :mod:`al.learning_curve` implements the methods needed to
run a given active learning strategy.
"""

import numpy as np


from collections import defaultdict

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy, ErrorReductionStrategy

class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, num_trials):
        """Runs a given active learning strategy multiple trials and returns
        the average performance.

        **Parameters**

        * X_pool - returned from load_svmlight_file
        * y_pool - returned from load_svmlight_file
        * X_test - returned from load_svmlight_file
        * y_test - returned from load_svmlight_file
        * al_strategy - Represent a list of strategies for choosing next samples (default - rand).
        * classifier_name - Represents the classifier that will be used (default - MultinomialNB) .
        * classifier_arguments - Represents the arguments that will be passed to the classifier (default - '').
        * bootstrap_size - Sets the Boot strap (default - 10).
        * step_size - Sets the step size (default - 10).
        * budget - Sets the budget (default - 500).
        * num_trials - Number of trials (default - 10).

        **Returns**

        * (values, avg_accu, avg_auc) - training_size, respective average performance

        """

        self.all_performances = {}
        
        average_performances = {}
        
        labels = np.unique(y_pool)
        
        measures = ["accuracy", "auc"]
        
        for measure in ["precision_", "recall_", "f1_"]:
            for label in labels:
                measures.append(measure+str(label))
        
        
        for measure in measures:
            self.all_performances[measure] = defaultdict(list)
            average_performances[measure] = {}

        for t in range(num_trials):
            print "trial", t
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t)
        
        return self.all_performances
        
        # For now, assume each performance measure is evaluated at the same budget levels
        bs = sorted(self.all_performances["accuracy"].keys())
        
        for b in bs:
            for measure in measures:
                average_performances[measure][b] = np.mean(self.all_performances[measure][b])
        
        # for compatibility with the command line and gui, for now return only budget, accuracy, and auc
        
        return bs, average_performances["accuracy"], average_performances["auc"]

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t):
        """Helper method for running multiple trials."""
        
        # This is a hack for running the empirical study experiments
        
        if len(y_pool) > 10000:
            rs = np.random.RandomState(t)
            indices = rs.permutation(len(y_pool))
            pool = set(indices[:10000])
        else:
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
        
        labels = np.unique(y_pool)

        #Loop for prediction
        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            pool.difference_update(newIndices)

            trainIndices.extend(newIndices)

            model = classifier_name(**classifier_arguments)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # Measures
            
            self.all_performances["accuracy"][len(trainIndices)].append(metrics.accuracy_score(y_test, y_pred))
            self.all_performances["auc"][len(trainIndices)].append(metrics.roc_auc_score(y_test, y_probas[:,1]))
            
            for label in labels:            
                self.all_performances["precision_"+str(label)][len(trainIndices)].append(metrics.precision_score(y_test, y_pred, pos_label=label))
                self.all_performances["recall_"+str(label)][len(trainIndices)].append(metrics.recall_score(y_test, y_pred, pos_label=label))
                self.all_performances["f1_"+str(label)][len(trainIndices)].append(metrics.f1_score(y_test, y_pred, pos_label=label))
                
            
