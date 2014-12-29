"""Dependencies for both cl and gui"""

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
