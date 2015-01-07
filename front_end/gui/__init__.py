"""Dependences for gui"""
from Tkinter import *
from PIL import Image, ImageTk
import tkMessageBox, re, os, signal
from time import sleep
import tkFont, tkFileDialog

import matplotlib.pyplot as pl

"""load_data"""
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
