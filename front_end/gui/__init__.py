"""Dependences for gui"""
from Tkinter import *
from PIL import Image, ImageTk
import tkMessageBox, re, os, signal
from time import sleep
import tkFont, tkFileDialog

"""load_data"""
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
