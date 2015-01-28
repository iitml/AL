Active Learning Library
==
# pyAL
--------------
pyAL is an python library that implements common active learning strategies. The project is currently developed by the members of the Machine Learning Lab at IIT: (http://ml.cs.iit.edu). 

Currently supported strategies:

1. Random sampling
2. Uncertainty sampling
3. Query-by-committee
4. Expected error reduction
5. Log-gain

# Related Links 
--------------
* Machine Learning Lab @ IIT (http://cs.iit.edu/~ml)
* Documentation: (http://iitml.github.io/AL/)
* Official source code repo: (https://github.com/scikit-learn/scikit-learn)
* HTML documentation (stable release): (http://scikit-learn.org)


# Dependencies
-----------
pyAL is tested to work under Python 2.7. The required dependencies to build the software are scikit-learn >= 0.15, NumPy >= 1.6.2, SciPy >= 0.9 and a working C/C++ compiler.

For runing the GUI ...

# Installation
----------------
<implement and complete this section>
This package uses distutils, which is the default way of installing python modules. To install in your home directory, use:

python setup.py install --user
To install for all users on Unix/Linux:
```
$ python setup.py build
$ sudo python setup.py install
```

# Content of this Repository
This repository is organized as follows:

* ```al```: contains the code
* ```documentation```: contains the documentation of the classes
* ```front_end```: containts the GUI
* ```utils```: contains the utilities

# Development

## Code - GIT

You can check the latest sources with the command:

```
git clone git@github.com:iitml/AL.git
```

# Testing

After installation, you can launch the test suite from outside the source directory (you will need to have the nose package installed):

```
how to run the test
```
