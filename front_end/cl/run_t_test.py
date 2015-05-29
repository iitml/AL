"""
The :mod:`front_end.cl.run_t_test` implements the methods needed to
read results and run t-tests on them.
"""

import sys
import csv
import numpy as np
from scipy import stats

if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    
    X1 = []
    with open(filename1) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)#skip names
        next(csvreader, None)#skip bootstrap
        for row in csvreader:
            X1.append(row[1:])
    
    X1=np.array(X1, dtype=float)
    
    X2 = []
    with open(filename2) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)#skip names
        next(csvreader, None)#skip bootstrap
        for row in csvreader:
            X2.append(row[1:])
    
    X2=np.array(X2, dtype=float)
    
    # This is an approximation of the area under the curve
    means1 = np.mean(X1, axis=0)
    means2 = np.mean(X2, axis=0)
    
    print "Pairing the area under the learning curves"
    
    print stats.mstats.ttest_ind(means1, means2)
    
    print "Pairing the mean learning curves"
    
    means1 = np.mean(X1, axis=1)
    means2 = np.mean(X2, axis=1)
    
    print stats.mstats.ttest_ind(means1, means2)
    
    
            
            
