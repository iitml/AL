"""
The :mod:`utils.data_utils` implements the necessary tools for loading
the datasets.
"""

from sklearn.datasets import load_svmlight_file, load_svmlight_files

def load_data(dataset1, dataset2=None):
    """Load a dataset in the svmlight / libsvm format.
    
    Parameters
    ----------
    dataset1: str
        Path to the file of the first dataset.
        
    dataset2: str or None
        If not None, path to the file of second dataset
        
    Returns
    -------
    X_1: scipy.sparse matrix

    y_1: ndarray
    
    If dataset2 is not None, return also
    
    X_2: scipy.sparse matrix

    y_2: ndarray
    
    """
    if dataset2:
        return load_svmlight_files([dataset1, dataset2])
    else:
        return load_svmlight_file(dataset1)
