"""
The command-line module to run the active learning strategies.
"""

def load_data(dataset1, dataset2=None):
    """Loads the dataset(s) given in the the svmlight / libsvm format
    and assumes a train/test split

    Parameters                                                                                                                                       \
    ----------
    dataset1: str
        Path to the file of the first dataset.
    dataset2: str or None
        If not None, path to the file of second dataset

    Returns
    ----------
    Pool and test files:
    X_pool, X_test, y_pool, y_test
    """
    if dataset2:
        X_pool, y_pool = load_svmlight_file(dataset1)
        num_pool, num_feat = X_pool.shape

        # Splitting 2/3 of data as training data and 1/3 as testing
        # Data selected randomly
        X_test, y_test = load_svmlight_file(dataset2, n_features=num_feat)

    else:
        X, y = load_svmlight_file(dataset1)
        X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=(1./3.), random_state=42)

    return (X_pool, X_test, y_pool, y_test)

if __name__ == '__main__':
    pass
