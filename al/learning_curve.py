"""
The :mod:`al.learning_curve` implements the methods needed to
run a given active learning strategy.
"""

def run_trials(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, num_trials):
    """Runs a given active learning strategy multiple trials and returns
    the average performance.
    
    Parameters
    ----------
    TBC.
    
    Returns
    -------
    TBC.
    """
    pass

def _run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget):
    """Helper method for running multiple trials."""
    pass
