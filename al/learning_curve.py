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

    acc_by_trial = []
    auc_by_trial = []

    for t in num_trials:
        print "trial", t
        acc_l, auc_l = run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget)
        acc_by_trial.append(acc_l)
        auc_by_trial.append(auc_l)

def run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget):
    """Helper method for running multiple trials."""
    accuracies = defaultdict(lambda: [])
    aucs = defaultdict(lambda: [])

    # Gaussian Naive Bayes requires denses matrizes
    if (classifier) == type(GaussianNB()):
        X_pool_csr = X_pool.toarray()
    else:
        X_pool_csr = X_pool.tocsr()

    pool = set(range(len(y_pool)))

    trainIndices = []

    bootstrapped = False

    # Choosing strategy
    if al_strategy == 'erreduct':
        active_s = ErrorReductionStrategy(classifier=classifier_name, seed=t, sub_pool=sub_pool, classifier_args=classifier_arguments)
    elif al_strategy == 'loggain':
        active_s = LogGainStrategy(classifier=classifier_name, seed=t, sub_pool=sub_pool, classifier_args=classifier_arguments)
    elif al_strategy == 'qbc':
        active_s = QBCStrategy(classifier=classifier_name, classifier_args=classifier_arguments)
    elif al_strategy == 'rand':
        active_s = RandomStrategy(seed=t)
    elif al_strategy == 'unc':
        active_s = UncStrategy(seed=t, sub_pool=sub_pool)

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

        accuracies[len(trainIndices)].append(accu)
        aucs[len(trainIndices)].append(auc)

    return accuracies, aucs
