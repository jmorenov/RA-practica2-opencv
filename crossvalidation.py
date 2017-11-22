import numpy as np
from sklearn.model_selection import LeaveOneOut
import knn

def count_success(Y_Pred, Y):
    return np.sum(Y_Pred == Y)

def loo(knn_model, data, responses, max_k = 13):
    loo = LeaveOneOut()
    iterations = loo.get_n_splits(data)

    k = 1

    while k <= max_k:
        success = 0
        i = 0
        for train_index, test_index in loo.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = responses[train_index], responses[test_index]
            ret, results, neighbours, dist = knn.predict(knn_model, X_train, k)
            success += count_success(results.transpose(), y_train)
            i += 1

        print 'K = ' + str(k) + ' | Success rate: ' + str((float(success) / float(((len(data)) - 1) * iterations)) * 100) + '%'

        k += 2
