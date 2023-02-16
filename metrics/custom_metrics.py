import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score

min_max_score = {
    1: (2, 12),
    2: (0, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}

class KappaScore():
    def __init__(self, weights='quadratic'):
        self.weights = weights
        self.pred_list = []
        self.true_list = []
        self.id_list = []

    def update_state(self, y_pred, y_true, ids):
        self.pred_list.extend(y_pred.numpy().flatten().tolist())
        self.true_list.extend(y_true.numpy().flatten().tolist())
        self.id_list.extend(ids.numpy().flatten().tolist())

    def reset_states(self):
        self.pred_list.clear()
        self.true_list.clear()
        self.id_list.clear()

    def result(self):
        kappa_list = []
        for prompt_id in range(1, 9):
            minscore, maxscore = min_max_score[prompt_id]
            id_list = np.array(self.id_list)
            y_pred_arr = np.array(self.pred_list)
            y_true_arr = np.array(self.true_list)
            if np.sum(id_list==prompt_id) == 0:
                continue
            mask = id_list == prompt_id
            y_true = y_true_arr[mask]
            y_true = np.round((maxscore - minscore) * y_true + minscore)
            y_pred = y_pred_arr[mask]
            y_pred = np.round((maxscore - minscore) * y_pred + minscore)
            kappa_score = cohen_kappa_score(y_true, y_pred,
                                            weights=self.weights,
                                            labels=[i for i in range(minscore, maxscore+1)])
            kappa_list.append(kappa_score)
        return np.mean(kappa_list)


class CorrelationCoefficient():
    def __init__(self):
        self.pred_list = []
        self.true_list = []

    def update_state(self, y_pred, y_true):
        self.pred_list.extend(y_pred.numpy().flatten().tolist())
        self.true_list.extend(y_true.numpy().flatten().tolist())

    def reset_states(self):
        self.pred_list.clear()
        self.true_list.clear()

    def result(self):
        return np.corrcoef(np.array([self.true_list, self.pred_list]))[0, 1]