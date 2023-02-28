import numpy as np

from caffeine.utils.metric import *


def test_bcl_metrics():
    label = np.array([[0], [0], [0], [1], [1]])
    label1 = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])

    preds_very_good = np.array([[0.01], [0.01], [0.01], [0.99], [0.99]])
    preds_good = np.array([[0.1], [0.3], [0.1], [0.9], [0.7]])
    preds_middle = np.array([[0.7], [0.3], [0.3], [0.9], [0.6]])
    preds_bad = np.array([[0.7], [0.5], [0.8], [0.1], [0.3]])
    preds_bad1= np.array([[0.3,0.7], [0.5,0.5], [0.2,0.8], [0.9,0.1], [0.7,0.3]])

    bcl_very_good = bcl_metrics(preds_very_good, label)
    bcl_good = bcl_metrics(preds_good, label)
    bcl_middle= bcl_metrics(preds_middle, label)
    bcl_bad = bcl_metrics(preds_bad.T, label)

    for m in ['ks', 'auc']:
        assert bcl_very_good[m] >= bcl_good[m] and bcl_good[m] > bcl_middle[m] and bcl_middle[m] > bcl_bad[m]
        assert bcl_metrics(preds_very_good, label)[m] == bcl_metrics(preds_very_good, label1)[m]
        assert bcl_metrics(preds_bad, label)[m] == bcl_metrics(preds_bad1, label)[m]
        assert bcl_metrics(preds_bad, label1)[m] == bcl_metrics(preds_bad1, label)[m]
        assert bcl_metrics(preds_bad, label)[m] == bcl_metrics(preds_bad1, label1)[m]

    rmse_value = rmse(label, preds_very_good)
    mae_value = mae(label, preds_very_good)

if __name__ == '__main__':
    test_bcl_metrics()
