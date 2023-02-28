#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: metric.py
#
#  Create date: 2020/12/21
#
import numpy as np
import pandas as pd
from flex.constants import OTP_PN_FL
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from typing import Optional, Tuple, List, Dict, Union

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.utils.common_tools import bcl_repr, mcl_repr
from caffeine.utils.dataframe import parse_ibonddf
from caffeine.utils.exceptions import ArgumentError


def rmse(
    label: np.ndarray,
    prediction: np.ndarray
) -> float:
    return metrics.mean_squared_error(
        y_true = label,
        y_pred = prediction
    )


def mae(
    label: np.ndarray,
    prediction: np.ndarray
) -> float:
    return metrics.mean_absolute_error(
        y_true = label,
        y_pred = prediction
    )

def roc_curve(
        label: np.ndarray,
        prediction: np.ndarray,
        thresholds: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Calculate roc_curve at thresholds.
    """
    prediction = prediction.astype(np.float32)
    num_sample = len(prediction)
    num_thresh = len(thresholds)

    thresh_matrix = np.repeat(
        thresholds.flatten()[:, np.newaxis].astype(np.float32),
        num_sample,
        axis=1
    )
    result_matrix = ((prediction - thresh_matrix) >= 0).astype(np.int8)
    del thresh_matrix

    label_matrix = np.repeat(
        label.flatten()[np.newaxis, :].astype(np.int8),
        num_thresh,
        axis=0
    )

    # label 1, result 1
    tp_matrix = label_matrix * result_matrix
    # label 0, result 0
    tn_matrix = (1 - label_matrix) * (1 - result_matrix)
    # label 0, result 1
    fp_matrix = (1 - label_matrix) * result_matrix
    # label 1, result 0
    fn_matrix = label_matrix * (1 - result_matrix)

    del label_matrix
    del result_matrix

    TP = np.sum(tp_matrix, axis=1).flatten()
    TN = np.sum(tn_matrix, axis=1).flatten()
    FP = np.sum(fp_matrix, axis=1).flatten()
    FN = np.sum(fn_matrix, axis=1).flatten()

    tpr = TP / (TP + FN + 1.e-15)
    fpr = FP / (FP + TN + 1.e-15)

    return fpr, tpr, thresholds


def bcl_metrics(prediction: np.ndarray, label: np.ndarray, 
                f_betas: List[float] = [1., 0.5, 2.],
                thresh: Optional[float] = None,
                thresh_step: float = 0.01,
                thresh_criteria: str = 'fbeta'
               ) -> Dict:
    '''
    Caculate accuracy metrics for binary classification.

    Args:
        prediction: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
        label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.
        f_betas: list of floats, a list of beta for f beta scores.
        thresh: optional float, the threshold to make binary labels, if None, use
            the thresh_criteria to find best threshold.
        thresh_step: optional float, if thresh is None, the step to search the
            best threshold.
        thresh_criteria: string, criteria to find best threshold if thresh is
            None, in [
                'fbeta'
            ].

    Returns:
        dict: keys are metric names and threshold in [
                'thresh', 
                'auc', 
                'ks', 
                'confusion_matrix', 
                'fbeta', 
                'precision',
                'recall',
                'accuracy'
            ] and values are metric values.

    -----

    **Examples:**

    >>> out = bcl_metrics(label=[0, 0, 1, 1], prediction=[0.1, 0.9, 0.3, 0.7])
    '''
    if len(f_betas) <= 0 and thresh is None:
        raise
    prediction = bcl_repr(prediction)
    label = bcl_repr(label)

    # auc & ks
    thresh_proposals = np.arange(0, 1+thresh_step/2., thresh_step)
    fpr, tpr, thresholds = roc_curve(label, prediction, thresh_proposals)
    auc = metrics.auc(fpr, tpr)
    ks = np.max(tpr - fpr)

    # === search for best threshold ===
    if thresh is None:
        thresh_proposals = np.arange(0, 1+thresh_step/2., thresh_step)
        if thresh_criteria == 'fbeta':
            if len(f_betas) <= 0:
                raise ArgumentError(f'Argument f_betas is empty when thresh_criteria is fbeta.')
            proposal_scores = [metrics.fbeta_score(label, prediction > t, beta = f_betas[0])
                            for t in thresh_proposals]
        else:
            raise ArgumentError(f'Argument thresh_criteria {thresh_criteria} is not recognized.')
        best_thresh_idx = np.argmax(proposal_scores)
        thresh = thresh_proposals[best_thresh_idx]
    # ================================

    # threshold based metrics
    if 1. not in f_betas:
        f_betas = f_betas + [1.] # force f1
    fbeta_scores = {f'f{beta}': float(metrics.fbeta_score(label, prediction > thresh, beta = beta))
                    for beta in f_betas}
    confusion_matrix = metrics.confusion_matrix(label, prediction>thresh)
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (1.e-9 + tp + tn +fp + fn)
    precision = tp / (1.e-9 + tp + fp)
    recall = tp / (1.e-9 + tp + fn)

    # curves and tables
    n_thresh = thresholds.shape[0]
    curve = {
        'threshold': thresholds.tolist(),
        'tpr': tpr.tolist(),
        'fpr': fpr.tolist(),
        'auc_m': [float(auc)]*n_thresh,
        'ks': (tpr - fpr).tolist()
    }
    detail_table = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'f1score': float(fbeta_scores['f1.0'])
    }

    # return
    return {
        'thresh': float(thresh),
        'auc': float(auc),
        'ks': float(ks),
        'confusion_matrix': confusion_matrix.tolist(),
        'fbeta': fbeta_scores,
        'precision': float(precision),
        'recall': float(recall),
        'acc': float(accuracy),
        'curve': curve,
        'detail_table': detail_table
    }


def mcl_metrics(prediction: np.ndarray, label: np.ndarray, 
                f_betas: List[float] = [1., 0.5, 2.],
                classes: List = []
               ) -> Dict:
    prediction_s = mcl_repr(prediction, squeeze=True, classes=classes)
    prediction = mcl_repr(prediction, squeeze=False, classes=classes)
    label_s = mcl_repr(label, squeeze=True, classes=classes)
    label = mcl_repr(label, squeeze=False, classes=classes)

    accuracy = metrics.accuracy_score(
        y_true = label_s,
        y_pred = prediction_s
    )

    '''
    macro_f1 = metrics.f1_score(
        y_true = label_s,
        y_pred = prediction_s,
        average = 'macro'
    )

    micro_f1 = metrics.f1_score(
        y_true = label_s,
        y_pred = prediction_s,
        average = 'micro'
    )
    '''
    classification_report = metrics.classification_report(
        y_true = label_s,
        y_pred = prediction_s,
        output_dict = True
    )

    return {
        'acc': float(accuracy),
        'classification_report': classification_report,
        #'macro_f1': float(macro_f1),
        #'micro_f1': float(micro_f1)
    }


class Cluster_Metric():
    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict):
        """
        Init operations for Cluster Metric.

        Args:
            meta_params, dict, a dict of meta parameters:
            {
                'protocol': dict, flex protocol.
                'federal_info_parse': parse federal info into attributes
                'n_clusters': int, number of clusters.
                'sample_number': int, optional, sample data numbers for compute Silhouette Coefficient. Default: 5000
            }
        """
        self._protocol = meta_param['protocol']
        self._federal_info_parser = meta_param['federal_info_parser']
        self._sample_numbers = meta_param['sample_number'] if 'sample_number' in meta_param else 5000
        self._n_clusters = meta_param['n_clusters']
        self._role = self._federal_info_parser.local_role
        self._federation = self._federal_info_parser._federation

    @ClassMethodAutoLog()
    def metrics(self, preds: Optional[IBondDataFrame]=None, data: Optional[IBondDataFrame] = None) -> Dict:
        """
        Clustering evaluation.

        Args:
            preds: ibond dataframe, cluster prediction of input data, includes cluster label and cluster near neighbor label.
            data: ibond dataframe, input data.

        Return:
            metric, dict, a dict of cluster metrics includes supervised metrics and unsupervised metrics. may be includes:
            {
                'SC': float, Silhouette Coefficient, the best value is 1.0 and the worst value is -1.0. Values near 0 indicate overlapping clusters.
                'CHI': float, Calinski-Harabasz Index, the score is defined as ratio between the within-cluster dispersion nnd the between-cluster dispersion.
                'DBI': float, Davies-Bouldin Index, the minimum score is 0.0, with lower values indicating better clustering.
                'ARI': float, Adjust Rand Index, the best value is 1.0, random labeling have an ARI close 0.0.
                'AMI': float, Adjust Mutual Info Score,the best value is 1.0, can be negative.
                'HS': float, Homogeneity Score,the best value is 1.0 and the worst value is 0.0.
                'CS': float, Completeness Score,the best value is 1.0 and the worst value is 0.0.
                'VM': float, V Measure,the best value is 1.0 and the worst value is 0.0.
                'FMI': float, Fowlkes-Mallows Index, the best value is 1.0 and the worst value is 0.0.
             }
        """
        metric = {}
        if self._role in ['guest', 'host']:

            data_shape = data.shape[0] if data is not None else 0
            total_val_data = self._protocol[OTP_PN_FL].param_negotiate(param='equal', data=data_shape, tag='kmeans_pred_metrics')

            if total_val_data == 0:
                return metric

            data_parse = parse_ibonddf(data)
            preds_label = preds[preds.data_desc['cluster_label']].to_numpy()
            near_neighbor_label = preds[preds.data_desc['near_neighbor_label']].to_numpy()
            if len(data_parse['feat_cols']) > 0:
                data_x = data[data_parse['feat_cols']].to_numpy()
                unsupervised_metric = self.unsupervised_metrics(data_x, preds_label, near_neighbor_label)
                for key in list(unsupervised_metric.keys()):
                    metric[key] = unsupervised_metric[key]
            if len(data_parse['y_cols']) > 0:
                data_y = data[data_parse['y_cols']].to_numpy()
                preds_label = preds_label.reshape(data_y.shape[0])
                supervised_metric = self.supervised_metrics(data_y, preds_label)
                for key in list(supervised_metric.keys()):
                    metric[key] = supervised_metric[key]
        if self._role == 'coordinator':
            total_val_data = self._protocol[OTP_PN_FL].param_negotiate(param='equal', data=None, tag='kmeans_pred_metrics')

            if total_val_data == 0:
                return metric

            unsupervised_metric = self.unsupervised_metrics(None,None,None)

        return metric

    @ClassMethodAutoLog()
    def supervised_metrics(self, data_y: np.ndarray, preds_label: np.ndarray) -> Dict:
        """
        Supervised clustering evaluation.

        Args:
            data_y: np.ndarray, ground truth label of eval data.
            preds: np.ndarray, cluster label of eval data.

        Return:
            metric, dict, a dict of supervised cluster metrics, includes:
            {
                'ARI': float, Adjust Rand Index, the best value is 1.0, random labeling have an ARI close 0.0.
                'AMI': float, Adjust Mutual Info Score,the best value is 1.0, can be negative.
                'HS': float, Homogeneity Score,the best value is 1.0 and the worst value is 0.0.
                'CS': float, Completeness Score,the best value is 1.0 and the worst value is 0.0.
                'VM': float, V Measure,the best value is 1.0 and the worst value is 0.0.
                'FMI': float, Fowlkes-Mallows Index, the best value is 1.0 and the worst value is 0.0.
             }
        """
        supervised_metric = {
            'ARI': None,
            'AMI': None,
            'HS': None,
            'CS': None,
            'VM': None,
            'FMI': None
        }
        result_roc = {}
        for j in range(data_y.shape[1]):
            y = data_y[:, j]
            col_ari = metrics.adjusted_rand_score(y, preds_label)
            col_ami = metrics.adjusted_mutual_info_score(y, preds_label)
            col_hs = metrics.homogeneity_score(y, preds_label)
            col_cs = metrics.completeness_score(y, preds_label)
            col_vm = metrics.v_measure_score(y, preds_label)
            col_fmi = metrics.fowlkes_mallows_score(y, preds_label)
            result_roc[j] = {
                'ARI': col_ari,
                'AMI': col_ami,
                'HS': col_hs,
                'CS': col_cs,
                'VM': col_vm,
                'FMI': col_fmi
            }
            if supervised_metric['CS'] is None or supervised_metric['CS'] < col_cs:
                for key in list(result_roc[j].keys()):
                    supervised_metric[key] = result_roc[j][key]

        return supervised_metric

    @ClassMethodAutoLog()
    def unsupervised_metrics(self, data: Optional[np.ndarray]=None, cluster_label: Optional[np.ndarray]=None, near_neighbor_label: Optional[np.ndarray]= None) -> Dict:
        """
        Unsupervised clustering evaluation.

        Args:
            data_y: np.ndarray, optional, ground truth label of eval data. Default: None
            cluster_label: np.ndarray, optional, cluster label of eval data. Default: None
            near_neighbor_label: np.ndarray, optional, the nearest neighbor cluster label of eval data. Default: None

        Return:
            metric, dict, a dict of unsupervised cluster metrics,includes:
            {
                'SC': float, Silhouette Coefficient, the best value is 1.0 and the worst value is -1.0. Values near 0 indicate overlapping clusters.
                'CHI': float, Calinski-Harabasz Index, the score is defined as ratio between the within-cluster dispersion nnd the between-cluster dispersion.
                'DBI': float, Davies-Bouldin Index, the minimum score is 0.0, with lower values indicating better clustering.
            }
        """
        unsupervised_metric = {
            'SC': None,
            'CHI': None,
            'DBI': None
        }
        ch = self.unsupervised_metric_chi(data, cluster_label)
        dbi = self.unsupervised_metric_dbi(data, cluster_label)
        sc = self.unsupervised_metric_sc(data, cluster_label, near_neighbor_label)
        if ch is not None:
            unsupervised_metric['CHI'] = float(ch)
        if dbi is not None:
            unsupervised_metric['DBI'] = float(dbi)
        if sc is not None:
            unsupervised_metric['SC'] = float(sc)

        return unsupervised_metric

    @ClassMethodAutoLog()
    def unsupervised_metric_sc(self, data_x: np.ndarray, cluster_label: np.ndarray, near_neighbor_label: np.ndarray) -> Union[float, None]:
        """
        Compute the mean Silhouette Coefficient of all samples. The Silhouette Coefficient for a sample is (b-a)/max(a,b).
        b is the mean distance between a sample and the nearest cluster, a is the mean intra-cluster distance.

        Args:
            data_x: np.ndarray, feature data.
            cluster_label: np.ndarray, cluster label of input data.
            near_neighbor_label: np.ndarray, the nearest neighbor cluster label for all input data.

        Return:
            sc: Union[float, None], guest/host return float sc, and the coordinator return None.
        """
        if self._role in ['guest', 'host']:
            # sample data
            data_size = len(data_x)
            if data_size > self._sample_numbers:
                if self._role == 'guest':
                    sample_index = np.random.choice(data_size, self._sample_numbers, replace=False)
                    # guest send sample index to all hosts
                    if len(self._federation['host']) > 0:
                        temp_index = self._protocol[OTP_PN_FL].param_broadcast(sample_index)
                if self._role == 'host':
                    sample_index = self._protocol[OTP_PN_FL].param_broadcast()
                sample_data_x = data_x[sample_index]
                sample_cluster_label = cluster_label[sample_index]
                sample_near_neighbor_label = near_neighbor_label[sample_index]
            else:
                sample_data_x = data_x
                sample_cluster_label = cluster_label
                sample_near_neighbor_label = near_neighbor_label

            data_cluster_dict = {}
            for c in range(self._n_clusters):
                data_cluster_dict[str(c)] = sample_data_x[np.where(sample_cluster_label == c)[0]]

            data_size = len(sample_data_x)
            intra_dist = np.zeros((data_size, 1), dtype=np.float32)
            inter_dist = np.zeros((data_size, 1), dtype=np.float32)
            for i in range(data_size):
                label = int(sample_cluster_label[i])
                near_label = int(sample_near_neighbor_label[i])
                data_reshape = sample_data_x[i, :].reshape(1,-1)
                intra_dist[i,0] = np.mean(pairwise_distances(data_cluster_dict[str(label)], data_reshape, squared = True))
                inter_dist[i,0] = np.mean(pairwise_distances(data_cluster_dict[str(near_label)], data_reshape, squared=True))

            if len(self._federation['host']) > 0:
                intra_dist = self._protocol[OTP_PN_FL].param_negotiate(param='sum',data=intra_dist)
                inter_dist = self._protocol[OTP_PN_FL].param_negotiate(param='sum',data=inter_dist)

            dist_diff = inter_dist - intra_dist
            sc = np.mean(dist_diff / np.maximum(intra_dist, inter_dist))
            return sc

        if self._role == 'coordinator':
            if len(self._federation['host']) > 0:
                intra_dist = self._protocol[OTP_PN_FL].param_negotiate(param='sum',data=None)
                inter_dist = self._protocol[OTP_PN_FL].param_negotiate(param='sum',data=None)
            return None

    @ClassMethodAutoLog()
    def unsupervised_metric_chi(self, data_x: np.ndarray, cluster_label: np.ndarray) -> Union[float, None]:
        """
        Compute the Calinski-Harabasz Index. The score is defined as ratio between the with-in cluster dispersion and the
        between-cluster dispersion.

        Args:
            data_x: np.ndarray, feature data.
            cluster_label: np.ndarray, cluster label of input data.

        Return:
            chi: Union[float, None], guest/host return float chi, and the coordinator return None.
        """
        if self._role in ['guest', 'host']:
            data_size = len(data_x)
            mean_x = np.mean(data_x, axis=0)
            extra_disp, intra_disp = 0., 0.
            for i in range(self._n_clusters):
                index_cluster = np.where(cluster_label == i)[0]
                data_cluster = data_x[index_cluster]
                mean_k = np.mean(data_cluster, axis=0)
                extra_disp += len(index_cluster) * np.sum(np.square(mean_k - mean_x))
                intra_disp += np.sum(np.square(data_cluster - mean_k))

            if len(self._federation['host']) > 0:
                extra_disp = self._protocol[OTP_PN_FL].param_negotiate(param ='sum', data = extra_disp)
                intra_disp = self._protocol[OTP_PN_FL].param_negotiate(param ='sum', data = intra_disp)

            chi = extra_disp * (data_size - self._n_clusters) / (intra_disp * (self._n_clusters - 1.)) if intra_disp!=0 else 0

            return chi

        if self._role == 'coordinator':
            if len(self._federation['host']) > 0:
                extra_disp= self._protocol[OTP_PN_FL].param_negotiate(param='sum', data=None)
                intra_disp = self._protocol[OTP_PN_FL].param_negotiate(param='sum', data=None)
            return None

    @ClassMethodAutoLog()
    def unsupervised_metric_dbi(self, data_x: np.ndarray, cluster_label: np.ndarray) -> Union[float, None]:
        """
        Compute  Davies-Bouldin Index. The score is defined as the average similarity measure of each cluster with its
        most similar cluster, where similarity is the ratio of with-in cluster distance to between-cluster distance.

        Args:
            data_x: np.ndarray, feature data.
            cluster_label: np.ndarray, cluster label of input data.

        Return:
            chi: Union[float, None], guest/host return float dbi, and the coordinator return None.
        """
        if self._role in ['guest', 'host']:
            mean_cluster= np.zeros((self._n_clusters, data_x.shape[1]), dtype=np.float32)
            intra_disp = np.zeros((self._n_clusters, 1), dtype=np.float32)
            for i in range(self._n_clusters):
                data_cluster = data_x[np.where(cluster_label == i)[0]]
                mean_cluster[i,:] = np.mean(data_cluster, axis=0)
                intra_disp[i,:] = np.mean(np.sum(np.square(data_cluster - mean_cluster[i,:]),axis =1))

            center_dist = pairwise_distances(mean_cluster, squared=True)

            if len(self._federation['host']) > 0:
                intra_disp = self._protocol[OTP_PN_FL].param_negotiate(param ='sum', data = intra_disp)
                center_dist = self._protocol[OTP_PN_FL].param_negotiate(param ='sum', data = center_dist)

            center_dist[center_dist == 0] = np.inf
            combina_intra_disp = intra_disp[:, None] + intra_disp
            r = np.max(combina_intra_disp / center_dist, axis=1)
            dbi = np.mean(r)
            return dbi

        if self._role == 'coordinator':
            if len(self._federation['host']) > 0:
                intra_disp = self._protocol[OTP_PN_FL].param_negotiate(param='sum', data=None)
                center_dist = self._protocol[OTP_PN_FL].param_negotiate(param='sum', data=None)

            return None

if __name__ == '__main__':
    out = bcl_metrics(label=[0, 0, 1, 1], prediction=[0.1, 0.9, 0.3, 0.7])
    print(out)
