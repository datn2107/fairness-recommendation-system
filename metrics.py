import numpy as np
from typing import List
import sklearn.metrics as metrics

from data.converter import DataConverter


EPSILON = 1e-10
INF = 1e10


class Metrics:
    """
    A class to calculate the metrics for evaluating the recommendation system.

    Note: This class only supports the binary relevance score.
    """

    def __init__(self):
        pass

    @staticmethod
    def dcg_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the discounted cumulative gain (DCG) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCG score.
        """
        return metrics.dcg_score(R, S, k=k)

    @staticmethod
    def ndcg_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the normalized discounted cumulative gain (NDCG) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The NDCG score.
        """
        return metrics.ndcg_score(R, S, k=k)

    @staticmethod
    def map_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean average precision (MAP) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MAP score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k
        return np.mean(np.sum(S * R, axis=1) / k)

    @staticmethod
    def lrap_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the label-ranking average precision (LRAP) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The LRAP score.
        """
        return metrics.label_ranking_average_precision_score(R, S)

    @staticmethod
    def dcf_score(S: np.ndarray, groups: List[np.array], k: int = 30) -> float:
        """
        Calculate the deviation from producer fairness (DCF) score.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        groups (List[np.array]): The list of group indices.
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCF score.
        """
        assert len(groups) == 2, "The number of groups must be 2."

        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        cnt = np.sum(B, axis=0) / B.shape[0]
        cnt_g = [np.sum(cnt[g]) / len(g) for g in groups]

        return cnt_g[0] - cnt_g[1]

    @staticmethod
    def mdg_score_each_item(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean discounted gain (MDG) score for each item.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MDG score for each item.
        """
        matrix_rank = DataConverter.convert_score_matrix_to_rank_matrix(S)
        matrix_rank = matrix_rank * (matrix_rank < k)

        return np.mean(R / np.log2(matrix_rank + 2), axis=0)

    @staticmethod
    def mdg_score(
        p: float,
        S: np.ndarray = None,
        B: np.ndarray = None,
        k: int = 30,
        items_mdg: np.array = None,
    ) -> float:
        """
        Calculate the mean discounted gain (MDG) score.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        items_mdg (np.array): The MDG score for each item.
        p (float): The proportion of items to consider.

        Returns:
        float: The MDG score.
        """
        if items_mdg is None:
            if S is None and B is None:
                raise ValueError("S or B or items_mdg must be provided.")
            if B is None:
                B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k)
            items_mdg = Metrics.mdg_score_each_item(B, S, k)

        n_item = items_mdg.shape[0]
        k = int(n_item * p)
        if k > 0:
            partition_idx = np.argpartition(items_mdg, k)[:k]
        else:
            partition_idx = np.argpartition(items_mdg, k)[k:]

        return np.mean(items_mdg[partition_idx])

    @staticmethod
    def precision_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the precision score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The precision score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        groundtruth = np.sum(R, axis=1)
        groundtruth = np.where(groundtruth < k, groundtruth, k)
        true_positive = np.sum(B * R, axis=1)

        divider = np.where(groundtruth == 0, 1, groundtruth)
        return np.sum(true_positive / divider) / np.count_nonzero(groundtruth)

    @staticmethod
    def recall_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the recall score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The recall score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        groundtruth = np.sum(R, axis=1)
        true_positive = np.sum(B * R, axis=1)

        divider = np.where(groundtruth == 0, 1, groundtruth)
        return np.sum(true_positive / divider) / np.count_nonzero(groundtruth)

    @staticmethod
    def f1_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the F1 score.

        Parameters:1
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The F1 score.
        """
        precision = Metrics.precision_score(R, S, k)
        recall = Metrics.recall_score(R, S, k)

        return 2 * (precision * recall) / (precision + recall)

    def u_mmf(
        R: np.ndarray, p_u_coverage: float = 0.5, p_i_consider: float = 0.1, k: int = 30
    ):
        """
        Calculate the user max-min fairness (U-MMF) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        p_u_coverage (float): The proportion of expected coverage. Default is 0.5.
        p_i_consider (float): The percentage of items to consider. Default is 0.1.
        k (int): The number of recommened items for each user. Default is 30.

        Returns:
        float: The U-MMF score.
        """
        n_users, n_items = R.shape[:2]

        max_coverage_user = n_items * k * p_i_consider
        expected_coverage_user = max_coverage_user * p_u_coverage

        n_items_interval = int(n_items * p_i_consider)
        items_cnt = np.sum(R, axis=0)
        items_idx = np.argsort(items_cnt)

        score = INF
        for i in range(0, n_items, n_items_interval):
            users_coverage_interval = np.sum(
                np.sum(R[:, items_idx[i : i + n_items_interval]], axis=1) > 0
            )
            score = min(score, users_coverage_interval / expected_coverage_user)

        return score

    def u_pf(
        R: np.ndarray, p_u_coverage: float = 0.5, p_i_consider: float = 0.1, k: int = 30
    ):
        """
        Calculate the user proportion fairness (U-PF) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        p_u_coverage (float): The proportion of expected coverage. Default is 0.5.
        p_i_consider (float): The percentage of items to consider. Default is 0.1.
        k (int): The number of recommened items for each user. Default is 30.

        Returns:
        float: The U-PF score.
        """
        n_users, n_items = R.shape[:2]

        max_coverage_user = n_items * k * p_i_consider
        expected_coverage_user = max_coverage_user * p_u_coverage

        n_items_interval = int(n_items * p_i_consider)
        items_cnt = np.sum(R, axis=0)
        items_idx = np.argsort(items_cnt)

        score = []
        for i in range(0, n_items, n_items_interval):
            users_coverage_interval = np.sum(
                np.sum(R[:, items_idx[i : i + n_items_interval]], axis=1) > 0
            )
            score.append(users_coverage_interval / expected_coverage_user)

        return np.mean(score)


def get_metric(R, S, B, top_k):
    entity = {}
    entity["precision"] = Metrics.precision_score(R, S, k=top_k)
    entity["recall"] = Metrics.recall_score(R, S, k=top_k)
    entity["ndcg"] = Metrics.ndcg_score(R, S, k=top_k)

    entity["mdg_min_10"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=0.1)
    entity["mdg_min_20"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=0.2)
    entity["mdg_min_30"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=0.3)
    entity["mdg_max_10"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.1)
    entity["mdg_max_20"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.2)
    entity["mdg_max_30"] = Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.3)
    entity["u_mmf_5"] = Metrics.u_mmf(B, p_u_coverage=1, p_i_consider=0.05, k=top_k)
    entity["u_mmf_10"] = Metrics.u_mmf(B, p_u_coverage=1, p_i_consider=0.1, k=top_k)
    entity["u_mmf_15"] = Metrics.u_mmf(B, p_u_coverage=1, p_i_consider=0.15, k=top_k)
    entity["u_pf_5"] = Metrics.u_pf(B, p_u_coverage=1, p_i_consider=0.05, k=top_k)
    entity["u_pf_10"] = Metrics.u_pf(B, p_u_coverage=1, p_i_consider=0.1, k=top_k)
    entity["u_pf_15"] = Metrics.u_pf(B, p_u_coverage=1, p_i_consider=0.15, k=top_k)
    return entity


if __name__ == "__main__":
    R = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    S = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    print("Test Metrics Strategy Pattern Design.")
    print(Metrics.dcg_score(R, S))
    print(Metrics.ndcg_score(R, S))

    print(Metrics.map_score(R, S))
    print(Metrics.lrap_score(R, S))

    print(Metrics.dcf_score(S, [[0], [1]]))

    print(Metrics.mdg_score_each_item(R, S))
    print(Metrics.mdg_score(Metrics.mdg_score_each_item(R, S), 0.1))
    print(Metrics.mdg_score(Metrics.mdg_score_each_item(R, S), -0.1))

    print(Metrics.precision_score(R, S, k=2))
    print(Metrics.recall_score(R, S, k=2))
    print(Metrics.f1_score(R, S, k=2))
