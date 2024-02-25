import numpy as np
import sklearn.metrics as metrics


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def dcg_score(cls, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the discounted cumulative gain (DCG) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCG score.
        """
        return metrics.dcg_score(R, S, k=k)

    @staticmethod
    def ndcg_score(self, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the normalized discounted cumulative gain (NDCG) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The NDCG score.
        """
        return metrics.ndcg_score(R, S, k=k)

    @staticmethod
    def map_score(self, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean average precision (MAP) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MAP score.
        """
        pass

    @staticmethod
    def lrap_score(self, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the label-ranking average precision (LRAP) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The LRAP score.
        """
        pass

    @staticmethod
    def dcf_score(self, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the deviation from producer fairness (DCF) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCF score.
        """
        pass

    @staticmethod
    def mdg_score(self, R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean discounted gain (MDG) score.

        Parameters:
        R (np.ndarray): The relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MDG score.
        """
        pass


if __name__ == "__main__":
    R = np.array([[0, 0, 0, 0, 2], [0, 1, 0, 1, 1]])
    S = np.array([[0, 0.05, 0.1, 0.2, 0.3]])
    # S = np.array([[1, 3, 2, 4, 5]])

    print(metrics.dcg_score(R, S, k=2))
    print(metrics.ndcg_score(R, S, k=2))
