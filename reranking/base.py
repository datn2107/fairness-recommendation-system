import numpy as np
from time import time
from typing import Any
from abc import ABC, abstractmethod


class ReRankingStrategy(ABC):
    @abstractmethod
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> Any:
        """
        Optimize the recommend result.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        The optimized result. The return type is depend on the strategy.
            W: The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.
            group_items_cnt: (Optional) The number of items in each group.
        """
        raise NotImplementedError


class ReRanking:
    def __init__(self, strategy: ReRankingStrategy = None):
        self.strategy = strategy

    def set_strategy(self, strategy: ReRankingStrategy):
        self.strategy = strategy

    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        start_time = time()
        W = self.strategy.optimize(S, k, *args, **kargs)
        total_time = time() - start_time
        print(f"Optimize time: {total_time}")
        return W, total_time

    def apply_reranking_matrix(self, S: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Apply the reranking matrix to the score matrix.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        W (np.ndarray): The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.

        Returns:
        np.ndarray: The reranked score matrix of shape (n_users, n_items).
        """
        return (S + 1e-6) * B
