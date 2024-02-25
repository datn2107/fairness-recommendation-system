import numpy as np
from abc import ABC, abstractmethod


def create_rank_matrix(S: np.ndarray) -> np.ndarray:
    """
    Create the rank matrix based on the input matrix S.

    Parameters:
    S (np.ndarray): The predicted score matrix of shape (n_users, n_items).

    Returns:
    np.ndarray: The matrix indicate the rank of each item for each user (n_users, n_items).
    """
    rank = np.zeros_like(S, dtype=np.int32)
    ind_matrix = np.argsort(S, axis=1)[:, ::-1]
    rank[np.arange(S.shape[0])[:, np.newaxis], ind_matrix] = np.tile(
        np.arange(1, S.shape[1] + 1), (S.shape[0], 1)
    )

    return rank


def create_top_item_matrix(S: np.ndarray) -> np.ndarray:
    """
    Calculate the user rank matrix based on the input matrix S.

    Parameters:
    S (np.ndarray): The predicted score matrix of shape (n_users, n_items).

    Returns:
    np.ndarray: The matrix indicate top items for each user (n_users, n_items).
    """
    return np.argsort(S, axis=1)[:, ::-1]


class DataConversionStrategy(ABC):
    @abstractmethod
    def convert_to_relevance_matrix(
        self, data: np.ndarray, k: int = 30, rank_relevance: bool = False
    ) -> np.ndarray:
        """
        Convert the rank matrix to relevance matrix based on the input relevance matrix R.

        Parameters:
        data (np.ndarray): The matrix has shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30. (Only use for binary relevance score)
        rank_relevance (bool): Whether to use the rank or binary as relevance score. Default is False.

        Returns:
        np.ndarray: The relevance score matrix of shape (n_users, n_items).
        """
        raise NotImplementedError


class RankMatrixConversionStrategy(DataConversionStrategy):
    def convert_to_relevance_matrix(
        self, data: np.ndarray, k: int = 30, rank_relevance: bool = False
    ) -> np.ndarray:
        """
        Convert the rank matrix to relevance matrix based on the input relevance matrix R.

        Parameters:
        data (np.ndarray): The matrix indicate the rank of each item for each user (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30. (Only use for binary relevance score)
        rank_relevance (bool): Whether to use the rank or binary as relevance score. Default is False.

        Returns:
        np.ndarray: The relevance score matrix of shape (n_users, n_items).
        """
        relevance_matrix = np.zeros_like(data, dtype=np.float32)

        if rank_relevance:
            relevance_matrix = data
        else:
            relevance_matrix[data <= k] = 1

        return relevance_matrix


class TopItemMatrixConversionStrategy(DataConversionStrategy):
    def convert_to_relevance_matrix(
        self, data: np.ndarray, k: int = 30, rank_relevance: bool = False
    ) -> np.ndarray:
        """
        Convert the top item matrix to relevance matrix based on the input relevance matrix R.

        Parameters:
        data (np.ndarray): The matrix indicate top items for each user (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30. (Only use for binary relevance score)
        rank_relevance (bool): Whether to use the rank or binary as relevance score. Default is False.

        Returns:
        np.ndarray: The relevance score matrix of shape (n_users, n_items).
        """
        relevance_matrix = np.zeros_like(data, dtype=np.float32)
        n_item = data.shape[1]

        ind = np.arange(relevance_matrix.shape[0])[:, np.newaxis]
        if rank_relevance:
            relevance_matrix[ind, data] = n_item - np.tile(
                np.arange(n_item), (relevance_matrix.shape[0], 1)
            )
        else:
            relevance_matrix[ind, data[:, :k]] = 1

        return relevance_matrix


class DataConversionContext:
    """
    Strategy Pattern Design for Data Conversion.
    """
    DataConvertStrategy = None

    def __init__(self, strategy: DataConversionStrategy = None):
        self.DataConvertStrategy = strategy

    def set_strategy(self, strategy: DataConversionStrategy):
        self.DataConvertStrategy = strategy

    def convert_to_relevance_matrix(
        self, data: np.ndarray, k: int = 30, rank_relevance: bool = False
    ) -> np.ndarray:
        return self.DataConvertStrategy.convert_to_relevance_matrix(
            data, k, rank_relevance
        )


if __name__ == '__main__':
    print("Test Data Conversion Strategy Pattern Design.")

    S = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    rank_matrix = create_rank_matrix(S)
    top_item_matrix = create_top_item_matrix(S)

    print(rank_matrix)
    print(top_item_matrix)

    context = DataConversionContext(RankMatrixConversionStrategy())
    print(context.convert_to_relevance_matrix(rank_matrix, k=2, rank_relevance=True))

    context.set_strategy(TopItemMatrixConversionStrategy())
    print(context.convert_to_relevance_matrix(top_item_matrix, k=2, rank_relevance=True))

    context.set_strategy(RankMatrixConversionStrategy())
    print(context.convert_to_relevance_matrix(rank_matrix, k=2, rank_relevance=False))

    context.set_strategy(TopItemMatrixConversionStrategy())
    print(context.convert_to_relevance_matrix(top_item_matrix, k=2, rank_relevance=False))