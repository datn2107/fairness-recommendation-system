import numpy as np

def normalize(S: np.ndarray, base_val: float = 0) -> np.ndarray:
    """
    Normalize the score matrix.

    Parameters:
    S (np.ndarray): The score matrix of shape (n_users, n_items).
    base_val (float): The base value to add to the score matrix to avoid value 0.

    Returns:
    The normalized score matrix.
    """
    S = S - np.min(S) + base_val
    S = S / (np.max(S) -  np.min(S))

    return S


def normalize_by_row(S: np.ndarray, base_val: float = 0) -> np.ndarray:
    """
    Normalize the score matrix by row.

    Parameters:
    S (np.ndarray): The score matrix of shape (n_users, n_items).
    base_val (float): The base value to add to the score matrix to avoid value 0.

    Returns:
    The normalized score matrix.
    """
    S = S - np.min(S, axis=1)[:, None] + base_val
    S = S / (np.max(S, axis=1) - np.min(S, axis=1))[:, None]

    return S


def preprocess_clcrec_result(S: np.ndarray) -> np.ndarray:
    """
    Preprocess the CLCRec result.

    Parameters:
    S (np.ndarray): The predicted score matrix of shape (n_users, n_items).

    Returns:
    The preprocessed score matrix.
    """
    S = normalize(S)

    return S


def preprocess_ccfcrec_result(S: np.ndarray) -> np.ndarray:
    """
    Preprocess the CCF-Rec result.

    Parameters:
    S (np.ndarray): The predicted score matrix of shape (n_users, n_items).

    Returns:
    The preprocessed score matrix.
    """
    ranking = np.argsort(S, axis=0)
    ranking_col_idx = np.tile(np.arange(S.shape[1]), (S.shape[0], 1))

    S[ranking, ranking_col_idx] = np.tile(np.arange(S.shape[0]), (S.shape[1], 1)).T
    S = normalize(S, 0.01)

    return S


def divide_group(B: np.ndarray, group_p: float) -> np.ndarray:
    """
    Divide the score matrix into two groups.

    Parameters:
    B (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
    group_p (float): The percentage of the group. The value should be in the range (0, 1).

    Returns:
    The divided score matrix.
    """

    n_items = B.shape[1]
    items_cnt = np.sum(B, axis=0)
    items_idx = np.argsort(items_cnt)

    group_size = int(n_items * group_p)
    group_items = items_idx[:group_size], items_idx[group_size:]

    return group_items


if __name__ == '__main__':
    # Test
    S = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.3], [0.3, 0.1, 0.2], [0.4, 0.5, 0.6]])

    print(S[[0, 1, 2, 3], [0, 1, 2, 0]])
    print(preprocess_ccfcrec_result(S))
