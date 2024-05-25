import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import trange

from reranking.base import ReRankingStrategy


def get_item_provider_mapper(S: np.ndarray, p=0.05):
    items_count = np.sum(S, axis=0)
    items_id_sorted = np.argsort(items_count)

    item_interval = int(len(items_id_sorted) * p)
    provider_id = np.zeros(len(items_id_sorted), dtype=np.int32)
    for i, s in enumerate(range(0, len(items_id_sorted), item_interval)):
        provider_id[items_id_sorted[s : s + item_interval]] = i
    return provider_id


class UMMFReRanking(ReRankingStrategy):
    def relabel_provider(self, interactions, preference_scores=None, p=0.05):
        if preference_scores is not None:
            provider_id = get_item_provider_mapper(preference_scores, p)
            interactions[:, 3] = provider_id[interactions[:, 1]]

        interactions[:, 3] = np.unique(interactions[:, 3], return_inverse=True)[1]
        return interactions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cpu_layer(self, ordered_tilde_dual, rho, lambd):
        m = len(rho)
        answer = cp.Variable(m)
        objective = cp.Minimize(
            cp.sum_squares(
                cp.multiply(rho, answer) - cp.multiply(rho, ordered_tilde_dual)
            )
        )

        constraints = []
        for i in range(1, m + 1):
            constraints += [cp.sum(cp.multiply(rho[:i], answer[:i])) >= -lambd]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return answer.value

    def compute_next_dual(self, eta, rho, dual, gradient, lambd):
        tilde_dual = dual - eta * gradient / rho / rho
        order = np.argsort(tilde_dual * rho)
        ordered_tilde_dual = tilde_dual[order]
        ordered_next_dual = self.cpu_layer(ordered_tilde_dual, rho[order], lambd)
        return ordered_next_dual[order.argsort()]

    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        lambd = kargs.get("lambd", 0.1)
        alpha = kargs.get("alpha", 0.1)
        eta = kargs.get("eta", 1e-3)
        interactions = kargs.get("interactions")
        p = kargs.get("p", 0.05)

        n_users, n_items = S.shape[:2]
        interactions = self.relabel_provider(interactions, S, p)

        n_providers = len(np.unique(interactions[:, 3]))
        providers_cnt = np.unique(interactions[:, 3], return_counts=True)[1]

        rho = (1 + 1 / n_providers) * providers_cnt / np.sum(providers_cnt)
        UI_matrix = S[np.arange(n_users)]

        # normalize user-item perference score to [0,1]
        UI_matrix = self.sigmoid(UI_matrix)

        # set item-provider matrix
        set_item_provider = set(zip(interactions[:, 1], interactions[:, 3]))
        item2provider = {x: y for x, y in set_item_provider}

        A = np.zeros((n_items, n_providers))
        iid2pid = []
        for i in range(n_items):
            iid2pid.append(item2provider[i])
            A[i, item2provider[i]] = 1


        result_x = []

        mu_t = np.zeros(n_providers)
        B_t = n_users * k * rho
        sum_dual = 0
        eta_t = eta / np.sqrt(n_users)
        gradient_cusum = np.zeros(n_providers)
        for t in trange(n_users):
            x_title = UI_matrix[t, :] - np.matmul(A, mu_t)
            mask = np.matmul(A, (B_t > 0).astype(np.float32))

            mask = (1.0 - mask) * -10000.0
            x = np.argsort(x_title + mask, axis=-1)[::-1]
            x_allocation = x[:k]
            re_allocation = np.argsort(UI_matrix[t, x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            result_x.append(x_allocation)

            B_t = B_t - np.sum(A[x_allocation], axis=0, keepdims=False)
            gradient = -np.mean(A[x_allocation], axis=0, keepdims=False) + B_t / (
                n_users * k
            )

            gradient = alpha * gradient + (1 - alpha) * gradient_cusum
            gradient_cusum = gradient
            mu_t = self.compute_next_dual(eta_t, rho, mu_t, gradient, lambd)
            sum_dual += mu_t


        W = np.zeros((n_users, n_items))
        for i in range(n_users):
            W[i, result_x[i]] = 1

        return W
