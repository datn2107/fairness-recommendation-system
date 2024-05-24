import os
import argparse
import torch
import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import trange

from data.preprocessor import preprocess_clcrec_result
from data.converter import DataConverter, InteractionDataConverterStrategy
from metrics import get_metric


def relabel_provider(interactions, preference_scores=None, p=0.05):
    if preference_scores is not None:
        items_count = np.sum(preference_scores, axis=0)
        items_id_sorted = np.argsort(items_count)

        item_interval = int(len(items_id_sorted) * p)
        provider_id = np.zeros(len(items_id_sorted))
        for i, s in enumerate(range(0, len(items_id_sorted), item_interval)):
            provider_id[items_id_sorted[s : s + item_interval]] = i
        interactions[:, 3] = provider_id[interactions[:, 1]]

    interactions[:, 3] = np.unique(interactions[:, 3], return_inverse=True)[1]
    return interactions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def CPU_layer(ordered_tilde_dual, rho, lambd):
    m = len(rho)
    answer = cp.Variable(m)
    objective = cp.Minimize(
        cp.sum_squares(cp.multiply(rho, answer) - cp.multiply(rho, ordered_tilde_dual))
    )
    constraints = []
    for i in range(1, m + 1):
        constraints += [cp.sum(cp.multiply(rho[:i], answer[:i])) >= -lambd]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return answer.value


def compute_next_dual(eta, rho, dual, gradient, lambd):
    tilde_dual = dual - eta * gradient / rho / rho
    order = np.argsort(tilde_dual * rho)
    ordered_tilde_dual = tilde_dual[order]
    ordered_next_dual = CPU_layer(ordered_tilde_dual, rho[order], lambd)
    return ordered_next_dual[order.argsort()]


def p_mmf_cpu(
    trained_preference_scores, cold_test_interactions, top_k, lambd, alpha, eta
):
    n_users, n_items = trained_preference_scores.shape[:2]
    T = n_users

    # create dataframe contain 4 columns: uid, iid, time, provider from cold_test_interactions
    datas = pd.DataFrame(
        {
            "uid": cold_test_interactions[:, 0],
            "iid": cold_test_interactions[:, 1],
            "time": cold_test_interactions[:, 2],
            "provider": cold_test_interactions[:, 3],
        }
    )
    uid_field, iid_field, time_field, provider_field = datas.columns

    n_providers = len(datas[provider_field].unique())
    providerLen = np.array(datas.groupby(provider_field).size().values)
    rho = (1 + 1 / n_providers) * providerLen / np.sum(providerLen)
    datas.sort_values(by=[time_field], ascending=True, inplace=True)
    batch_size = int(len(datas) // T)

    data_val = np.array(datas[uid_field].values[-batch_size * T :]).astype(np.int32)
    UI_matrix = trained_preference_scores[data_val]

    # normalize user-item perference score to [0,1]
    UI_matrix = sigmoid(UI_matrix)
    tmp = datas[[iid_field, provider_field]].drop_duplicates()
    item2provider = {x: y for x, y in zip(tmp[iid_field], tmp[provider_field])}

    # A is item-provider matrix
    A = np.zeros((n_items, n_providers))
    iid2pid = []
    for i in range(n_items):
        iid2pid.append(item2provider[i])
        A[i, item2provider[i]] = 1

    K = top_k

    print(n_users, n_items)
    prev_result_x = None
    for b in trange(batch_size):
        min_index = b * T
        max_index = (b + 1) * T
        batch_UI = UI_matrix[min_index:max_index, :]

        mu_t = np.zeros(n_providers)
        B_t = T * K * rho
        # print(np.float32(B_t>0))
        sum_dual = 0
        result_x = []
        eta = eta / np.sqrt(T)
        gradient_cusum = np.zeros(n_providers)
        for t in range(T):
            alpha = alpha
            x_title = batch_UI[t, :] - np.matmul(A, mu_t)
            mask = np.matmul(A, (B_t > 0).astype(np.float32))

            mask = (1.0 - mask) * -10000.0
            x = np.argsort(x_title + mask, axis=-1)[::-1]
            x_allocation = x[:K]
            re_allocation = np.argsort(batch_UI[t, x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            result_x.append(x_allocation)
            B_t = B_t - np.sum(A[x_allocation], axis=0, keepdims=False)
            gradient = -np.mean(A[x_allocation], axis=0, keepdims=False) + B_t / (T * K)

            gradient = alpha * gradient + (1 - alpha) * gradient_cusum
            gradient_cusum = gradient
            mu_t = compute_next_dual(eta, rho, mu_t, gradient, lambd)
            sum_dual += mu_t

    return result_x


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_dir", type=str, required=True)
    args = argparser.parse_args()

    dataset_dir = args.dataset_dir
    test_cold_interaction = np.load(
        os.path.join(dataset_dir, "test_cold_interactions_formatted.npy")
    )
    test_cold_items = np.load(
        os.path.join(dataset_dir, "test_cold_items.npy"), allow_pickle=True
    ).item()
    clcrec_result = np.load(os.path.join(dataset_dir, "clcrec_result_formated.npy"))

    converter = DataConverter(InteractionDataConverterStrategy())
    R = converter.convert_to_relevance_matrix(
        test_cold_interaction,
        rank_relevance=False,
        n_users=clcrec_result.shape[0],
        n_items=clcrec_result.shape[1],
    )

    clcrec_result = preprocess_clcrec_result(clcrec_result)
    test_cold_interaction = relabel_provider(test_cold_interaction, clcrec_result)
    result = p_mmf_cpu(clcrec_result, test_cold_interaction, 30, 0.1, 0.1, 1e-3)
    print(result.shape)

    metric = get_metric(R, result, result, 30)
    print(metric)
