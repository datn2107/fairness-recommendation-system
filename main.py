import os
import argparse
import numpy as np
import pandas as pd

from data.converter import DataConverter, InteractionDataConverterStrategy
from data.preprocessor import preprocess_clcrec_result, preprocess_ccfcrec_result, divide_group
from metrics import Metrics
from reranking import ReRanking, WorstOffNumberOfItemAndGroupFairnessReRanking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=30)
    args = parser.parse_args()
    top_k = args.top_k

    dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
    data_converter = DataConverter(InteractionDataConverterStrategy())

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset {args.dataset_name} not found in {args.dataset_dir}."
        )

    if not os.path.exists(
        os.path.join(dataset_dir, "test_cold_interactions_formated.npy")
    ):
        raise FileNotFoundError(
            f"Dataset {args.dataset_name} is not formatted. Please run formatter.py first."
        )

    # Load the dataset
    test_cold_interaction = np.load(
        os.path.join(dataset_dir, "test_cold_interactions_formated.npy")
    )
    test_cold_items = np.load(
        os.path.join(dataset_dir, "test_cold_items.npy"), allow_pickle=True
    ).item()
    tmp_result = np.load(os.path.join(dataset_dir, "ccfcrec_result.npy"))

    R = data_converter.convert_to_relevance_matrix(
        test_cold_interaction,
        rank_relevance=False,
        n_users=tmp_result.shape[0],
        n_items=len(test_cold_items),
    )

    assert np.sum(R > 0) == test_cold_interaction.shape[0]

    # Load the predicted score matrix
    for model_name in ["ccfcrec", "clcrec"]:
        if model_name == "clcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result_formated.npy"))
            S = preprocess_clcrec_result(S)
        elif model_name == "ccfcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result.npy"))
            S = preprocess_ccfcrec_result(S)
        else:
            raise ValueError("Invalid model name.")

        print(model_name.upper())
        print("Precision:", Metrics.precision_score(R, S, k=top_k))
        print("Recall:", Metrics.recall_score(R, S, k=top_k))
        print("NDCG:", Metrics.ndcg_score(R, S, k=top_k))

        B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k=top_k)
        print("MDG_min_10:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.1))
        print("MDG_min_20:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.2))
        print("MDG_min_30:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.3))
        print("MDG_max_10:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.1))
        print("MDG_max_20:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.2))
        print("MDG_max_30:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.3))

        group_items = divide_group(B, group_p=0.7)

        reranking = ReRanking(WorstOffNumberOfItemAndGroupFairnessReRanking())
        W = reranking.optimize(S, k=top_k, i_epsilon=args.epsilon, group_items=group_items)
        S_reranked = reranking.apply_reranking_matrix(S, W)

        print("Precision (reranked):", Metrics.precision_score(R, S_reranked, k=top_k))
        print("Recall (reranked):", Metrics.recall_score(R, S_reranked, k=top_k))
        print("NDCG (reranked):", Metrics.ndcg_score(R, S_reranked, k=top_k))

        print("MDG_min_10 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.1))
        print("MDG_min_20 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.2))
        print("MDG_min_30 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.3))
        print("MDG_max_10 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.1))
        print("MDG_max_20 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.2))
        print("MDG_max_30 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.3))
        print()
