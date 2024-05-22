import os
import argparse
import numpy as np
import pandas as pd

from data.converter import DataConverter, InteractionDataConverterStrategy
from data.preprocessor import preprocess_clcrec_result, preprocess_ccfcrec_result, divide_group
from metrics import Metrics
from reranking import ReRanking, ReRankingStrategyFractory


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
    entity["u_mmf_30"] = Metrics.u_mmf(R, S, p=0.3, percentage=10, k=top_k)
    entity["u_mmf_50"] = Metrics.u_mmf(R, S, p=0.5, percentage=10, k=top_k)
    entity["u_mmf_70"] = Metrics.u_mmf(R, S, p=0.7, percentage=10, k=top_k)
    return entity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--result-path", type=str, default="results.csv")
    parser.add_argument("--reranking", action="store_true")
    parser.add_argument("--epsilon", type=float, default=30)
    parser.add_argument("--group-p", type=float, default=0.7)
    parser.add_argument("--strategy-name", type=str, default="worst_off_number_of_item_or_tools")
    parser.add_argument("--strategy-type", type=str, default="SAT")
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

    if os.path.exists(args.result_path):
        print("Warning: The result file already exists. It will be overwritten.")
    result = pd.DataFrame()

    # Load the predicted score matrix
    for model_name in ["clcrec", "ccfcrec"]:
        if model_name == "clcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result_formated.npy"))
            S = preprocess_clcrec_result(S)
        elif model_name == "ccfcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result.npy"))
            S = preprocess_ccfcrec_result(S)
        else:
            raise ValueError("Invalid model name.")

        B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k=top_k)

        entity = get_metric(R, S, B, top_k)

        df_entity = pd.DataFrame([entity])
        result = pd.concat([result, df_entity], ignore_index=True)
        np.save(os.path.join(dataset_dir, f"{model_name}_result_binary.npy"), B)
        print(model_name.upper())
        print(entity)

        if args.reranking and model_name == "clcrec":
            # group_items = divide_group(B, group_p=0.7)

            reranking = ReRanking(ReRankingStrategyFractory.create(args.strategy_name)(strategy_type=args.strategy_type))
            W = reranking.optimize(S, k=top_k, epsilon=args.epsilon)
            S_reranked = reranking.apply_reranking_matrix(S, W)

            entity = get_metric(R, S_reranked, W, top_k)

            df_entity = pd.DataFrame([entity])
            result = pd.concat([result, df_entity], ignore_index=True)
            np.save(os.path.join(dataset_dir, f"{model_name}_result_reranked_binary.npy"), Ư)
            print(model_name.upper() + " RERANKED")
            print(entity)

    result.to_csv(args.result_path, index=False)
