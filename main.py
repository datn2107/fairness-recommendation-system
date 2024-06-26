import os
import argparse
import numpy as np
import pandas as pd

from data.converter import DataConverter, InteractionDataConverterStrategy
from data.preprocessor import (
    preprocess_clcrec_result,
    preprocess_ccfcrec_result,
    divide_group,
)
from metrics import get_metric
from reranking.base import ReRanking
from reranking.fractory import ReRankingStrategyFractory
from reranking.u_mmf import get_item_provider_mapper


def load_result(model_name, dataset_dir):
    S = np.load(os.path.join(dataset_dir, f"{model_name}_result.npy"))
    if model_name == "clcrec":
        S = preprocess_clcrec_result(S)
    elif model_name == "ccfcrec":
        S = preprocess_ccfcrec_result(S)

    return S


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--result-path", type=str, default="results.csv")
    parser.add_argument(
        "--reranked_output_path", type=str, default="reranked_result_binary.npy"
    )
    parser.add_argument("--reranking", action="store_true")
    parser.add_argument("--epsilon", type=float, default=30)
    parser.add_argument("--group-p", type=float, default=0.7)
    parser.add_argument("--strategy-name", type=str, default="ummf")
    parser.add_argument("--strategy-type", type=str, default="SAT")
    args = parser.parse_args()

    top_k = args.top_k
    save_dir = os.path.dirname(args.result_path)
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

    # Convert the interaction data to relevance matrix
    R = data_converter.convert_to_relevance_matrix(
        test_cold_interaction,
        rank_relevance=False,
        n_users=tmp_result.shape[0],
        n_items=len(test_cold_items),
    )

    if os.path.exists(args.result_path):
        print("Warning: The result file already exists. It will be overwritten.")
    result = pd.DataFrame()

    # Compute the predicted score matrix
    for model_name in ["clcrec"]:
        S = load_result(model_name, dataset_dir)
        item_provider_mapper = get_item_provider_mapper(S, p=args.epsilon)

        # Before applying reranking
        B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k=top_k)
        entity = get_metric(R, S, B, top_k, item_provider_mapper)

        # Save the result
        df_entity = pd.DataFrame([entity])
        result = pd.concat([result, df_entity], ignore_index=True)

        np.save(os.path.join(save_dir, f"{model_name}_result_binary.npy"), B)
        print(model_name.upper())
        print(entity)

        # Apply reranking
        if args.reranking:
            reranking = ReRanking(ReRankingStrategyFractory.create(args.strategy_name))
            if args.strategy_name == "ummf":
                W, time = reranking.optimize(
                    S, k=top_k, p=args.epsilon, interactions=test_cold_interaction
                )
            else:
                W, time = reranking.optimize(
                    S, k=top_k, epsilon=args.epsilon, strategy_type=args.strategy_type
                )
            # S_reranked = reranking.apply_reranking_matrix(S, W)

            entity = get_metric(R, S, W, top_k, item_provider_mapper)
            entity["time"] = time

            # Save the result
            df_entity = pd.DataFrame([entity])
            result = pd.concat([result, df_entity], ignore_index=True)

            dir = os.path.dirname(args.reranked_output_path)
            basename = os.path.basename(args.reranked_output_path)

            np.save(os.path.join(dir, f"{model_name}_{basename}"), W)
            print(model_name.upper() + " RERANKED")
            print(entity)

    result.to_csv(args.result_path, index=False)
