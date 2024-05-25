import os
import argparse
import numpy as np
import pandas as pd

from metrics import get_metric
from reranking.u_mmf import get_item_provider_mapper
from data.converter import DataConverter, InteractionDataConverterStrategy
from data.preprocessor import preprocess_clcrec_result, preprocess_ccfcrec_result


def load_result(model_name, dataset_dir):
    S = np.load(os.path.join(dataset_dir, f"{model_name}_result.npy"))
    if model_name == "clcrec":
        S = preprocess_clcrec_result(S)
    elif model_name == "ccfcrec":
        S = preprocess_ccfcrec_result(S)

    return S


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["PDLP", "MIP", "UMMF"]
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    interactions = np.load(
        os.path.join(dataset_dir, "test_cold_interactions_provider_formated.npy")
    )
    test_cold_items = np.load(
        os.path.join(dataset_dir, "test_cold_items.npy"), allow_pickle=True
    ).item()

    data_converter = DataConverter(InteractionDataConverterStrategy())
    result = pd.DataFrame()
    for model in ["clcrec", "ccfcrec"]:
        S = load_result(model, dataset_dir)
        B = data_converter.convert_score_matrix_to_relevance_matrix(S, k=args.top_k)
        R = data_converter.convert_to_relevance_matrix(
            interactions,
            rank_relevance=False,
            n_users=B.shape[0],
            n_items=B.shape[1],
        )
        item_provider_mapper = get_item_provider_mapper(S, p=0.05)

        entity = get_metric(R, S, B, args.top_k, item_provider_mapper)
        entity["model"] = model
        result = pd.concat([result, entity])

        for method in args.methods:
            B = np.load(
                os.path.join(dataset_dir, f"{model}_{method}_result_binary.npy")
            )
            entity = get_metric(R, S, B, args.top_k, item_provider_mapper)
            entity["model"] = model + "_" + method
            result = pd.concat([result, entity])

    result.to_csv(os.path.join(dataset_dir, "final_results.csv"), index=False)
    print(result)
