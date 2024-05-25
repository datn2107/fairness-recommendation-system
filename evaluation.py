import os
import argparse
import numpy as np

from metrics import get_metric
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

    for model in ["clcrec", "ccfcrec"]:
        S = load_result(model, dataset_dir)
        B = np.load(os.path.join(dataset_dir, f"{model}_result_binary.npy"))
        data_converter = DataConverter(InteractionDataConverterStrategy())
        R = data_converter.convert_to_relevance_matrix(
            interactions,
            rank_relevance=False,
            n_users=B.shape[0],
            n_items=B.shape[1],
        )

        for method in args.methods:
            B = np.load(
                os.path.join(dataset_dir, f"{model}_{method}_result_binary.npy")
            )
            metric = get_metric(R, method_result, B, args.top_k)
            print(f"{model}_{method}: {metric}")
