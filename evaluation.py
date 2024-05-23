import os
import argparse
import numpy as np

from metrics import get_metric
from data.converter import DataConverter, InteractionDataConverterStrategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", type=str)
    parser.add_argument("--groundtruth-interaction-path", type=str)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    matrix = np.load(args.matrix_path)
    groundtruth_interaction = np.load(args.groundtruth_interaction_path)

    data_converter = DataConverter(InteractionDataConverterStrategy())
    R = data_converter.convert_to_relevance_matrix(
        groundtruth_interaction,
        rank_relevance=False,
        n_users=matrix.shape[0],
        n_items=matrix.shape[1],
    )
    S = matrix
    B = DataConverter.convert_score_matrix_to_rank_matrix(S)

    print(get_metric(R, B, B, args.top_k))
