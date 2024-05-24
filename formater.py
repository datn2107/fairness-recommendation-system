import os
import argparse
import numpy as np


def map_item_id(interactions, cold_items):
    items_id = {cold_item: idx for idx, cold_item in enumerate(cold_items)}

    return items_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='dataset name')
    parser.add_argument('--dataset-dir', default="datasets", type=str, help='data directory')
    parser.add_argument('--remove-old-files', default=False, action='store_true', help='remove old files')
    args = parser.parse_args()

    dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    test_cold_interaction = np.load(os.path.join(dataset_dir, "test_cold_interactions.npy"), allow_pickle=True)
    test_cold_interaction_provider = np.load(os.path.join(dataset_dir, "test_cold_interactions_provider.npy"), allow_pickle=True)
    cold_items = list(np.load(os.path.join(dataset_dir, "test_cold_items.npy"), allow_pickle=True).item())


    items_id = map_item_id(test_cold_interaction, cold_items)
    cold_interactions_formated = np.array([[data[0], items_id[data[1]]] for data in test_cold_interaction])
    cold_interactions_provider_formated = np.array([[data[0], items_id[data[1]], data[2], data[3]] for data in test_cold_interaction_provider])

    np.save(os.path.join(dataset_dir, "test_cold_interactions_formated.npy"), cold_interactions_formated)
    np.save(os.path.join(dataset_dir, "test_cold_interactions_provider_formated.npy"), cold_interactions_provider_formated)

    clcrec_result = np.load(os.path.join(dataset_dir, "clcrec_result.npy"))
    if clcrec_result.shape[1] != len(cold_items):
        clcrec_result_formated = clcrec_result[:, cold_items]
        np.save(os.path.join(dataset_dir, "clcrec_result_formated.npy"), clcrec_result_formated)
        print(clcrec_result_formated.shape)

    if args.remove_old_files:
        os.remove(os.path.join(dataset_dir, "test_cold_interactions.npy"))
        os.remove(os.path.join(dataset_dir, "clcrec_result.npy"))
