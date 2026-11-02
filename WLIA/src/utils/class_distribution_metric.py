import numpy as np
from scipy.stats import entropy

def class_distribution_metrics(class_counts, total_classes=10):
    counts = np.array(class_counts)
    present = counts > 0
    k = present.sum()
    total = counts.sum()

    if k == 0:
        return {
            'effective_classes': 0,
            'normalized_entropy': 0,
            'coverage': 0,
            'gini': 0
        }

    p = counts[present] / total

    # Metrics
    enc = 1.0 / np.sum(p ** 2)
    norm_entropy = entropy(p) / np.log(k)
    coverage = k / total_classes
    gini = 1 - np.sum(p ** 2)

    return {
        'effective_classes': enc,
        'normalized_entropy': norm_entropy,
        'coverage': coverage,
        'gini': gini
    }

def client_wise_distribution(data_dist):
    for cid in range(data_dist.shape[0]):
        tmp_dist = data_dist[cid, :]

        temp_dist_result = class_distribution_metrics(tmp_dist, total_classes=data_dist.shape[1])

        print(f'Client {cid}: {tmp_dist.astype(int)}')
        for key, val in temp_dist_result.items():
            print(f'{key}: {val}')

        print('\n')