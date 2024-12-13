# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:29:07 2024

@author: 531725ns
"""

import numpy as np


def generate_band_buckets(band_index, signatures, r, sep):
    n, N = signatures.shape
    buckets = {}

    for j in range(N):
        key = sep.join(map(str, signatures[(band_index * r):((band_index + 1) * r), j]))
        buckets.setdefault(key, set()).add(j)

    return buckets


def construct_candidate_pairs(buckets_per_band, signatures):
    n, N = signatures.shape
    same_bucket_counts = np.zeros((N, N))

    for band_map in buckets_per_band:
        for bucket in band_map.values():
            for i in bucket:
                for j in bucket:
                    if i == j:
                        continue

                    same_bucket_counts[i, j] = 1

    candidate_pair_matrix = same_bucket_counts

    return candidate_pair_matrix


def lsh(signatures, b, r, sep='-'):
    
    if len(signatures.shape) != 2:
        raise ValueError("Invalid signatures shape")

    buckets_per_band = [generate_band_buckets(band, signatures, r, sep) for band in range(b)]
    return construct_candidate_pairs(buckets_per_band, signatures)


